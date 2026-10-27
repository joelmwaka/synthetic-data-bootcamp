"""Shared workflow for night / headlight synthesis (notebook, app, batch worker).

Pipeline per frame::

    analyze_frame  → SceneAnalysis (geometry cached on disk; lighting recomputed)
    generate_image → Klein instruct edit of the photo or the physics prior
    score_candidate→ physics metrics (+ optional VLM critique) → gates / rank
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from aieng.syn_data.image.data.frames import CAP_CALIBRATION, Frame
from aieng.syn_data.image.generate import night


DEFAULT_WORK_SIZE = (768, 432)

DEFAULT_GATES: dict[str, float] = {
    "min_beam_corr": 0.6,
    "max_phantom_light": 5e-4,
    "min_lit_structure": 0.25,
    "min_mean_luma": 0.03,
    "max_mean_luma": 0.25,
    "min_vlm_overall": 6.0,
}

PRESET_DESCRIPTIONS = {
    "naive": "What a user types first.",
    "named_source": "Physically accurate text that names the headlights (tends to draw them).",
    "effect_only": "Same physics, describes only where the light lands.",
    "hybrid_prior": "Effect-only prompt that refines the physics render.",
}


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars / NaN / tuples into strict-JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- scene analysis ------------------------------------------------------------


@dataclass
class SceneAnalysis:
    """Everything the night workflow knows about one frame."""

    frame_id: str
    work_size: tuple[int, int]
    camera: night.RailCamera
    work_camera: night.RailCamera
    geometry: night.SceneGeometry
    disparity: np.ndarray
    rails: np.ndarray
    rig: night.HeadlightRig
    look: night.NightLook
    irradiance: np.ndarray
    prior: Image.Image
    facts: dict[str, Any]
    gauge: list[dict[str, float]]

    @property
    def box_scale(self) -> tuple[float, float]:
        """Factor from original-image label coordinates to the work grid."""
        return (
            self.work_size[0] / self.camera.width,
            self.work_size[1] / self.camera.height,
        )

    def meta(self) -> dict[str, Any]:
        """JSON-safe summary for run records."""
        return dict(
            to_jsonable(
                {
                    "work_size": list(self.work_size),
                    "camera": asdict(self.camera),
                    "depth_fit": self.geometry.depth_fit,
                    "gauge_check": self.gauge,
                    "rig": asdict(self.rig),
                    "look": asdict(self.look),
                    "facts": self.facts,
                },
            ),
        )


class GeometryCache:
    """On-disk cache of per-frame geometry (depth + segmentation are the slow part)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @staticmethod
    def key(frame: Frame, work_size: tuple[int, int], model_key: str) -> str:
        """Cache key from frame identity, analysis grid, and model ids."""
        stat = frame.image_path.stat()
        raw = f"{frame.dataset}|{frame.frame_id}|{stat.st_size}|{work_size}|{model_key}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, frame: Frame, key: str) -> Path:
        return self.root / frame.frame_id / f"{key}.npz"

    def load(
        self, frame: Frame, key: str
    ) -> tuple[night.SceneGeometry, np.ndarray] | None:
        """Return cached ``(geometry, disparity)`` or ``None``."""
        path = self._path(frame, key)
        if not path.is_file():
            return None
        with np.load(path) as data:
            fit = json.loads(str(data["depth_fit"]))
            geometry = night.SceneGeometry(
                points=data["points"].astype(np.float64),
                normals=data["normals"].astype(np.float64),
                ground=data["ground"],
                sky=data["sky"],
                depth_fit={
                    k: float("nan") if v is None else float(v) for k, v in fit.items()
                },
            )
            return geometry, data["disparity"]

    def save(
        self,
        frame: Frame,
        key: str,
        geometry: night.SceneGeometry,
        disparity: np.ndarray,
    ) -> Path:
        """Write ``(geometry, disparity)`` atomically."""
        path = self._path(frame, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.npz")
        np.savez_compressed(
            tmp,
            points=geometry.points.astype(np.float32),
            normals=geometry.normals.astype(np.float32),
            ground=geometry.ground,
            sky=geometry.sky,
            disparity=disparity.astype(np.float32),
            depth_fit=json.dumps(to_jsonable(geometry.depth_fit)),
        )
        tmp.replace(path)
        return path


ModelLoader = Callable[[], tuple[Any, Any]]


def model_key(cfg: Any) -> str:
    """Geometry-cache key part identifying the depth and segmentation models."""
    cond = cfg.conditioning
    return f"{cond.depth.model_id}|{cond.segmentation.model_name}"


def load_conditioning_models(cfg: Any) -> tuple[Any, Any]:
    """Load the configured depth estimator and segmenter."""
    from aieng.syn_data.image.generate.conditioning import (  # noqa: PLC0415
        DepthEstimator,
        Segmenter,
    )

    return DepthEstimator.from_config(cfg), Segmenter.from_config(cfg)


def compute_geometry(
    frame: Frame,
    camera: night.RailCamera,
    work_size: tuple[int, int],
    *,
    depth_model: Any,
    segmenter: Any,
) -> tuple[night.SceneGeometry, np.ndarray]:
    """Run depth + segmentation and build calibrated geometry for a frame."""
    work_image = frame.load_image().resize(work_size, Image.Resampling.LANCZOS)
    disparity = depth_model.predict(work_image).depth_map
    segmentation = segmenter.predict(work_image)
    id2label = segmenter.model.config.id2label
    sky_ids = [int(k) for k, v in id2label.items() if v == "sky"]
    mask = frame.load_mask()
    track = None
    if mask is not None:
        track = np.asarray(mask.resize(work_size, Image.Resampling.NEAREST)) > 127
    geometry = night.build_scene_geometry(
        camera.resized(*work_size),
        disparity,
        track,
        sky_mask=np.isin(segmentation.label_map, sky_ids),
    )
    return geometry, disparity


def light_scene(
    frame: Frame,
    camera: night.RailCamera,
    geometry: night.SceneGeometry,
    work_size: tuple[int, int],
    *,
    rig: night.HeadlightRig,
    look: night.NightLook,
    rails: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Image.Image]:
    """Compute irradiance, the rail mask, and the physics-prior render (fast)."""
    work_camera = camera.resized(*work_size)
    irradiance = night.normalized_irradiance(geometry, rig, work_camera)
    if rails is None:
        rails = night.rail_mask(frame.annotation or {}, camera, work_size)
    prior = night.render_night(
        frame.load_image(), geometry, irradiance, look, rails=rails
    )
    return irradiance, rails, prior


def analyze_frame(
    frame: Frame,
    *,
    work_size: tuple[int, int] = DEFAULT_WORK_SIZE,
    load_models: ModelLoader | None = None,
    cache: GeometryCache | None = None,
    model_key: str = "",
    rig_overrides: dict[str, Any] | None = None,
    look: night.NightLook | None = None,
) -> SceneAnalysis:
    """Build the full scene analysis; models load only on a cache miss.

    Requires a calibrated frame (``calibration`` capability).
    """
    if CAP_CALIBRATION not in frame.capabilities or frame.camera_path is None:
        raise ValueError(
            f"{frame.frame_id} has no camera calibration — the night physics needs it"
        )
    camera = night.RailCamera.from_opencv_yaml(frame.camera_path)
    with Image.open(frame.image_path) as image:
        if image.size != (camera.width, camera.height):
            raise ValueError(
                f"{frame.frame_id}: image {image.size} != calibration {(camera.width, camera.height)}"
            )
    key = GeometryCache.key(frame, work_size, model_key)
    cached = cache.load(frame, key) if cache else None
    if cached is None:
        if load_models is None:
            raise RuntimeError("Geometry not cached and no model loader given")
        depth_model, segmenter = load_models()
        geometry, disparity = compute_geometry(
            frame, camera, work_size, depth_model=depth_model, segmenter=segmenter
        )
        if cache:
            cache.save(frame, key, geometry, disparity)
    else:
        geometry, disparity = cached
    annotation = frame.annotation or {}
    rig = night.HeadlightRig(centre_x_m=night.ego_centre_x(annotation, camera))
    if rig_overrides:
        rig = replace(rig, **rig_overrides)
    look = look or night.NightLook()
    irradiance, rails, prior = light_scene(
        frame, camera, geometry, work_size, rig=rig, look=look
    )
    facts = night.scene_facts(annotation, camera, rig, n_switches=len(frame.boxes))
    return SceneAnalysis(
        frame_id=frame.frame_id,
        work_size=work_size,
        camera=camera,
        work_camera=camera.resized(*work_size),
        geometry=geometry,
        disparity=disparity,
        rails=rails,
        rig=rig,
        look=look,
        irradiance=irradiance,
        prior=prior,
        facts=facts,
        gauge=night.rail_gauge_check(annotation, camera),
    )


# --- prompts and generation ----------------------------------------------------


def prompt_presets(facts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Ablation presets: ``{name: {"input", "slots", "prompt"}}``."""
    presets: dict[str, dict[str, Any]] = {
        "naive": {"input": "photo", "slots": None, "prompt": night.naive_prompt()},
        "named_source": {
            "input": "photo",
            "slots": night.physics_prompt_slots(facts, name_source=True),
        },
        "effect_only": {"input": "photo", "slots": night.physics_prompt_slots(facts)},
        "hybrid_prior": {"input": "prior", "slots": night.prior_refine_slots(facts)},
    }
    for preset in presets.values():
        if preset["slots"] is not None:
            preset["prompt"] = night.compose_prompt(preset["slots"])
    return presets


def generate_image(
    comparer: Any,
    generation_cfg: Any,
    source: Image.Image,
    prompt: str,
    *,
    work_size: tuple[int, int] = DEFAULT_WORK_SIZE,
    seed_offset: int = 0,
) -> Image.Image:
    """Run one Klein instruct edit (``MethodComparer.run_method``) on the work grid."""
    anomaly = OmegaConf.create(
        {"id": "night_headlights", "prompt": prompt, "max_side": max(work_size)}
    )
    result = comparer.run_method(
        "instruct",
        source,
        depth=None,
        segmentation=None,
        generation_cfg=generation_cfg,
        anomaly_cfg=anomaly,
        seed_offset=seed_offset,
    )
    out: Image.Image = result.image.convert("RGB")
    return (
        out
        if out.size == work_size
        else out.resize(work_size, Image.Resampling.LANCZOS)
    )


# --- candidates, scoring, gates ------------------------------------------------


@dataclass
class Candidate:
    """One generated night image plus its evaluation and review state."""

    label: str
    input: str
    prompt: str
    slots: dict[str, str] | None = None
    seed_offset: int = 0
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    metrics: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] | None = None
    physics_score: float = 0.0
    vlm_overall: float | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"
    note: str = ""
    box_visible: list[bool] | None = None
    image_file: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""
        return dict(to_jsonable(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        """Rebuild from :meth:`to_dict` output (unknown keys ignored)."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def visible_flags(self) -> list[bool]:
        """Per-box visibility: reviewer override, else the metric flag."""
        boxes = self.metrics.get("boxes") or []
        if self.box_visible is not None and len(self.box_visible) == len(boxes):
            return [bool(v) for v in self.box_visible]
        return [bool(b.get("visible")) for b in boxes]


Critic = Callable[[Image.Image, Image.Image], Any]


def make_critic(cfg: Any) -> Critic:
    """Bind :func:`critique_night` to the configured judge model."""
    from aieng.syn_data.image.judge.night_critic import critique_night  # noqa: PLC0415

    judge = cfg.judge
    kwargs = {
        "model": str(judge.model_id),
        "api_base_url": judge.get("api_base_url"),
        "provider": judge.get("api_provider"),
    }

    def _critic(original: Image.Image, candidate: Image.Image) -> Any:
        return critique_night(original, candidate, **kwargs)

    return _critic


def evaluate_gates(
    metrics: dict[str, Any],
    vlm_overall: float | None,
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Accept-gate check; returns ``{"passed": bool, "reasons": [...]}``.

    Missing metrics fail their gate. The VLM threshold applies only when a VLM
    score exists.
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    reasons = []

    def _num(key: str) -> float:
        value = metrics.get(key)
        return float("nan") if value is None else float(value)

    if not _num("beam_corr") >= g["min_beam_corr"]:
        reasons.append(f"beam_corr {_num('beam_corr'):.2f} < {g['min_beam_corr']}")
    if not _num("phantom_light") <= g["max_phantom_light"]:
        reasons.append(
            f"phantom_light {_num('phantom_light'):.5f} > {g['max_phantom_light']}"
        )
    if not _num("lit_structure") >= g["min_lit_structure"]:
        reasons.append(
            f"lit_structure {_num('lit_structure'):.2f} < {g['min_lit_structure']}"
        )
    luma = _num("mean_luma")
    if not g["min_mean_luma"] <= luma <= g["max_mean_luma"]:
        reasons.append(
            f"mean_luma {luma:.3f} outside [{g['min_mean_luma']}, {g['max_mean_luma']}]"
        )
    if vlm_overall is not None and vlm_overall < g["min_vlm_overall"]:
        reasons.append(f"vlm_overall {vlm_overall:.1f} < {g['min_vlm_overall']}")
    return {"passed": not reasons, "reasons": reasons}


def score_candidate(
    candidate: Candidate,
    image: Image.Image,
    frame: Frame,
    analysis: SceneAnalysis,
    *,
    critic: Critic | None = None,
    gates: dict[str, float] | None = None,
) -> Candidate:
    """Fill metrics, optional critique, scores, and gate result (in place)."""
    original = frame.load_image()
    metrics = night.night_metrics(
        original,
        image,
        analysis.irradiance,
        analysis.geometry,
        boxes=list(frame.boxes),
        box_scale=analysis.box_scale,
    )
    candidate.metrics = to_jsonable(metrics)
    candidate.critique = None
    candidate.vlm_overall = None
    if critic is not None:
        work_image = original.resize(analysis.work_size, Image.Resampling.LANCZOS)
        try:
            result = critic(work_image, image)
        except Exception as exc:  # noqa: BLE001 - API failures must not lose the candidate
            candidate.critique = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            candidate.critique = to_jsonable(
                {
                    "scores": result.scores,
                    "overall": result.overall,
                    "critique": result.critique,
                    "prompt_fix": result.prompt_fix,
                    "error": result.error,
                },
            )
            candidate.vlm_overall = result.overall
    candidate.physics_score = night.selection_score(metrics)
    candidate.gate = evaluate_gates(candidate.metrics, candidate.vlm_overall, gates)
    return candidate


def rank_key(candidate: Candidate, pool: list[Candidate]) -> float:
    """Return the combined score if all of ``pool`` has VLM scores, else physics."""
    if pool and all(c.vlm_overall is not None for c in pool):
        return night.selection_score(candidate.metrics, candidate.vlm_overall)
    return candidate.physics_score


def history_entry(candidate: Candidate) -> dict[str, Any]:
    """Compact record for the VLM prompt rewriter."""
    m = candidate.metrics
    c = candidate.critique or {}
    return dict(
        to_jsonable(
            {
                "label": candidate.label,
                "input": candidate.input,
                "metrics": {
                    k: m.get(k)
                    for k in (
                        "beam_corr",
                        "phantom_light",
                        "lit_structure",
                        "clipped_lit",
                        "mean_luma",
                    )
                },
                "vlm_scores": c.get("scores"),
                "vlm_overall": candidate.vlm_overall,
                "critique": c.get("critique"),
                "prompt_fix": c.get("prompt_fix"),
            },
        ),
    )
