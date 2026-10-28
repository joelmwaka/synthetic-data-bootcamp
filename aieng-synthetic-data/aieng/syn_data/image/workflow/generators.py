"""Registry of image generators for the night workflow (app, batch worker, notebooks).

Each :class:`GeneratorSpec` maps a user-facing model choice onto one
``MethodComparer`` method plus generation-config overrides, declares its tunable
parameters, the Hugging Face repos it needs, and how it expects prompts:

- ``instruction`` — long edit instructions (Klein, Qwen-Image-Edit, Nano Banana 2)
- ``short_instruction`` — a short edit command (InstructPix2Pix; CLIP 77-token limit)
- ``caption`` — a description of the target image + negative prompt (SD1.5 ControlNet /
  inpaint; CLIP 77-token limit)

:func:`availability` reports whether a generator is ready, needs a first-use
download, or cannot run here (missing API key, too little GPU memory, offline).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from aieng.syn_data.image.config import merge_skip_null


KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"
SD15 = "runwayml/stable-diffusion-v1-5"
SD15_INPAINT = "runwayml/stable-diffusion-inpainting"
CONTROLNET_DEPTH = "lllyasviel/sd-controlnet-depth"
CONTROLNET_SEG = "lllyasviel/sd-controlnet-seg"
INSTRUCT_PIX2PIX = "timbrooks/instruct-pix2pix"
QWEN_IMAGE_EDIT = "Qwen/Qwen-Image-Edit"
NANO_BANANA_2 = "gemini-3.1-flash-image"
DEFAULT_GENERATOR = "klein4b_instruct"
CLIP_MAX_TOKENS = 77

# The API editor wraps prompts in a preamble; the library default asks to keep the
# lighting, which contradicts a night relight.
RELIGHT_PREAMBLE = (
    "Edit this forward-facing train cab photograph. Keep the camera angle, framing, "
    "track layout and every object in place; change only the lighting as instructed."
)

# ADE20K colours the SD1.5 segmentation ControlNet was trained with, indexed by the
# 0-based ADE20K label used by SegFormer (0 = wall, 2 = sky, 4 = tree, 6 = road).
# Source: lllyasviel/sd-controlnet-seg model card palette without its leading
# background row (the card's example maps labels off by one).
# fmt: off
ADE20K_PALETTE = np.array(
    [
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
        [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
        [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
        [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
        [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
        [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
        [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
        [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
        [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
        [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
        [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
        [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112],
        [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
        [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
        [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184],
        [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
        [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
        [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
        [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170],
        [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
        [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
        [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235],
        [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
        [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
        [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
        [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0],
        [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255],
    ],
    dtype=np.uint8,
)
# fmt: on


@dataclass(frozen=True)
class ParamSpec:
    """One tunable generation parameter, written to ``cfg_key`` (dot path)."""

    name: str
    label: str
    cfg_key: str
    default: float
    min: float
    max: float
    step: float
    kind: str = "float"
    help: str = ""

    def coerce(self, value: Any) -> float | int:
        """Clamp to range and cast (``int`` params are rounded)."""
        number = min(max(float(value), self.min), self.max)
        return round(number) if self.kind == "int" else number


@dataclass(frozen=True)
class GeneratorSpec:
    """A selectable generator: method, model overrides, params, and requirements."""

    id: str
    title: str
    method: str
    summary: str
    overrides: dict[str, Any] = field(default_factory=dict)
    params: tuple[ParamSpec, ...] = ()
    repos: tuple[str, ...] = ()
    download_gb: float = 0.0
    min_total_vram_gb: float = 0.0
    prompt_style: str = "instruction"
    max_prompt_tokens: int | None = None
    supports_negative: bool = False
    needs_conditioning: bool = False
    anomaly_extra: dict[str, Any] = field(default_factory=dict)
    recommended_preset: str = "hybrid_prior"
    cloud: bool = False
    seeded: bool = True

    def default_params(self) -> dict[str, float | int]:
        """Default parameter values."""
        return {p.name: p.coerce(p.default) for p in self.params}

    def resolve_params(self, params: dict[str, Any] | None) -> dict[str, float | int]:
        """Defaults overlaid with ``params``; unknown names raise ``ValueError``."""
        known = {p.name: p for p in self.params}
        unknown = set(params or {}) - set(known)
        if unknown:
            raise ValueError(f"{self.id}: unknown params {sorted(unknown)}")
        out = self.default_params()
        for name, value in (params or {}).items():
            out[name] = known[name].coerce(value)
        return out


GENERATORS: dict[str, GeneratorSpec] = {
    spec.id: spec
    for spec in (
        GeneratorSpec(
            id="klein4b_instruct",
            title="FLUX.2 Klein 4B — instruct edit",
            method="instruct",
            summary="Small distilled instruction editor (4 steps). Default; best physics scores so far.",
            overrides={"instruct_model_id": KLEIN_4B, "instruct_guidance_scale": 1.0},
            params=(
                ParamSpec(
                    "steps",
                    "Steps",
                    "instruct_num_inference_steps",
                    4,
                    1,
                    12,
                    1,
                    kind="int",
                ),
            ),
            repos=(KLEIN_4B,),
            download_gb=15.0,
        ),
        GeneratorSpec(
            id="klein4b_img2img",
            title="FLUX.2 Klein 4B — full-frame img2img",
            method="inpaint",
            summary=(
                "Klein inpaint pipeline with a full-frame mask: re-noises the input by `strength`, so lower "
                "values keep more of the original structure."
            ),
            overrides={
                "inpaint_model_id": KLEIN_4B,
                "inpaint_guidance_scale": 1.0,
                "padding_mask_crop": 0,
            },
            params=(
                ParamSpec(
                    "strength",
                    "Strength",
                    "strength",
                    0.75,
                    0.3,
                    1.0,
                    0.05,
                    help="Share of the image re-noised.",
                ),
                ParamSpec(
                    "steps",
                    "Steps",
                    "inpaint_num_inference_steps",
                    4,
                    1,
                    12,
                    1,
                    kind="int",
                ),
            ),
            repos=(KLEIN_4B,),
            download_gb=15.0,
            needs_conditioning=True,
            anomaly_extra={"edit_mask": {"mode": "full", "recomposite": False}},
        ),
        GeneratorSpec(
            id="sd15_controlnet",
            title="SD 1.5 + ControlNet (depth + segmentation)",
            method="controlnet_dual",
            summary=(
                "Full-frame img2img locked to the scene's depth map and ADE20K segmentation, so geometry "
                "is held while the look changes. Prompt describes the target image (77 CLIP tokens). "
                "Use the physics render as input: SD1.5 cannot darken a dusk photo to night."
            ),
            overrides={
                "family": "sd15",
                "base_model_id": SD15,
                "controlnet": {"depth": CONTROLNET_DEPTH, "seg": CONTROLNET_SEG},
                "global_anomaly_ids": ["night_headlights"],
                "controlnet_strength_cap_global": 0.95,
            },
            params=(
                ParamSpec(
                    "strength", "Strength", "controlnet_strength", 0.5, 0.3, 0.95, 0.05
                ),
                ParamSpec(
                    "steps", "Steps", "num_inference_steps", 30, 10, 60, 1, kind="int"
                ),
                ParamSpec(
                    "guidance",
                    "Guidance",
                    "guidance_scale",
                    6.5,
                    2.0,
                    6.5,
                    0.5,
                    help="Capped at 6.5 by the pipeline.",
                ),
                ParamSpec(
                    "depth_scale",
                    "Depth control",
                    "controlnet_scale.depth",
                    0.8,
                    0.0,
                    1.5,
                    0.05,
                ),
                ParamSpec(
                    "seg_scale",
                    "Segmentation control",
                    "controlnet_scale.seg",
                    0.5,
                    0.0,
                    1.5,
                    0.05,
                ),
            ),
            repos=(SD15, CONTROLNET_DEPTH, CONTROLNET_SEG),
            download_gb=7.5,
            prompt_style="caption",
            max_prompt_tokens=CLIP_MAX_TOKENS,
            supports_negative=True,
            needs_conditioning=True,
            recommended_preset="caption",
        ),
        GeneratorSpec(
            id="sd15_inpaint",
            title="SD 1.5 inpainting — full-frame",
            method="inpaint",
            summary=(
                "Stable Diffusion inpainting checkpoint with a full-frame mask (img2img without structural "
                "control, so it can move or invent scene content). Prompt describes the target image "
                "(77 CLIP tokens); use the physics render as input."
            ),
            overrides={
                "family": "sd15",
                "inpaint_model_id": SD15_INPAINT,
                "padding_mask_crop": 0,
            },
            params=(
                ParamSpec("strength", "Strength", "strength", 0.5, 0.3, 0.99, 0.05),
                ParamSpec(
                    "steps", "Steps", "num_inference_steps", 30, 10, 60, 1, kind="int"
                ),
                ParamSpec(
                    "guidance", "Guidance", "guidance_scale", 7.5, 2.0, 12.0, 0.5
                ),
            ),
            repos=(SD15_INPAINT,),
            download_gb=2.6,
            prompt_style="caption",
            max_prompt_tokens=CLIP_MAX_TOKENS,
            supports_negative=True,
            needs_conditioning=True,
            anomaly_extra={"edit_mask": {"mode": "full", "recomposite": False}},
            recommended_preset="caption",
        ),
        GeneratorSpec(
            id="instruct_pix2pix",
            title="InstructPix2Pix (SD 1.5)",
            method="instruct",
            summary=(
                "Classic SD1.5 instruction editor; runs at 512 px and wants a short command (77 CLIP tokens). "
                "On a dusk photo it tends towards sunset, so start from the physics render."
            ),
            overrides={"family": "sd15", "instruct_model_id": INSTRUCT_PIX2PIX},
            params=(
                ParamSpec(
                    "steps",
                    "Steps",
                    "instruct_num_inference_steps",
                    30,
                    10,
                    60,
                    1,
                    kind="int",
                ),
                ParamSpec(
                    "image_guidance",
                    "Image guidance",
                    "instruct_image_guidance",
                    1.5,
                    1.0,
                    3.0,
                    0.1,
                    help="Higher keeps more of the input image.",
                ),
                ParamSpec(
                    "text_guidance",
                    "Text guidance",
                    "instruct_guidance_scale",
                    7.5,
                    3.0,
                    12.0,
                    0.5,
                ),
            ),
            repos=(INSTRUCT_PIX2PIX,),
            download_gb=5.5,
            prompt_style="short_instruction",
            max_prompt_tokens=CLIP_MAX_TOKENS,
            recommended_preset="short_instruction",
        ),
        GeneratorSpec(
            id="qwen_image_edit",
            title="Qwen-Image-Edit (~20B, large local)",
            method="vlm_generate_local",
            summary="Large local instruction editor. Its ~58 GB of bf16 weights must fit in GPU memory.",
            overrides={
                "vlm_local_model_id": QWEN_IMAGE_EDIT,
                "vlm_local_max_side": 768,
            },
            params=(
                ParamSpec(
                    "steps",
                    "Steps",
                    "vlm_local_num_inference_steps",
                    20,
                    8,
                    50,
                    1,
                    kind="int",
                ),
                ParamSpec(
                    "true_cfg",
                    "True CFG scale",
                    "vlm_local_true_cfg_scale",
                    4.0,
                    1.0,
                    8.0,
                    0.5,
                ),
            ),
            repos=(QWEN_IMAGE_EDIT,),
            download_gb=58.0,
            min_total_vram_gb=64.0,
            recommended_preset="effect_only",
        ),
        GeneratorSpec(
            id="nano_banana_api",
            title="Nano Banana 2 — cloud API (Vector proxy)",
            method="vlm_generate_api",
            summary=(
                f"Cloud image editing with `{NANO_BANANA_2}` through the same proxy as the VLM judge. "
                "The frame is sent to the API; results are not seed-reproducible."
            ),
            overrides={
                "vlm_api_enabled": True,
                "vlm_api_model": NANO_BANANA_2,
                "vlm_mode": "edit",
                "vlm_aspect_ratio": "16:9",
                "vlm_edit_preamble": RELIGHT_PREAMBLE,
            },
            recommended_preset="effect_only",
            cloud=True,
            seeded=False,
        ),
    )
}


def get_generator(generator_id: str) -> GeneratorSpec:
    """Look up a generator; raise ``KeyError`` with the valid ids."""
    if generator_id not in GENERATORS:
        raise KeyError(
            f"Unknown generator {generator_id!r}; choose from {sorted(GENERATORS)}"
        )
    return GENERATORS[generator_id]


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def effective_generation_cfg(
    generation_cfg: Any, spec: GeneratorSpec, params: dict[str, Any] | None = None
) -> DictConfig:
    """Return ``generation_cfg`` with the generator's overrides and parameter values."""
    patch: dict[str, Any] = {}
    for key, value in spec.overrides.items():
        _set_path(patch, key, value)
    for name, value in spec.resolve_params(params).items():
        param = next(p for p in spec.params if p.name == name)
        _set_path(patch, param.cfg_key, value)
    base = OmegaConf.create(OmegaConf.to_container(generation_cfg, resolve=True))
    merged: DictConfig = merge_skip_null(base, OmegaConf.create(patch))
    return merged


def comparer_key(spec: GeneratorSpec, generation_cfg: Any) -> tuple[str, ...]:
    """Identity of the loaded models (parameters alone never require a reload)."""
    cfg = effective_generation_cfg(generation_cfg, spec)
    controlnet = cfg.get("controlnet") or {}
    return (
        spec.method,
        str(cfg.get("family")),
        str(cfg.get("instruct_model_id")),
        str(cfg.get("inpaint_model_id")),
        str(cfg.get("base_model_id")),
        str(controlnet.get("depth")),
        str(controlnet.get("seg")),
        str(cfg.get("vlm_local_model_id")),
    )


# --- availability ----------------------------------------------------------------


@dataclass(frozen=True)
class Availability:
    """``ready`` | ``download`` (first use downloads weights) | ``blocked``."""

    status: str
    reason: str

    @property
    def usable(self) -> bool:
        """Whether generation can be attempted."""
        return self.status != "blocked"


def repo_cached(repo_id: str) -> bool:
    """Whether a model repo's index/config is in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

    for name in ("model_index.json", "config.json"):
        hit = try_to_load_from_cache(repo_id, name)
        if isinstance(hit, str):
            return True
    return False


@lru_cache(maxsize=1)
def total_vram_gb() -> float:
    """Total GPU memory across visible CUDA devices (0 without CUDA)."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0.0
        return (
            sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in range(torch.cuda.device_count())
            )
            / 1e9
        )
    except Exception:  # noqa: BLE001 - a probe must never break the UI
        return 0.0


def availability(
    spec: GeneratorSpec,
    *,
    api_key_available: bool,
    vram_gb: float | None = None,
    is_cached: Callable[[str], bool] = repo_cached,
    offline: bool | None = None,
) -> Availability:
    """Whether ``spec`` can run on this machine, and why not."""
    if spec.cloud:
        if not api_key_available:
            return Availability(
                "blocked",
                "Needs a judge/VLM API key (OPENAI_API_KEY) in the environment.",
            )
        return Availability(
            "ready", "Cloud API — the source frame is sent to the Vector proxy."
        )
    vram = total_vram_gb() if vram_gb is None else vram_gb
    if spec.min_total_vram_gb and vram < spec.min_total_vram_gb:
        return Availability(
            "blocked",
            f"Needs ≥{spec.min_total_vram_gb:.0f} GB of GPU memory; this machine has {vram:.0f} GB.",
        )
    missing = [r for r in spec.repos if not is_cached(r)]
    if not missing:
        return Availability("ready", "Weights cached locally.")
    if offline is None:
        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    if offline:
        return Availability(
            "blocked",
            f"Weights not cached and HF_HUB_OFFLINE is set: {', '.join(missing)}.",
        )
    return Availability(
        "download",
        f"First use downloads ~{spec.download_gb:.0f} GB ({', '.join(missing)}); later runs use the cache.",
    )


# --- prompts and conditioning ----------------------------------------------------


@lru_cache(maxsize=1)
def _clip_tokenizer() -> Any:
    from transformers import CLIPTokenizer  # noqa: PLC0415

    return CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=True
    )


def clip_token_count(text: str) -> int:
    """Count CLIP tokens incl. start/end (SD1.5 vocabulary); estimate if offline."""
    try:
        return len(_clip_tokenizer()(text, truncation=False)["input_ids"])
    except Exception:  # noqa: BLE001 - tokenizer not cached → estimate
        return int(len(text.split()) * 1.3) + 2


def ade20k_colour_map(label_map: np.ndarray) -> np.ndarray:
    """Colour 0-based ADE20K labels with the palette the seg ControlNet expects."""
    labels = np.asarray(label_map)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(ADE20K_PALETTE))
    out[valid] = ADE20K_PALETTE[labels[valid]]
    return out


def conditioning_from_maps(
    disparity: np.ndarray, label_map: np.ndarray
) -> tuple[Any, Any]:
    """Build ``(DepthResult, SegmentationResult)`` from cached scene maps."""
    from aieng.syn_data.image.generate.conditioning import (  # noqa: PLC0415
        DepthResult,
        SegmentationResult,
    )

    if disparity.shape != label_map.shape:
        raise ValueError(f"depth {disparity.shape} and labels {label_map.shape} differ")
    grey = (np.clip(disparity, 0, 1) * 255).astype(np.uint8)
    coloured = ade20k_colour_map(label_map)
    depth = DepthResult(
        depth_map=disparity.astype(np.float32), colormap=np.stack([grey] * 3, axis=-1)
    )
    segmentation = SegmentationResult(
        label_map=label_map.astype(np.int32),
        colored_map=coloured,
        overlay=coloured,
        num_regions=int(len(np.unique(label_map))),
    )
    return depth, segmentation


def run_generator(
    comparer: Any,
    generation_cfg: Any,
    spec: GeneratorSpec,
    source: Image.Image,
    prompt: str,
    *,
    params: dict[str, Any] | None = None,
    negative_prompt: str = "",
    conditioning: tuple[Any, Any] | None = None,
    work_size: tuple[int, int],
    seed_offset: int = 0,
) -> Image.Image:
    """Run one edit with ``spec`` and return it on the ``work_size`` grid."""
    if spec.needs_conditioning and conditioning is None:
        raise ValueError(f"{spec.id} needs depth + segmentation conditioning")
    cfg = effective_generation_cfg(generation_cfg, spec, params)
    anomaly: dict[str, Any] = {
        "id": "night_headlights",
        "prompt": prompt,
        "max_side": max(work_size),
        **spec.anomaly_extra,
    }
    if negative_prompt and spec.supports_negative:
        anomaly["negative_prompt"] = negative_prompt
    depth, segmentation = conditioning if conditioning is not None else (None, None)
    result = comparer.run_method(
        spec.method,
        source,
        depth=depth,
        segmentation=segmentation,
        generation_cfg=cfg,
        anomaly_cfg=OmegaConf.create(anomaly),
        seed_offset=seed_offset,
    )
    if getattr(result, "error", None):
        raise RuntimeError(f"{spec.id} failed: {result.error}")
    out: Image.Image = result.image.convert("RGB")
    return (
        out
        if out.size == work_size
        else out.resize(work_size, Image.Resampling.LANCZOS)
    )
