"""Dataset-agnostic access to source frames and their annotations.

A :class:`Frame` bundles one real image with whatever the dataset provides:
boxes, Labels4Rail track polylines and masks, scene tags, and camera
calibration. ``capabilities`` tells callers (apps, notebooks) which features
are meaningful for that frame — e.g. the night physics prior needs
``calibration``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from aieng.syn_data.image.data.loader import list_sample_images


CAP_BOXES = "boxes"
CAP_TRACKS = "tracks"
CAP_MASK = "mask"
CAP_TAGS = "tags"
CAP_CALIBRATION = "calibration"


@dataclass(frozen=True)
class Frame:
    """One real source image plus its annotations."""

    dataset: str
    frame_id: str
    image_path: Path
    boxes: tuple[dict[str, Any], ...] = ()
    tags: dict[str, tuple[str, ...]] = field(default_factory=dict)
    annotation: dict[str, Any] | None = None
    mask_path: Path | None = None
    camera_path: Path | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        """Features this frame supports."""
        caps = set()
        if self.boxes:
            caps.add(CAP_BOXES)
        if self.annotation and self.annotation.get("tracks"):
            caps.add(CAP_TRACKS)
        if self.mask_path is not None:
            caps.add(CAP_MASK)
        if any(self.tags.values()):
            caps.add(CAP_TAGS)
        if self.camera_path is not None:
            caps.add(CAP_CALIBRATION)
        return frozenset(caps)

    def load_image(self) -> Image.Image:
        """Load the RGB image."""
        with Image.open(self.image_path) as image:
            return image.convert("RGB")

    def load_mask(self) -> Image.Image | None:
        """Load the track mask (``L``), if any."""
        if self.mask_path is None:
            return None
        with Image.open(self.mask_path) as mask:
            return mask.convert("L")

    def summary(self) -> dict[str, Any]:
        """Flat row for tables / filtering."""
        labels = Counter(str(b["label"]) for b in self.boxes)
        return {
            "frame_id": self.frame_id,
            "n_boxes": len(self.boxes),
            "labels": dict(labels),
            "tags": {k: list(v) for k, v in self.tags.items()},
            "capabilities": sorted(self.capabilities),
        }


class FrameSource(Protocol):
    """Read-only collection of frames for one dataset."""

    dataset: str

    def frame_ids(self) -> list[str]:
        """Return sorted frame ids."""
        ...

    def get(self, frame_id: str) -> Frame:
        """Load one frame; raise ``KeyError`` if unknown."""
        ...


def switch_boxes(
    annotation: dict[str, Any], size: tuple[int, int]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Normalise Labels4Rail switch marks into ``bbox_xyxy`` boxes.

    Switches need exactly two marks; exact duplicates are dropped. Returns the
    boxes (with ``kind`` / ``direction`` attributes) and counted issues.
    Out-of-image boxes raise ``ValueError`` — they indicate corrupt data.
    """
    width, height = size
    boxes: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    seen = set()
    for switch_id, switch in (annotation.get("switches") or {}).items():
        marks = switch.get("marks") or []
        if len(marks) != 2:
            issues["switches_without_two_marks"] += 1
            continue
        x1, x2 = sorted(float(m["x"]) for m in marks)
        y1, y2 = sorted(float(m["y"]) for m in marks)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Switch {switch_id} box outside image: {marks}")
        key = (x1, y1, x2, y2)
        if key in seen:
            issues["duplicate_boxes"] += 1
            continue
        seen.add(key)
        boxes.append(
            {
                "label": "switch",
                "bbox_xyxy": [x1, y1, x2, y2],
                "switch_id": str(switch_id),
                "kind": str(switch.get("kind", "unknown")),
                "direction": str(switch.get("direction", "unknown")),
            },
        )
    return boxes, issues


class L4RSource:
    """Labels4Rail season folder (``images``, ``annotations``, ``masks``, ``camera``).

    Only frames with both an image and an annotation are listed. Boxes are the
    annotated switches; tags are the annotation ``tag groups``.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        *,
        dataset: str = "l4r_nlb",
        camera_yaml: Path | str | None = None,
    ):
        self.dataset = dataset
        self.raw_dir = Path(raw_dir)
        self.camera_path = (
            Path(camera_yaml)
            if camera_yaml
            else self.raw_dir / "camera" / "camera.yaml"
        )
        if (
            not (self.raw_dir / "images").is_dir()
            or not (self.raw_dir / "annotations").is_dir()
        ):
            raise FileNotFoundError(f"{self.raw_dir} needs images/ and annotations/")
        if not self.camera_path.is_file():
            raise FileNotFoundError(f"Missing camera calibration {self.camera_path}")
        images = {p.stem for p in (self.raw_dir / "images").glob("*.png")}
        annotations = {p.stem for p in (self.raw_dir / "annotations").glob("*.json")}
        self._ids = sorted(images & annotations)

    def frame_ids(self) -> list[str]:
        """Return sorted frame ids."""
        return list(self._ids)

    def get(self, frame_id: str) -> Frame:
        """Load one frame; raise ``KeyError`` if unknown."""
        if frame_id not in self._ids:
            raise KeyError(frame_id)
        image_path = self.raw_dir / "images" / f"{frame_id}.png"
        annotation = json.loads(
            (self.raw_dir / "annotations" / f"{frame_id}.json").read_text(
                encoding="utf-8"
            )
        )
        with Image.open(image_path) as image:
            size = image.size
        boxes, _ = switch_boxes(annotation, size)
        mask = self.raw_dir / "masks" / f"{frame_id}.png"
        tags = {
            str(k): tuple(str(x) for x in v or [])
            for k, v in (annotation.get("tag groups") or {}).items()
        }
        return Frame(
            dataset=self.dataset,
            frame_id=frame_id,
            image_path=image_path,
            boxes=tuple(boxes),
            tags=tags,
            annotation=annotation,
            mask_path=mask if mask.is_file() else None,
            camera_path=self.camera_path,
        )


class SamplesSource:
    """Generic ``samples/`` folder with an optional filename-keyed ``labels.json``."""

    def __init__(self, samples_dir: Path | str, *, dataset: str):
        self.dataset = dataset
        self.samples_dir = Path(samples_dir)
        labels_path = self.samples_dir / "labels.json"
        raw = (
            json.loads(labels_path.read_text(encoding="utf-8"))
            if labels_path.is_file()
            else {}
        )
        self._labels: dict[str, list[dict[str, Any]]] = {}
        for name, boxes in raw.items():
            self._labels[Path(name).stem] = [
                {
                    **b,
                    "label": str(b["label"]),
                    "bbox_xyxy": [float(x) for x in b["bbox_xyxy"]],
                }
                for b in boxes
            ]
        self._paths = {p.stem: p for p in list_sample_images(self.samples_dir)}

    def frame_ids(self) -> list[str]:
        """Return sorted frame ids."""
        return sorted(self._paths)

    def get(self, frame_id: str) -> Frame:
        """Load one frame; raise ``KeyError`` if unknown."""
        path = self._paths[frame_id]
        tag = frame_id.split("_", 1)[0] if "_" in frame_id else ""
        return Frame(
            dataset=self.dataset,
            frame_id=frame_id,
            image_path=path,
            boxes=tuple(self._labels.get(frame_id, [])),
            tags={"prefix": (tag,)} if tag else {},
        )


def frame_source_for(cfg: Any) -> FrameSource:
    """Pick a source from a composed config.

    Datasets whose ``data.yaml`` sets ``raw_dir`` (Labels4Rail) use
    :class:`L4RSource`; everything else reads ``paths.samples_dir``.
    """
    data = cfg.get("data") or {}
    dataset = str(cfg.get("dataset_name"))
    raw_dir = data.get("raw_dir")
    if raw_dir:
        return L4RSource(raw_dir, dataset=dataset, camera_yaml=data.get("camera_yaml"))
    return SamplesSource(cfg.paths.samples_dir, dataset=dataset)
