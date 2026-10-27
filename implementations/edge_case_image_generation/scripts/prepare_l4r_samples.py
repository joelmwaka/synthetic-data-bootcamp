#!/usr/bin/env python3
"""Prepare switch-bearing railway editing seeds without modifying raw data."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from aieng.syn_data.image.data.frames import switch_boxes
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "l4r_nlb"
RAW = DATA / "raw" / "L4R_NLB_fall"
SAMPLES = DATA / "samples"
SAMPLE_COUNT = 16
MIN_FRAME_GAP = 20


def _collect_candidates() -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    for annotation in sorted((RAW / "annotations").glob("*.json")):
        source = RAW / "images" / f"{annotation.stem}.png"
        mask = RAW / "masks" / source.name
        if not source.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing image or mask for {annotation}")
        raw = json.loads(annotation.read_text())
        with Image.open(source) as image:
            width, height = image.size
        switches, found = switch_boxes(raw, (width, height))
        issues.update(found)
        boxes = [{"label": "switch", "bbox_xyxy": s["bbox_xyxy"]} for s in switches]
        attributes = [{"id": s["switch_id"], "kind": s["kind"], "direction": s["direction"]} for s in switches]
        if not boxes:
            if raw["switches"]:
                issues["images_without_usable_switch_boxes"] += 1
            continue
        largest = max((x2 - x1) * (y2 - y1) / (width * height) for x1, y1, x2, y2 in (s["bbox_xyxy"] for s in switches))
        candidates.append(
            {
                "source_id": annotation.stem,
                "frame": int(annotation.stem.rsplit("_", 1)[1]),
                "file": f"scene_{annotation.stem}.png",
                "image": str(source.relative_to(DATA)),
                "annotation": str(annotation.relative_to(DATA)),
                "mask": str(mask.relative_to(DATA)),
                "camera": "raw/L4R_NLB_fall/camera/camera.yaml",
                "season": "fall",
                "width": width,
                "height": height,
                "tags": raw["tag groups"],
                "switches": attributes,
                "boxes": boxes,
                "largest_box_area_fraction": largest,
            },
        )
    return candidates, issues


def _write_preview(selected: list[dict[str, Any]]) -> None:
    tile_width, tile_height = 480, 310
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 4), "white")
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(selected):
        with Image.open(SAMPLES / row["file"]) as original:
            tile = original.convert("RGB")
        draw = ImageDraw.Draw(tile)
        for box in row["boxes"]:
            draw.rectangle(box["bbox_xyxy"], outline="yellow", width=6)
        tile.thumbnail((tile_width, tile_height - 40))
        x = (index % 4) * tile_width
        y = (index // 4) * tile_height
        sheet.paste(tile, (x, y))
        draw = ImageDraw.Draw(sheet)
        kinds = "/".join(sorted({s["kind"] for s in row["switches"]}))
        draw.text(
            (x + 5, y + tile_height - 35),
            f"{row['source_id']} | {len(row['boxes'])} switches | {kinds}",
            fill="black",
            font=font,
        )
    sheet.save(DATA / "switch_contact_sheet.jpg", quality=90)


def main() -> None:
    """Prepare a reproducible switch shortlist and its detection metadata."""
    if SAMPLES.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample directory: {SAMPLES}")
    candidates, issues = _collect_candidates()
    # Favor prominent switches and multi-switch layouts, then separate frame IDs.
    ranked = sorted(
        candidates,
        key=lambda row: (
            -math.sqrt(row["largest_box_area_fraction"]) * (1 + 0.15 * (len(row["boxes"]) - 1)),
            row["source_id"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if all(abs(row["frame"] - other["frame"]) >= MIN_FRAME_GAP for other in selected):
            selected.append(row)
            if len(selected) == SAMPLE_COUNT:
                break
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError(f"Only {len(selected)} sufficiently separated candidates found")
    selected.sort(key=lambda row: row["frame"])
    SAMPLES.mkdir()
    labels = {}
    for row in selected:
        source = DATA / row["image"]
        (SAMPLES / row["file"]).symlink_to(Path("..") / source.relative_to(DATA))
        labels[row["file"]] = row["boxes"]
    (SAMPLES / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    metadata = {
        "dataset": "l4r_nlb",
        "source": "https://zenodo.org/records/14260575",
        "purpose": "Switch-bearing editing seeds, not a train/test split or negative pool",
        "selection": {
            "count": SAMPLE_COUNT,
            "minimum_frame_id_gap": MIN_FRAME_GAP,
            "ranking": "sqrt(largest box area fraction) * (1 + 0.15 * (box count - 1))",
            "note": "Frame-ID spacing is not a guarantee of visual or route independence",
        },
        "candidate_count": len(candidates),
        "annotation_issues": dict(issues),
        "rows": selected,
    }
    (SAMPLES / "source_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _write_preview(selected)
    print(f"Selected {len(selected)} of {len(candidates)} usable switch images")
    print(f"Annotation issues excluded: {dict(issues)}")
    print(f"Prepared samples: {SAMPLES}")
    print(f"Preview: {DATA / 'switch_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
