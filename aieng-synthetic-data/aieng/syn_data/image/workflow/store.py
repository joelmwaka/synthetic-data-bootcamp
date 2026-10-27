"""Persistent, reproducible run records and dataset export.

Layout under ``root`` (typically ``outputs/<dataset>/studio``)::

    runs/<run_id>/run.json             frame, settings, analysis summary, candidates
    runs/<run_id>/candidates/<id>.png  generated images (work-grid resolution)
    exports/<export_id>/images/…       accepted images
    exports/<export_id>/labels.json    filename → boxes (scaled to the image)
    exports/<export_id>/provenance.json how every exported image was produced

``run.json`` is updated under a per-run file lock with an atomic replace, so
the app and a batch worker can append to the same run safely.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock
from PIL import Image

from aieng.syn_data.image.data.frames import Frame
from aieng.syn_data.image.workflow.night import Candidate, to_jsonable, utc_now


REVIEW_FIELDS = {"decision", "note", "box_visible"}
DECISIONS = {"pending", "accepted", "rejected"}
_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def studio_root(cfg: Any) -> Path:
    """``outputs/<dataset>/studio`` — shared by the app, worker, and notebooks."""
    return Path(cfg.paths.outputs_dir) / "studio"


class RunStore:
    """File-backed store of generation runs (one run = one source frame)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.exports_dir = self.root / "exports"

    def _run_dir(self, run_id: str) -> Path:
        if not _RUN_ID.match(run_id):
            raise ValueError(f"Invalid run id {run_id!r}")
        return self.runs_dir / run_id

    def _lock(self, run_id: str) -> FileLock:
        return FileLock(str(self._run_dir(run_id) / "run.lock"), timeout=60)

    def _write(self, run_id: str, record: dict[str, Any]) -> None:
        path = self._run_dir(run_id) / "run.json"
        tmp = path.with_name(f".run.{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(to_jsonable(record), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    def _update(self, run_id: str, mutate: Callable[[dict[str, Any]], Any]) -> Any:
        with self._lock(run_id):
            record = self.load_run(run_id)
            result = mutate(record)
            record["updated_at"] = utc_now()
            self._write(run_id, record)
            return result

    def create_run(
        self,
        frame: Frame,
        *,
        kind: str,
        settings: dict[str, Any],
        analysis: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a run for ``frame``; an existing ``run_id`` is reused unchanged."""
        if run_id is None:
            stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
            run_id = f"{stamp}_{frame.frame_id}_{uuid.uuid4().hex[:6]}"
        run_dir = self._run_dir(run_id)
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
        with self._lock(run_id):
            if (run_dir / "run.json").is_file():
                return run_id
            with Image.open(frame.image_path) as image:
                size = list(image.size)
            self._write(
                run_id,
                {
                    "run_id": run_id,
                    "kind": kind,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "frame": {
                        "dataset": frame.dataset,
                        "frame_id": frame.frame_id,
                        "image_path": str(frame.image_path),
                        "size": size,
                        "boxes": list(frame.boxes),
                        "tags": {k: list(v) for k, v in frame.tags.items()},
                    },
                    "settings": settings,
                    "analysis": analysis or {},
                    "candidates": [],
                },
            )
        return run_id

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Load a run record."""
        data: dict[str, Any] = json.loads(
            (self._run_dir(run_id) / "run.json").read_text(encoding="utf-8")
        )
        return data

    def list_runs(self) -> list[dict[str, Any]]:
        """All runs, newest first (unreadable records are skipped)."""
        if not self.runs_dir.is_dir():
            return []
        out = []
        for path in self.runs_dir.glob("*/run.json"):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)

    def candidates(self, run_id: str) -> list[Candidate]:
        """Candidates of a run, oldest first."""
        return [Candidate.from_dict(c) for c in self.load_run(run_id)["candidates"]]

    def add_candidate(
        self, run_id: str, candidate: Candidate, image: Image.Image
    ) -> Candidate:
        """Save the image and append the candidate record."""
        name = f"{candidate.candidate_id}.png"
        path = self._run_dir(run_id) / "candidates" / name
        tmp = path.with_name(f".{name}.{uuid.uuid4().hex}.png")
        image.save(tmp)
        tmp.replace(path)
        candidate.image_file = f"candidates/{name}"

        def _append(record: dict[str, Any]) -> None:
            if any(
                c["candidate_id"] == candidate.candidate_id
                for c in record["candidates"]
            ):
                raise ValueError(f"Duplicate candidate {candidate.candidate_id}")
            record["candidates"].append(candidate.to_dict())

        self._update(run_id, _append)
        return candidate

    def update_candidate(
        self, run_id: str, candidate_id: str, **changes: Any
    ) -> Candidate:
        """Apply review changes (``decision``, ``note``, ``box_visible``)."""
        unknown = set(changes) - REVIEW_FIELDS
        if unknown:
            raise ValueError(f"Not reviewable: {sorted(unknown)}")
        if "decision" in changes and changes["decision"] not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")

        def _apply(record: dict[str, Any]) -> Candidate:
            for raw in record["candidates"]:
                if raw["candidate_id"] == candidate_id:
                    if "box_visible" in changes and changes["box_visible"] is not None:
                        n = len((raw.get("metrics") or {}).get("boxes") or [])
                        if len(changes["box_visible"]) != n:
                            raise ValueError(f"box_visible needs {n} flags")
                    raw.update(to_jsonable(changes))
                    raw["reviewed_at"] = utc_now()
                    return Candidate.from_dict(raw)
            raise KeyError(candidate_id)

        result: Candidate = self._update(run_id, _apply)
        return result

    def image_path(self, run_id: str, candidate: Candidate) -> Path:
        """Absolute path of a candidate image."""
        return self._run_dir(run_id) / candidate.image_file

    def accepted(self) -> list[tuple[dict[str, Any], Candidate]]:
        """All accepted ``(run, candidate)`` pairs."""
        return [
            (run, cand)
            for run in self.list_runs()
            for cand in (Candidate.from_dict(c) for c in run.get("candidates", []))
            if cand.decision == "accepted"
        ]

    def export_accepted(
        self, *, include_dark_boxes: bool = False, name: str | None = None
    ) -> Path:
        """Copy accepted images with labels and provenance into a new export folder.

        Boxes are scaled from source-image to generated-image coordinates.
        Boxes judged dark (reviewer override, else metric flag) are dropped
        unless ``include_dark_boxes`` — then kept with ``visible_at_night=False``.
        """
        pairs = self.accepted()
        if not pairs:
            raise ValueError("No accepted candidates to export")
        export_id = (
            name
            or f"export_{utc_now().replace(':', '').replace('-', '').replace('+0000', 'Z')}"
        )
        if not _RUN_ID.match(export_id):
            raise ValueError(f"Invalid export name {export_id!r}")
        dest = self.exports_dir / export_id
        if dest.exists():
            raise FileExistsError(dest)
        (dest / "images").mkdir(parents=True)
        labels: dict[str, list[dict[str, Any]]] = {}
        provenance = []
        for run, cand in pairs:
            src = self._run_dir(run["run_id"]) / cand.image_file
            fname = f"{run['frame']['frame_id']}_{run['run_id'][-6:]}_{cand.candidate_id}.png"
            shutil.copy2(src, dest / "images" / fname)
            with Image.open(src) as image:
                out_w, out_h = image.size
            src_w, src_h = run["frame"]["size"]
            sx, sy = out_w / src_w, out_h / src_h
            boxes = []
            for box, visible in zip(
                cand.metrics.get("boxes") or [], cand.visible_flags()
            ):
                if not visible and not include_dark_boxes:
                    continue
                x1, y1, x2, y2 = box["bbox_xyxy"]
                boxes.append(
                    {
                        "label": box["label"],
                        "bbox_xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                        "visible_at_night": bool(visible),
                    },
                )
            labels[fname] = boxes
            provenance.append(
                {
                    "file": fname,
                    "run_id": run["run_id"],
                    "candidate_id": cand.candidate_id,
                    "source": run["frame"],
                    "settings": run.get("settings", {}),
                    "analysis": run.get("analysis", {}),
                    "candidate": cand.to_dict(),
                },
            )
        (dest / "labels.json").write_text(
            json.dumps(to_jsonable(labels), indent=2) + "\n", encoding="utf-8"
        )
        (dest / "provenance.json").write_text(
            json.dumps(to_jsonable(provenance), indent=2) + "\n", encoding="utf-8"
        )
        return dest
