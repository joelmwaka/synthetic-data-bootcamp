"""Batch job specs and the per-job loop executed by the worker."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from aieng.syn_data.image.generate import night
from aieng.syn_data.image.workflow import night as wf

from studio.engine import Engine


PROMPT_MODES = ("preset", "custom")


def build_spec(
    *,
    dataset: str,
    hardware: str,
    frame_ids: list[str],
    mode: str,
    preset: str | None = None,
    slots: dict[str, str] | None = None,
    input_kind: str = "prior",
    refresh_scene: bool = True,
    seed_offsets: list[int] | None = None,
    use_vlm: bool = False,
    gates: dict[str, float] | None = None,
    auto_accept: bool = False,
    label: str = "batch",
) -> dict[str, Any]:
    """Validate and build a JSON-safe job spec."""
    if not frame_ids:
        raise ValueError("Select at least one frame")
    if mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {PROMPT_MODES}")
    if mode == "preset" and preset not in wf.PRESET_DESCRIPTIONS:
        raise ValueError(f"Unknown preset {preset!r}")
    if mode == "custom":
        if not slots:
            raise ValueError("Custom mode needs prompt slots")
        night.compose_prompt(slots)
    if input_kind not in ("photo", "prior"):
        raise ValueError("input_kind must be photo or prior")
    offsets = sorted({int(s) for s in (seed_offsets or [0])})
    unknown = set(gates or {}) - set(wf.DEFAULT_GATES)
    if unknown:
        raise ValueError(f"Unknown gates {sorted(unknown)}")
    return {
        "dataset": dataset,
        "hardware": hardware,
        "frame_ids": list(dict.fromkeys(frame_ids)),
        "mode": mode,
        "preset": preset,
        "slots": slots,
        "input_kind": input_kind,
        "refresh_scene": bool(refresh_scene),
        "seed_offsets": offsets,
        "use_vlm": bool(use_vlm),
        "gates": {**wf.DEFAULT_GATES, **(gates or {})},
        "auto_accept": bool(auto_accept),
        "label": label,
    }


def resolve_prompt(spec: dict[str, Any], facts: dict[str, Any]) -> tuple[str, dict[str, str] | None, str]:
    """Return ``(prompt, slots, input_kind)`` for one frame.

    Presets are rebuilt from each frame's facts. Custom slots are reused, with
    the purely factual ``scene`` slot refreshed per frame when requested.
    """
    if spec["mode"] == "preset":
        preset = wf.prompt_presets(facts)[spec["preset"]]
        return preset["prompt"], preset["slots"], preset["input"]
    slots = dict(spec["slots"])
    if spec.get("refresh_scene", True) and "scene" in slots:
        slots["scene"] = night.physics_prompt_slots(facts)["scene"]
    return night.compose_prompt(slots), slots, spec["input_kind"]


def run_job(
    job_id: int,
    spec: dict[str, Any],
    engine: Engine,
    *,
    report: Callable[[dict[str, Any]], None],
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    """Process every frame of a job; per-frame failures are recorded, not fatal."""
    progress: dict[str, Any] = {
        "frames_total": len(spec["frame_ids"]),
        "frames_done": 0,
        "candidates": 0,
        "passed": 0,
        "accepted": 0,
        "run_ids": [],
        "frame_errors": {},
        "message": "starting",
        "cancelled": False,
    }

    def _msg(text: str) -> None:
        progress["message"] = text
        report(progress)

    for frame_id in spec["frame_ids"]:
        if should_cancel():
            progress["cancelled"] = True
            break
        try:
            frame = engine.frame(frame_id)
            _msg(f"{frame_id}: analysing scene")
            analysis = engine.analyze(frame)
            prompt, slots, input_kind = resolve_prompt(spec, analysis.facts)
            run_id = engine.store.create_run(
                frame,
                kind="batch",
                settings={"job_id": job_id, "spec": spec},
                analysis=analysis.meta(),
            )
            progress["run_ids"].append(run_id)
            cands = engine.generate(
                run_id,
                frame,
                analysis,
                label=spec["label"],
                input_kind=input_kind,
                prompt=prompt,
                slots=slots,
                seed_offsets=spec["seed_offsets"],
                use_vlm=spec["use_vlm"],
                gates=spec["gates"],
                progress=_msg,
                should_stop=should_cancel,
            )
            for cand in cands:
                progress["candidates"] += 1
                if cand.gate.get("passed"):
                    progress["passed"] += 1
                    if spec["auto_accept"]:
                        engine.store.update_candidate(
                            run_id,
                            cand.candidate_id,
                            decision="accepted",
                            note="auto: passed batch gates",
                        )
                        progress["accepted"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the batch
            progress["frame_errors"][frame_id] = f"{type(exc).__name__}: {exc}"
            progress["last_traceback"] = traceback.format_exc(limit=5)
        progress["frames_done"] += 1
        _msg(f"{progress['frames_done']}/{progress['frames_total']} frames done")
    if should_cancel():
        progress["cancelled"] = True
    _msg("cancelled" if progress["cancelled"] else "finished")
    return progress
