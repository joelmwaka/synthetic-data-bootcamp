"""Unit tests for the job queue, batch logic, and overlays (no GPU, no Streamlit)."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from aieng.syn_data.image.generate import night
from aieng.syn_data.image.workflow import generators as gens
from aieng.syn_data.image.workflow import night as wf
from PIL import Image
from studio import batch, jobs, overlays


FACTS = {
    "environment": ["rural"],
    "weather": ["cloudy"],
    "track_layout": ["curve"],
    "n_tracks": 2,
    "n_switches": 1,
    "lamp_height_m": 1.0,
    "beam_reach_m": 41,
}


def test_queue_claims_each_job_once(tmp_path):
    queue = jobs.JobQueue(tmp_path / "jobs.sqlite")
    ids = [queue.submit({"n": i}) for i in range(5)]
    claimed: list[int] = []
    lock = threading.Lock()

    def _worker(pid: int) -> None:
        while (job := queue.claim_next(pid)) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=_worker, args=(p,)) for p in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == ids
    assert all(queue.get(i).status == "running" for i in ids)


def test_queue_progress_cancel_and_finish(tmp_path):
    queue = jobs.JobQueue(tmp_path / "jobs.sqlite")
    first, second = queue.submit({"a": 1}), queue.submit({"b": 2})
    assert queue.request_cancel(second) == "cancelled"
    job = queue.claim_next(pid=123)
    assert job is not None and job.id == first and job.spec == {"a": 1}
    assert queue.claim_next(pid=123) is None
    queue.update_progress(first, {"frames_done": 1})
    assert queue.get(first).progress == {"frames_done": 1}
    assert queue.request_cancel(first) == "running"
    assert queue.cancel_requested(first)
    queue.finish(first, status="cancelled")
    assert queue.get(first).status == "cancelled"
    with pytest.raises(ValueError):
        queue.finish(first, status="queued")
    with pytest.raises(KeyError):
        queue.request_cancel(999)
    with pytest.raises(ValueError):
        queue.submit({"x": float("nan")})


def test_queue_recovers_orphaned_jobs(tmp_path):
    queue = jobs.JobQueue(tmp_path / "jobs.sqlite")
    job_id = queue.submit({})
    queue.claim_next(pid=111)
    queue.heartbeat(222)
    assert [w["pid"] for w in queue.live_workers()] == [222]
    assert queue.recover_orphans() == 1
    assert queue.get(job_id).status == "failed"
    queue.remove_worker(222)
    assert queue.live_workers() == []


def test_build_spec_validates():
    spec = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a", "a", "b"],
        mode="preset",
        preset="hybrid_prior",
        seed_offsets=[2, 0, 2],
    )
    assert spec["frame_ids"] == ["a", "b"] and spec["seed_offsets"] == [0, 2]
    assert spec["gates"] == wf.DEFAULT_GATES
    json.dumps(spec, allow_nan=False)
    with pytest.raises(ValueError, match="frame"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=[], mode="preset", preset="naive")
    with pytest.raises(ValueError, match="preset"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="nope")
    with pytest.raises(ValueError, match="slots"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="custom")
    with pytest.raises(ValueError, match="Unknown prompt slots"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="custom", slots={"bad": "x"})
    with pytest.raises(ValueError, match="gates"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="naive", gates={"x": 1})


def test_resolve_prompt_refreshes_only_the_scene_slot():
    custom = dict(night.physics_prompt_slots({**FACTS, "environment": ["urban"]}))
    custom["beam"] = "tuned beam text"
    spec = batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="custom", slots=custom, input_kind="photo")
    prompt, slots, input_kind, negative = batch.resolve_prompt(spec, FACTS)
    assert input_kind == "photo" and slots is not None and negative == ""
    assert "rural" in slots["scene"] and slots["beam"] == "tuned beam text"
    assert prompt == night.compose_prompt(slots)
    spec["refresh_scene"] = False
    _, kept, _, _ = batch.resolve_prompt(spec, FACTS)
    assert kept is not None and "urban" in kept["scene"]
    preset = batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="hybrid_prior")
    assert batch.resolve_prompt(preset, FACTS)[2] == "prior"


class _FakeStore:
    def __init__(self):
        self.updates = []
        self.runs = 0

    def create_run(self, frame, **kwargs):
        self.runs += 1
        return f"run{self.runs}"

    def update_candidate(self, run_id, cand_id, **changes):
        self.updates.append((run_id, cand_id, changes))


class _FakeEngine:
    def __init__(self, fail=()):
        self.store = _FakeStore()
        self.fail = set(fail)

    def generator_availability(self, generator_id):
        return gens.Availability("blocked" if generator_id == "qwen_image_edit" else "ready", "test")

    def frame(self, frame_id):
        if frame_id in self.fail:
            raise KeyError(frame_id)
        return SimpleNamespace(frame_id=frame_id)

    def analyze(self, frame):
        return SimpleNamespace(facts=FACTS, meta=lambda: {})

    def generate(self, run_id, frame, analysis, *, seed_offsets, should_stop, **kwargs):
        self.generate_kwargs = kwargs
        out = []
        for i, seed in enumerate(seed_offsets):
            if should_stop():
                break
            out.append(wf.Candidate(label="b", input="prior", prompt="p", seed_offset=seed, gate={"passed": i == 0}))
        return out


def test_run_job_counts_gates_auto_accepts_and_survives_bad_frames():
    spec = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a", "bad", "c"],
        mode="preset",
        preset="hybrid_prior",
        seed_offsets=[0, 1],
        auto_accept=True,
    )
    engine = _FakeEngine(fail={"bad"})
    reports = []
    result = batch.run_job(1, spec, engine, report=lambda p: reports.append(dict(p)), should_cancel=lambda: False)
    assert result["frames_done"] == 3 and result["candidates"] == 4
    assert result["passed"] == 2 and result["accepted"] == 2
    assert list(result["frame_errors"]) == ["bad"]
    assert len(engine.store.updates) == 2 and not result["cancelled"]
    assert reports[-1]["message"] == "finished"


def test_run_job_stops_on_cancel():
    spec = batch.build_spec(dataset="d", hardware="h", frame_ids=["a", "b"], mode="preset", preset="naive")
    calls = iter([False, True, True, True])
    result = batch.run_job(1, spec, _FakeEngine(), report=lambda p: None, should_cancel=lambda: next(calls))
    assert result["cancelled"] and result["frames_done"] == 1


def test_draw_overlay_marks_boxes_tracks_and_dark_boxes():
    base = Image.new("RGB", (200, 100), (0, 0, 0))
    annotation = {
        "tracks": {
            "0": {
                "relative position": "ego",
                "left rail": {"points": [{"x": 50, "y": 190}, {"x": 90, "y": 120}]},
                "right rail": {"points": [{"x": 150, "y": 190}, {"x": 110, "y": 120}]},
            },
        },
    }
    boxes = [{"label": "switch", "bbox_xyxy": [20, 20, 60, 60]}, {"label": "switch", "bbox_xyxy": [300, 20, 380, 60]}]
    out = overlays.draw_overlay(
        base,
        boxes=boxes,
        box_scale=(0.5, 0.5),
        visible=[True, False],
        annotation=annotation,
        horizon_row=100,
        show_labels=False,
    )
    arr = np.asarray(out)
    assert out.size == base.size
    assert tuple(arr[10, 15]) == overlays.BOX_COLOUR
    assert tuple(arr[10, 150]) == overlays.DARK_BOX_COLOUR
    assert (arr == overlays.EGO_COLOUR).all(axis=-1).any()
    assert (arr == overlays.HORIZON_COLOUR).all(axis=-1).any()
    mask = Image.new("L", (400, 200), 255)
    tinted = np.asarray(overlays.draw_overlay(base, mask=mask))
    assert tinted[..., 1].mean() > tinted[..., 0].mean()


def test_colourise_handles_nan_and_constant():
    arr = np.array([[np.nan, 1.0], [2.0, 3.0]])
    out = np.asarray(overlays.colourise(arr))
    assert out.shape == (2, 2, 3) and tuple(out[0, 0]) == (0, 0, 0)
    assert np.asarray(overlays.colourise(np.ones((3, 3)))).shape == (3, 3, 3)


def test_heartbeat_staleness(tmp_path):
    queue = jobs.JobQueue(tmp_path / "jobs.sqlite")
    queue.heartbeat(5)
    time.sleep(0.05)
    assert queue.live_workers(stale_after=0.01) == []


def test_build_spec_records_generator_params_and_plain_prompts():
    spec = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a"],
        mode="custom",
        prompt="night caption",
        negative_prompt="daylight",
        generator="sd15_controlnet",
        params={"strength": 2.0},
        input_kind="photo",
    )
    assert spec["generator"] == "sd15_controlnet" and spec["params"]["strength"] == 0.95
    assert spec["params"]["steps"] == 30 and spec["negative_prompt"] == "daylight"
    assert batch.resolve_prompt(spec, FACTS) == ("night caption", None, "photo", "daylight")
    klein = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a"],
        mode="custom",
        prompt="p",
        negative_prompt="ignored",
    )
    assert klein["generator"] == "klein4b_instruct" and klein["negative_prompt"] == ""
    preset = batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="caption")
    assert batch.resolve_prompt(preset, FACTS)[3] == wf.CAPTION_NEGATIVE
    with pytest.raises(KeyError):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="naive", generator="x")
    with pytest.raises(ValueError, match="unknown params"):
        batch.build_spec(dataset="d", hardware="h", frame_ids=["a"], mode="preset", preset="naive", params={"bogus": 1})
    json.dumps(spec, allow_nan=False)


def test_run_job_passes_generator_and_fails_fast_when_unavailable():
    spec = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a"],
        mode="preset",
        preset="caption",
        generator="sd15_inpaint",
        params={"strength": 0.5},
    )
    engine = _FakeEngine()
    batch.run_job(1, spec, engine, report=lambda p: None, should_cancel=lambda: False)
    assert engine.generate_kwargs["generator"] == "sd15_inpaint"
    assert engine.generate_kwargs["params"]["strength"] == 0.5
    assert engine.generate_kwargs["negative_prompt"] == wf.CAPTION_NEGATIVE
    blocked = batch.build_spec(
        dataset="d",
        hardware="h",
        frame_ids=["a", "b"],
        mode="preset",
        preset="naive",
        generator="qwen_image_edit",
    )
    with pytest.raises(RuntimeError, match="cannot run here"):
        batch.run_job(2, blocked, _FakeEngine(), report=lambda p: None, should_cancel=lambda: False)
