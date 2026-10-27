"""Tests for frame adapters and the shared night workflow (no GPU needed)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from aieng.syn_data.image.data import frames
from aieng.syn_data.image.generate import night
from aieng.syn_data.image.workflow import night as wf
from aieng.syn_data.image.workflow.store import RunStore


CAMERA_YAML = """%YAML:1.0
---
width: 192.
height: 108.
f: 200.0
roll: 0.0
pitch: -2.5
yaw: -1.4
camera_matrix: !!opencv-matrix
    rows: 3
    cols: 3
    dt: d
    data: [200., 0, 95.5, 0, 200., 53.5, 0, 0, 1]
tvec: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ -650., -2800., 0. ]
"""


def _annotation(switches):
    return {
        "switches": switches,
        "tag groups": {
            "environment": ["rural"],
            "weather": ["cloudy"],
            "track_layout": ["straight"],
        },
        "tracks": {
            "0": {
                "relative position": "ego",
                "left rail": {"points": [{"x": 80, "y": 107}, {"x": 94, "y": 60}]},
                "right rail": {"points": [{"x": 115, "y": 107}, {"x": 97, "y": 60}]},
            },
        },
    }


@pytest.fixture
def l4r_dir(tmp_path):
    root = tmp_path / "L4R_NLB_fall"
    for sub in ("images", "annotations", "masks", "camera"):
        (root / sub).mkdir(parents=True)
    (root / "camera" / "camera.yaml").write_text(CAMERA_YAML)
    switches = {
        "0": {
            "kind": "fork",
            "direction": "left",
            "marks": [{"x": 100, "y": 90}, {"x": 80, "y": 70}],
        },
        "1": {
            "kind": "fork",
            "direction": "left",
            "marks": [{"x": 100, "y": 90}, {"x": 80, "y": 70}],
        },
        "2": {"kind": "merge", "direction": "unknown", "marks": []},
    }
    for fid, sw in (("nlb_fall_000001", switches), ("nlb_fall_000002", {})):
        Image.new("RGB", (192, 108), (110, 120, 130)).save(
            root / "images" / f"{fid}.png"
        )
        (root / "annotations" / f"{fid}.json").write_text(json.dumps(_annotation(sw)))
        mask = np.zeros((108, 192), np.uint8)
        mask[60:, 80:116] = 255
        Image.fromarray(mask).save(root / "masks" / f"{fid}.png")
    Image.new("RGB", (192, 108)).save(
        root / "images" / "nlb_fall_000003.png"
    )  # no annotation
    return root


def test_switch_boxes_normalise_dedupe_and_count_issues():
    boxes, issues = frames.switch_boxes(
        _annotation(
            {
                "a": {"kind": "fork", "marks": [{"x": 10, "y": 9}, {"x": 2, "y": 3}]},
                "b": {"kind": "fork", "marks": [{"x": 2, "y": 3}, {"x": 10, "y": 9}]},
                "c": {"marks": [{"x": 1, "y": 1}]},
            },
        ),
        (20, 20),
    )
    assert [b["bbox_xyxy"] for b in boxes] == [[2.0, 3.0, 10.0, 9.0]]
    assert boxes[0]["kind"] == "fork"
    assert issues == {"duplicate_boxes": 1, "switches_without_two_marks": 1}
    with pytest.raises(ValueError, match="outside image"):
        frames.switch_boxes(
            {"switches": {"x": {"marks": [{"x": 0, "y": 0}, {"x": 50, "y": 5}]}}},
            (20, 20),
        )


def test_l4r_source_lists_annotated_frames_with_capabilities(l4r_dir):
    source = frames.L4RSource(l4r_dir)
    assert source.frame_ids() == ["nlb_fall_000001", "nlb_fall_000002"]
    frame = source.get("nlb_fall_000001")
    assert len(frame.boxes) == 1
    assert frame.capabilities == {"boxes", "tracks", "mask", "tags", "calibration"}
    assert frame.tags["environment"] == ("rural",)
    assert "boxes" not in source.get("nlb_fall_000002").capabilities
    with pytest.raises(KeyError):
        source.get("nlb_fall_000003")


def test_samples_source_reads_filename_keyed_labels(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "cone_a.jpg")
    Image.new("RGB", (10, 10)).save(tmp_path / "scene_b.jpg")
    (tmp_path / "labels.json").write_text(
        json.dumps({"cone_a.jpg": [{"label": "cone", "bbox_xyxy": [1, 2, 3, 4]}]}),
    )
    source = frames.SamplesSource(tmp_path, dataset="toy")
    assert source.frame_ids() == ["cone_a", "scene_b"]
    assert source.get("cone_a").boxes[0]["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert source.get("scene_b").capabilities == {"tags"}


def test_frame_source_for_picks_adapter(l4r_dir, tmp_path):
    cfg = {"dataset_name": "l4r_nlb", "data": {"raw_dir": str(l4r_dir)}}
    assert isinstance(frames.frame_source_for(cfg), frames.L4RSource)
    cfg = SimpleNamespace(paths=SimpleNamespace(samples_dir=str(tmp_path)))
    cfg_dict = {"dataset_name": "toy", "data": {}}

    class _Cfg(dict):
        paths = cfg.paths

    assert isinstance(frames.frame_source_for(_Cfg(cfg_dict)), frames.SamplesSource)


class _FakeDepth:
    calls = 0

    def predict(self, image):
        _FakeDepth.calls += 1
        h = image.size[1]
        rows = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        return SimpleNamespace(depth_map=np.repeat(rows, image.size[0], axis=1))


class _FakeSeg:
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "wall", 2: "sky"}))

    def predict(self, image):
        w, h = image.size
        labels = np.zeros((h, w), np.int32)
        labels[: h // 3] = 2
        return SimpleNamespace(label_map=labels)


def _analysis(frame, tmp_path):
    cache = wf.GeometryCache(tmp_path / "cache")
    return wf.analyze_frame(
        frame,
        work_size=(96, 54),
        load_models=lambda: (_FakeDepth(), _FakeSeg()),
        cache=cache,
        model_key="fake",
    ), cache


def test_analyze_frame_caches_geometry(l4r_dir, tmp_path):
    frame = frames.L4RSource(l4r_dir).get("nlb_fall_000001")
    _FakeDepth.calls = 0
    first, _ = _analysis(frame, tmp_path)
    second, _ = _analysis(frame, tmp_path)
    assert _FakeDepth.calls == 1
    np.testing.assert_allclose(
        first.irradiance, second.irradiance, rtol=1e-5, atol=1e-7
    )
    assert first.prior.size == (96, 54)
    assert first.facts["n_switches"] == 1
    assert json.dumps(first.meta(), allow_nan=False)
    with pytest.raises(RuntimeError, match="not cached"):
        wf.analyze_frame(
            frame, work_size=(80, 45), cache=wf.GeometryCache(tmp_path / "other")
        )


def test_analyze_frame_requires_calibration(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
    frame = frames.SamplesSource(tmp_path, dataset="toy").get("a")
    with pytest.raises(ValueError, match="calibration"):
        wf.analyze_frame(frame)


def test_gates_and_ranking():
    good = {
        "beam_corr": 0.8,
        "phantom_light": 0.0,
        "lit_structure": 0.4,
        "mean_luma": 0.1,
        "clipped_lit": 0.0,
    }
    assert wf.evaluate_gates(good, None)["passed"]
    assert not wf.evaluate_gates(good, 3.0)["passed"]
    bad = wf.evaluate_gates({**good, "phantom_light": 0.01, "beam_corr": None}, None)
    assert not bad["passed"] and len(bad["reasons"]) == 2
    a = wf.Candidate(
        label="a",
        input="photo",
        prompt="p",
        metrics=good,
        physics_score=0.7,
        vlm_overall=9.0,
    )
    b = wf.Candidate(
        label="b", input="photo", prompt="p", metrics=good, physics_score=0.8
    )
    assert wf.rank_key(a, [a, b]) == 0.7
    assert wf.rank_key(a, [a]) == night.selection_score(good, 9.0)


def test_score_candidate_survives_critic_failure(l4r_dir, tmp_path):
    frame = frames.L4RSource(l4r_dir).get("nlb_fall_000001")
    analysis, _ = _analysis(frame, tmp_path)

    def _boom(original, candidate):
        raise OSError("no key")

    cand = wf.score_candidate(
        wf.Candidate(label="x", input="prior", prompt="p"),
        analysis.prior,
        frame,
        analysis,
        critic=_boom,
    )
    assert cand.vlm_overall is None and "OSError" in cand.critique["error"]
    assert len(cand.metrics["boxes"]) == 1 and "passed" in cand.gate
    assert json.dumps(cand.to_dict(), allow_nan=False)


def test_presets_cover_ablation():
    facts = {
        "environment": [],
        "weather": [],
        "track_layout": [],
        "n_tracks": 1,
        "n_switches": 0,
        "lamp_height_m": 1.0,
        "beam_reach_m": 40,
    }
    presets = wf.prompt_presets(facts)
    assert set(presets) == set(wf.PRESET_DESCRIPTIONS)
    assert (
        presets["hybrid_prior"]["input"] == "prior"
        and presets["naive"]["slots"] is None
    )
    assert all(p["prompt"] for p in presets.values())


def test_run_store_review_and_export(l4r_dir, tmp_path):
    frame = frames.L4RSource(l4r_dir).get("nlb_fall_000001")
    analysis, _ = _analysis(frame, tmp_path)
    store = RunStore(tmp_path / "studio")
    run_id = store.create_run(
        frame, kind="interactive", settings={"seed": 42}, analysis=analysis.meta()
    )
    assert (
        store.create_run(frame, kind="interactive", settings={}, run_id=run_id)
        == run_id
    )
    cand = wf.score_candidate(
        wf.Candidate(label="x", input="prior", prompt="p"),
        analysis.prior,
        frame,
        analysis,
    )
    store.add_candidate(run_id, cand, analysis.prior)
    with pytest.raises(ValueError, match="Duplicate"):
        store.add_candidate(run_id, cand, analysis.prior)
    with pytest.raises(ValueError, match="No accepted"):
        store.export_accepted()
    with pytest.raises(ValueError, match="Not reviewable"):
        store.update_candidate(run_id, cand.candidate_id, prompt="hack")
    with pytest.raises(ValueError, match="flags"):
        store.update_candidate(run_id, cand.candidate_id, box_visible=[True, False])
    updated = store.update_candidate(
        run_id, cand.candidate_id, decision="accepted", note="ok", box_visible=[True]
    )
    assert updated.decision == "accepted" and updated.visible_flags() == [True]
    assert [r["run_id"] for r in store.list_runs()] == [run_id]

    dest = store.export_accepted(name="exp1")
    labels = json.loads((dest / "labels.json").read_text())
    ((fname, boxes),) = labels.items()
    assert (dest / "images" / fname).is_file()
    # source box [80, 70, 100, 90] at 192x108 → work grid 96x54 is half size
    assert boxes == [
        {
            "label": "switch",
            "bbox_xyxy": [40.0, 35.0, 50.0, 45.0],
            "visible_at_night": True,
        }
    ]
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov[0]["candidate"]["prompt"] == "p" and prov[0]["settings"] == {"seed": 42}
    with pytest.raises(FileExistsError):
        store.export_accepted(name="exp1")
    with pytest.raises(ValueError):
        store.load_run("../etc")

    store.update_candidate(run_id, cand.candidate_id, box_visible=[False])
    dest2 = store.export_accepted(name="exp2")
    assert next(iter(json.loads((dest2 / "labels.json").read_text()).values())) == []
