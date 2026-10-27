"""Tests for the physics-grounded night simulation and its VLM critic."""

import json

import numpy as np
import pytest
from PIL import Image

from aieng.syn_data.image.generate import night
from aieng.syn_data.image.judge import night_critic as critic


CAMERA_YAML = """%YAML:1.0
---
width: 1920.
height: 1080.
f: 2000.0
roll: 0.0
pitch: -2.5
yaw: -1.4

camera_matrix: !!opencv-matrix
    rows: 3
    cols: 3
    dt: d
    data: [2000., 0, 959.5, 0, 2000., 539.5, 0, 0, 1]

distortion_coefficients: [ 0., 0., 0., 0., 0. ]

tvec: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ -650., -2800., 0. ]
"""


@pytest.fixture
def camera(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text(CAMERA_YAML)
    return night.RailCamera.from_opencv_yaml(path)


def _straight_track_annotation(camera, centre_x=0.3, half_gauge=0.75):
    distances = np.array([8.5, 12.0, 18.0, 30.0, 60.0])
    rails = {}
    for side, offset in (("left", -half_gauge), ("right", half_gauge)):
        points = np.stack(
            [
                np.full_like(distances, centre_x + offset),
                np.full_like(distances, camera.height_m),
                distances,
            ],
            axis=-1,
        )
        pixels = camera.project(points)
        rails[f"{side} rail"] = {
            "points": [{"x": float(x), "y": float(y)} for x, y in pixels]
        }
    return {
        "tracks": {"0": {"relative position": "ego", **rails}},
        "tag groups": {"environment": ["rural"], "track_layout": ["straight"]},
    }


def test_camera_yaml_parses_with_validated_sign_convention(camera):
    assert camera.pitch_down_deg == pytest.approx(2.5)
    assert camera.yaw_deg == pytest.approx(1.4)
    assert camera.height_m == pytest.approx(2.8)
    assert camera.horizon_row == pytest.approx(539.5 - 2000 * np.tan(np.radians(2.5)))


def test_ground_projection_round_trip(camera):
    points = np.array([[0.3, camera.height_m, 10.0], [-2.0, camera.height_m, 45.0]])
    pixels = camera.project(points)
    recovered = camera.pixel_to_ground(pixels[:, 0], pixels[:, 1])
    np.testing.assert_allclose(recovered, points, atol=1e-6)
    above = camera.pixel_to_ground(960.0, camera.horizon_row - 5)
    assert np.isnan(above).all()


def test_resized_camera_preserves_geometry(camera):
    small = camera.resized(960, 540)
    point = np.array([0.5, camera.height_m, 20.0])
    full_px = camera.project(point)
    small_px = small.project(point)
    np.testing.assert_allclose((small_px + 0.5) * 2 - 0.5, full_px, atol=1e-6)


def test_gauge_check_recovers_straight_track(camera):
    annotation = _straight_track_annotation(camera)
    rows = night.rail_gauge_check(annotation, camera, rows=(1050, 900, 700, 620))
    assert len(rows) == 4
    for row in rows:
        assert row["gauge_m"] == pytest.approx(1.5, abs=1e-6)
        assert row["centre_x_m"] == pytest.approx(0.3, abs=1e-6)
    assert night.ego_centre_x(annotation, camera) == pytest.approx(0.3, abs=1e-6)


def test_irradiance_falls_off_and_rejects_lamps_above_camera(camera):
    rig = night.HeadlightRig()
    distances = np.array([10.0, 20.0, 40.0, 80.0])
    points = np.stack([np.zeros(4), np.full(4, camera.height_m), distances], axis=-1)
    normals = np.tile([0.0, -1.0, 0.0], (4, 1))
    e = rig.irradiance(points, normals, camera)
    assert np.all(np.diff(e) < 0)
    assert rig.reference_irradiance(camera) > 0
    assert 15 < rig.beam_reach_m(camera) < 300
    with pytest.raises(ValueError, match="below the camera"):
        night.HeadlightRig(lamp_height_m=3.0).lamp_positions(camera)


def test_geometry_without_depth_lights_ground_not_sky(camera):
    small = camera.resized(192, 108)
    geometry = night.build_scene_geometry(small)
    e = night.normalized_irradiance(geometry, night.HeadlightRig(), small)
    assert e[geometry.sky].max() == 0
    assert e[-1, 96] > e[70, 96] > 0
    render = night.render_night(
        Image.new("RGB", (192, 108), (120, 120, 120)), geometry, e
    )
    luma = np.asarray(render).mean(axis=-1)
    assert luma[-1, 96] > luma[5, 96]


def test_night_metrics_flag_phantom_light(camera):
    small = camera.resized(192, 108)
    geometry = night.build_scene_geometry(small)
    e = night.normalized_irradiance(geometry, night.HeadlightRig(), small)
    original = Image.new("RGB", (192, 108), (120, 120, 120))
    clean = night.render_night(original, geometry, e)
    phantom = np.asarray(clean).copy()
    phantom[10:20, 90:100] = 255
    clean_m = night.night_metrics(original, clean, e, geometry)
    phantom_m = night.night_metrics(original, Image.fromarray(phantom), e, geometry)
    assert clean_m["phantom_light"] == 0
    assert phantom_m["phantom_light"] > 5e-4
    assert night.selection_score(phantom_m) < night.selection_score(clean_m)


def test_prompts_are_effect_only_by_default():
    facts = {
        "environment": ["rural"],
        "weather": ["cloudy"],
        "track_layout": ["curve"],
        "n_tracks": 3,
        "n_switches": 2,
        "lamp_height_m": 1.0,
        "beam_reach_m": 40,
    }
    effect = night.compose_prompt(night.physics_prompt_slots(facts)).lower()
    named = night.compose_prompt(
        night.physics_prompt_slots(facts, name_source=True)
    ).lower()
    assert "headlight" not in effect and "lamp" not in effect
    assert "headlight" in named
    assert "2 railway switches" in effect
    with pytest.raises(ValueError, match="Unknown prompt slots"):
        night.compose_prompt({"bogus": "x"})


def test_parse_json_object_is_strict():
    assert critic.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        critic.parse_json_object("no json here")


def test_critique_reports_parse_failure_without_fake_scores(monkeypatch):
    monkeypatch.setattr(critic, "vision_chat", lambda *a, **k: "not json")
    img = Image.new("RGB", (8, 8))
    result = critic.critique_night(img, img, model="m")
    assert result.overall is None and result.error


def test_critique_parses_scores(monkeypatch):
    payload = dict.fromkeys(critic.CRITERIA, 7)
    payload.update(overall=8, critique="pool too bright", prompt_fix="dim it")
    monkeypatch.setattr(critic, "vision_chat", lambda *a, **k: json.dumps(payload))
    img = Image.new("RGB", (8, 8))
    result = critic.critique_night(img, img, model="m")
    assert result.overall == 8 and result.scores["falloff"] == 7
    assert result.prompt_fix == "dim it"


def test_rewrite_keeps_locked_slots_and_rejects_bad_keys(monkeypatch):
    slots = {key: f"{key} text" for key in night.PROMPT_SLOTS}
    proposed = {key: f"new {key}" for key in night.PROMPT_SLOTS}
    response = json.dumps({"slots": proposed, "rationale": "r"})
    monkeypatch.setattr(critic, "vision_chat", lambda *a, **k: response)
    img = Image.new("RGB", (8, 8))
    result = critic.rewrite_prompt(slots, [], img, img, model="m")
    assert result.error is None
    for key in night.LOCKED_SLOTS:
        assert result.slots[key] == slots[key]
    assert result.slots["beam"] == "new beam"
    assert "task" not in result.changed and "beam" in result.changed

    bad = json.dumps({"slots": {"beam": "x"}, "rationale": "r"})
    monkeypatch.setattr(critic, "vision_chat", lambda *a, **k: bad)
    result = critic.rewrite_prompt(slots, [], img, img, model="m")
    assert result.error and result.slots == slots
