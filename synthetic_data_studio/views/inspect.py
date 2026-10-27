"""Inspect: annotations, calibration, scene geometry, and the headlight prior."""

from __future__ import annotations

import numpy as np
import streamlit as st
from aieng.syn_data.image.data.frames import CAP_CALIBRATION
from aieng.syn_data.image.generate import night
from PIL import Image
from studio.overlays import class_map, colourise, draw_overlay
from studio.ui import RIG_FIELDS, get_analysis, require_frame, sidebar


DISPLAY_WIDTH = 1280

engine = sidebar()
frame = require_frame(engine)
if frame is None:
    st.stop()

st.title(f"Inspect · {frame.frame_id}")
caps = sorted(frame.capabilities)
st.caption("Capabilities: " + (" · ".join(f"`{c}`" for c in caps) if caps else "none (image only)"))

image = frame.load_image()
scale = DISPLAY_WIDTH / image.size[0]
display = image.resize((DISPLAY_WIDTH, round(image.size[1] * scale)), Image.Resampling.LANCZOS)
camera = night.RailCamera.from_opencv_yaml(frame.camera_path) if frame.camera_path else None

left, right = st.columns([3, 1])
with right:
    st.subheader("Overlays")
    show_boxes = st.toggle("Boxes", value=True, disabled="boxes" not in caps)
    show_tracks = st.toggle("Track polylines", value=True, disabled="tracks" not in caps)
    show_mask = st.toggle("Track mask", value=False, disabled="mask" not in caps)
    show_horizon = st.toggle("Rail-plane horizon", value=camera is not None, disabled=camera is None)
with left:
    overlay = draw_overlay(
        display,
        boxes=list(frame.boxes) if show_boxes else None,
        box_scale=(scale, scale),
        annotation=frame.annotation if show_tracks else None,
        mask=frame.load_mask() if show_mask else None,
        horizon_row=camera.horizon_row if (camera and show_horizon) else None,
    )
    st.image(overlay, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Labels")
    if frame.boxes:
        st.dataframe(
            [
                {
                    "#": i,
                    "label": b["label"],
                    "kind": b.get("kind", ""),
                    "direction": b.get("direction", ""),
                    "x1,y1,x2,y2": ", ".join(f"{v:.0f}" for v in b["bbox_xyxy"]),
                }
                for i, b in enumerate(frame.boxes)
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No boxes for this frame.")
with c2:
    st.subheader("Scene tags")
    if frame.tags:
        st.dataframe(
            [{"group": g, "values": ", ".join(v)} for g, v in frame.tags.items() if v],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No tags.")

if CAP_CALIBRATION not in frame.capabilities or camera is None:
    st.info("This frame has no camera calibration, so the night-headlight physics is unavailable.")
    st.stop()

st.divider()
st.subheader("Calibration check")
gauge = night.rail_gauge_check(frame.annotation or {}, camera)
if gauge:
    gauges = np.array([r["gauge_m"] for r in gauge])
    ok = bool(np.all(np.abs(gauges - 1.5) < 0.15))
    (st.success if ok else st.warning)(
        f"Ego-rail gauge {gauges.mean():.3f} ± {gauges.std():.3f} m over {gauges.size} distances "
        + ("— calibration fits this frame." if ok else "— drifts; the physics prior is unreliable here."),
    )
    st.dataframe(
        [{k: round(v, 2) for k, v in r.items()} for r in gauge],
        hide_index=True,
    )
else:
    st.caption("No ego track annotated — lamps default to the camera centre line.")

st.divider()
st.subheader("Scene analysis & headlight physics prior")
with st.expander("Headlight rig (applies to Generate and Review)"):
    overrides: dict[str, float] = dict(st.session_state.get("rig_overrides") or {})
    defaults = night.HeadlightRig()
    cols = st.columns(len(RIG_FIELDS))
    new = {}
    for col, (name, label, lo, hi, step) in zip(cols, RIG_FIELDS):
        value = float(overrides.get(name, getattr(defaults, name)))
        new[name] = col.slider(label, lo, hi, value, step)
    changed = {k: v for k, v in new.items() if abs(v - getattr(defaults, k)) > 1e-9}
    if changed != overrides:
        st.session_state["rig_overrides"] = changed
    if changed and st.button("Reset to defaults"):
        st.session_state["rig_overrides"] = {}
        st.rerun()

analysis = get_analysis(engine, frame, run_if_missing=False)
if analysis is None:
    st.caption("Geometry not computed yet for this frame (depth + segmentation, ~10–20 s the first time).")
    if st.button("Run scene analysis", type="primary"):
        analysis = get_analysis(engine, frame, run_if_missing=True)
if analysis is None:
    st.stop()

fit = analysis.geometry.depth_fit
g = analysis.geometry
st.caption(
    f"Depth aligned to the rail plane on {fit['n']:.0f} track pixels (rmse {fit['rmse']:.4f}) · "
    f"ground {g.ground.mean():.0%} · objects {(~g.ground & ~g.sky).mean():.0%} · sky {g.sky.mean():.0%} · "
    f"beam reach {analysis.facts['beam_reach_m']} m",
)
distance = np.linalg.norm(g.points, axis=-1)
row1 = st.columns(3)
row1[0].image(colourise(analysis.disparity, cmap="magma"), caption="Depth Anything (relative)", width="stretch")
row1[1].image(class_map(g), caption="sky / ground / vertical objects", width="stretch")
row1[2].image(
    colourise(np.nan_to_num(distance, nan=np.inf), cmap="viridis_r", log=True),
    caption="metric distance (log)",
    width="stretch",
)
row2 = st.columns(2)
row2[0].image(
    colourise(analysis.irradiance, log=True),
    caption="headlight irradiance (log; 1.0 = track at 15 m)",
    width="stretch",
)
row2[1].image(
    draw_overlay(analysis.prior, boxes=list(frame.boxes), box_scale=analysis.box_scale),
    caption="physics prior (night render)",
    width="stretch",
)
prior_metrics = night.night_metrics(
    image,
    analysis.prior,
    analysis.irradiance,
    g,
    boxes=list(frame.boxes),
    box_scale=analysis.box_scale,
)
st.dataframe(
    [
        {"box": i, "label": b["label"], "irradiance": round(b["irradiance"], 3), "visible_at_night": b["visible"]}
        for i, b in enumerate(prior_metrics["boxes"])
    ],
    hide_index=True,
)
st.page_link("views/generate.py", label="Generate night variants for this frame →", icon=":material/auto_awesome:")
