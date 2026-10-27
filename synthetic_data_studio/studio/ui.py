"""Streamlit glue: cached engine/analysis, sidebar, selection state, thumbnails."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import streamlit as st
from aieng.syn_data.image.config import configs_dir, list_dataset_names
from aieng.syn_data.image.data.frames import CAP_CALIBRATION, Frame
from aieng.syn_data.image.data.loader import project_root
from aieng.syn_data.image.workflow import night as wf
from PIL import Image

from studio.engine import Engine, GpuBusyError, load_cfg, vlm_key_available
from studio.overlays import thumbnail


DEFAULT_DATASET = "l4r_nlb"
DEFAULT_HARDWARE = "gpu_l4"
RIG_FIELDS = (
    ("lamp_height_m", "Lamp height above rails (m)", 0.4, 2.2, 0.1),
    ("lamp_spacing_m", "Lamp spacing (m)", 0.6, 2.4, 0.1),
    ("aim_down_deg", "Beam aim below horizontal (°)", -1.0, 4.0, 0.25),
    ("sigma_h_deg", "Horizontal beam width σ (°)", 2.0, 20.0, 0.5),
    ("sigma_down_deg", "Downward beam spread σ (°)", 1.0, 15.0, 0.5),
)


@st.cache_resource(show_spinner=False)
def impl_root() -> Path:
    """``implementations/edge_case_image_generation`` (configs, data, outputs)."""
    return project_root(Path(__file__).resolve())


@st.cache_resource(show_spinner="Loading configuration…")
def get_engine(dataset: str, hardware: str) -> Engine:
    """One engine (and Klein pipeline) per dataset/hardware for the server process."""
    return Engine(load_cfg(dataset, hardware, start=impl_root()))


def hardware_profiles() -> list[str]:
    """Available hardware profiles."""
    return sorted(p.stem for p in (configs_dir(impl_root()) / "hardware").glob("*.yaml"))


def current_engine() -> Engine:
    """Engine for the dataset/hardware chosen in the sidebar."""
    return get_engine(st.session_state["dataset"], st.session_state["hardware"])


def _init_state() -> None:
    state = st.session_state
    if "dataset" not in state:
        state["dataset"] = st.query_params.get("dataset", DEFAULT_DATASET)
    if "hardware" not in state:
        state["hardware"] = DEFAULT_HARDWARE
    if "frame_id" not in state:
        state["frame_id"] = st.query_params.get("frame")


def select_frame(frame_id: str | None) -> None:
    """Set the selected frame (also mirrored to the URL for deep links)."""
    st.session_state["frame_id"] = frame_id
    if frame_id:
        st.query_params["frame"] = frame_id
    elif "frame" in st.query_params:
        del st.query_params["frame"]


def _on_dataset_change() -> None:
    select_frame(None)
    st.query_params["dataset"] = st.session_state["dataset"]


def sidebar() -> Engine:
    """Render shared sidebar controls and return the active engine."""
    _init_state()
    datasets = list_dataset_names(start=impl_root())
    if st.session_state["dataset"] not in datasets:
        st.session_state["dataset"] = DEFAULT_DATASET if DEFAULT_DATASET in datasets else datasets[0]
    profiles = hardware_profiles()
    with st.sidebar:
        st.selectbox("Dataset", datasets, key="dataset", on_change=_on_dataset_change)
        st.selectbox("Hardware profile", profiles, key="hardware")
        engine = current_engine()
        frame_id = st.session_state.get("frame_id")
        st.caption(f"Selected frame: **{frame_id}**" if frame_id else "No frame selected")
        st.divider()
        if vlm_key_available():
            st.caption(f"VLM critic: `{engine.cfg.judge.model_id}` ✓")
        else:
            st.caption("VLM critic: no API key in the environment — physics metrics only")
        if engine.comparer_loaded:
            st.caption("Generator: FLUX.2 Klein loaded in this app")
            if st.button("Unload generator", help="Free GPU/CPU memory, e.g. before starting a batch worker"):
                engine.unload()
                st.rerun()
        else:
            st.caption("Generator: not loaded (loads on first generation, ~1 min)")
        st.caption(f"Outputs: `{engine.root.relative_to(impl_root())}`")
    return engine


@st.cache_data(show_spinner=False, max_entries=4000)
def thumb_bytes(path: str, mtime: float, max_side: int = 360) -> bytes:
    """JPEG thumbnail bytes (cache key includes the file mtime)."""
    with Image.open(path) as image:
        thumb = thumbnail(image, max_side)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def frame_thumb(path: Path, max_side: int = 360) -> bytes:
    """Thumbnail for a file path."""
    return thumb_bytes(str(path), path.stat().st_mtime, max_side)


@st.cache_data(show_spinner="Indexing frames…")
def frame_table(dataset: str, hardware: str) -> list[dict[str, Any]]:
    """Summary rows for every frame (for filtering in Browse / Batch)."""
    engine = get_engine(dataset, hardware)
    rows = []
    for frame_id in engine.source.frame_ids():
        frame = engine.frame(frame_id)
        rows.append({**frame.summary(), "image_path": str(frame.image_path)})
    return rows


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_analysis(
    dataset: str,
    hardware: str,
    frame_id: str,
    rig_items: tuple[tuple[str, float], ...],
) -> wf.SceneAnalysis:
    """Scene analysis per frame and rig setting (geometry itself is cached on disk)."""
    engine = get_engine(dataset, hardware)
    return engine.analyze(engine.frame(frame_id), rig_overrides=dict(rig_items) or None)


def rig_items() -> tuple[tuple[str, float], ...]:
    """Return the Inspect-page rig overrides as a hashable tuple."""
    overrides: dict[str, float] = st.session_state.get("rig_overrides") or {}
    return tuple(sorted(overrides.items()))


def require_frame(engine: Engine) -> Frame | None:
    """Return the selected frame, or show a hint and return ``None``."""
    frame_id = st.session_state.get("frame_id")
    if not frame_id:
        st.info("Select a frame on the **Browse** page first.")
        st.page_link("views/browse.py", label="Go to Browse", icon=":material/photo_library:")
        return None
    try:
        return engine.frame(frame_id)
    except KeyError:
        st.warning(f"Frame `{frame_id}` is not in dataset `{st.session_state['dataset']}`.")
        select_frame(None)
        return None


def get_analysis(engine: Engine, frame: Frame, *, run_if_missing: bool) -> wf.SceneAnalysis | None:
    """Return the cached analysis, computing it only when ``run_if_missing``."""
    if CAP_CALIBRATION not in frame.capabilities:
        return None
    if not run_if_missing and not engine.is_analyzed(frame):
        return None
    try:
        with st.spinner("Analysing scene (depth + segmentation on first run)…"):
            return cached_analysis(
                st.session_state["dataset"],
                st.session_state["hardware"],
                frame.frame_id,
                rig_items(),
            )
    except GpuBusyError as exc:
        st.error(str(exc))
        return None


def candidate_image(engine: Engine, run_id: str, cand: wf.Candidate) -> Image.Image:
    """Load a stored candidate image."""
    with Image.open(engine.store.image_path(run_id, cand)) as image:
        return image.convert("RGB")


def fmt(value: Any, spec: str = ".3f") -> str:
    """Format a possibly-missing number."""
    return "–" if value is None else format(float(value), spec)


def metrics_row(cand: wf.Candidate, pool: list[wf.Candidate]) -> dict[str, Any]:
    """Flat metrics dict for tables."""
    m = cand.metrics
    return {
        "label": cand.label,
        "seed": cand.seed_offset,
        "input": cand.input,
        "rank": round(wf.rank_key(cand, pool), 3),
        "physics": round(cand.physics_score, 3),
        "vlm": cand.vlm_overall,
        "beam_corr": m.get("beam_corr"),
        "phantom_light": m.get("phantom_light"),
        "lit_structure": m.get("lit_structure"),
        "clipped_lit": m.get("clipped_lit"),
        "mean_luma": m.get("mean_luma"),
        "gate": "pass" if cand.gate.get("passed") else "fail",
        "decision": cand.decision,
        "id": cand.candidate_id,
    }
