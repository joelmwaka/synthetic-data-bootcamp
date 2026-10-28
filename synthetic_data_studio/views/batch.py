"""Batch: queue a prompt over many frames; a separate worker process runs the jobs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from aieng.syn_data.image.data.frames import CAP_CALIBRATION
from aieng.syn_data.image.workflow import generators as gens
from aieng.syn_data.image.workflow import night as wf
from studio.batch import build_spec
from studio.engine import vlm_key_available
from studio.jobs import JobQueue
from studio.ui import frame_table, generator_controls, sidebar


WORKER = Path(__file__).resolve().parents[1] / "worker.py"
GATE_LABELS = {
    "min_beam_corr": ("Min. beam correlation", 0.0, 1.0, 0.05),
    "max_phantom_light": ("Max. phantom-light fraction", 0.0, 0.01, 0.0001),
    "min_lit_structure": ("Min. lit structure", 0.0, 1.0, 0.05),
    "min_mean_luma": ("Min. mean luma", 0.0, 0.5, 0.01),
    "max_mean_luma": ("Max. mean luma", 0.0, 0.5, 0.01),
    "min_vlm_overall": ("Min. VLM overall (when scored)", 0.0, 10.0, 0.5),
}

engine = sidebar()
state = st.session_state
st.title("Batch generation")
queue = JobQueue(engine.root / "jobs.sqlite")
rows = [r for r in frame_table(state["dataset"], state["hardware"]) if CAP_CALIBRATION in r["capabilities"]]
if not rows:
    st.info("Batch night synthesis needs calibrated frames (an `l4r_nlb` dataset).")
    st.stop()

st.subheader("1 · Frames")
all_ids = [r["frame_id"] for r in rows]
browse_ids = [f for f in state.get("browse_filtered", []) if f in set(all_ids)]
source = st.radio(
    "Frame selection",
    ["Browse filter", "Frames with boxes", "Pick manually"],
    horizontal=True,
    help="'Browse filter' uses the filters currently set on the Browse page.",
)
if source == "Browse filter":
    candidates = browse_ids or all_ids
elif source == "Frames with boxes":
    candidates = [r["frame_id"] for r in rows if r["n_boxes"] > 0]
else:
    candidates = st.multiselect(
        "Frames",
        all_ids,
        default=[state["frame_id"]] if state.get("frame_id") in all_ids else [],
    )
limit = st.number_input("Max. frames", 1, max(1, len(candidates)), min(16, max(1, len(candidates))))
frame_ids = candidates[: int(limit)]
st.caption(
    f"{len(frame_ids)} frame(s) selected" + (f" (of {len(candidates)})" if len(candidates) > len(frame_ids) else ""),
)

st.subheader("2 · Prompt")
mode = st.radio(
    "Prompt source",
    ["preset", "custom"],
    horizontal=True,
    format_func=lambda m: {"preset": "Preset (rebuilt per frame)", "custom": "From a reviewed candidate"}[m],
)
preset = None
slots = None
custom_prompt = None
negative = ""
input_kind = "prior"
refresh_scene = True
default_generator = gens.DEFAULT_GENERATOR
chosen_params: dict[str, float | int] = {}
if mode == "preset":
    preset = st.selectbox(
        "Preset",
        list(wf.PRESET_DESCRIPTIONS),
        index=3,
        format_func=lambda n: f"{n} — {wf.PRESET_DESCRIPTIONS[n]}",
    )
else:
    options = [(run["run_id"], c) for run in engine.store.list_runs() for c in run.get("candidates", [])]
    if not options:
        st.warning("No candidates yet — generate one on the Generate page.")
        st.stop()
    pick = st.selectbox(
        "Candidate",
        range(len(options)),
        format_func=lambda i: (
            f"{options[i][1]['label']} · {options[i][1].get('generator', gens.DEFAULT_GENERATOR)} · "
            f"{options[i][1]['decision']} · physics {options[i][1]['physics_score']:.3f} · run {options[i][0][-24:]}"
        ),
    )
    chosen = wf.Candidate.from_dict(options[int(pick)][1])
    input_kind = chosen.input
    negative = chosen.negative_prompt
    default_generator = chosen.generator if chosen.generator in gens.GENERATORS else gens.DEFAULT_GENERATOR
    chosen_params = dict(chosen.generator_params)
    if chosen.slots:
        slots = dict(chosen.slots)
        refresh_scene = st.checkbox("Refresh the factual `scene` slot for each frame", value=True)
        with st.expander("Slots"):
            st.json(slots)
    else:
        custom_prompt = chosen.prompt
        with st.expander("Prompt"):
            st.write(custom_prompt)
            if negative:
                st.caption(f"Negative: {negative}")

st.subheader("3 · Generator")
gen_key = f"batch_{options[int(pick)][1]['candidate_id']}" if mode == "custom" else "batch_preset"
if mode == "custom" and f"{gen_key}_generator" not in state:
    # Start from the chosen candidate's generator and parameters.
    state[f"{gen_key}_generator"] = default_generator
    for name, value in gens.get_generator(default_generator).resolve_params(chosen_params).items():
        state[f"{gen_key}_{default_generator}_{name}"] = value
gen_spec, gen_params, gen_status = generator_controls(engine, gen_key, default=default_generator)
if mode == "preset" and preset != gen_spec.recommended_preset:
    st.caption(f"Tip: `{gen_spec.recommended_preset}` is the recommended preset for this generator.")

c1, c2, c3 = st.columns(3)
n_seeds = c1.number_input("Seeds per frame", 1, 6, 2)
use_vlm = c2.checkbox(
    "VLM critic",
    value=False,
    disabled=not vlm_key_available(),
    help="Adds an API call per candidate." if vlm_key_available() else "No judge API key.",
)
auto_accept = c3.checkbox(
    "Auto-accept candidates that pass all gates",
    value=False,
    help="Otherwise everything stays pending for human review.",
)
with st.expander("Accept gates"):
    gates = {}
    gcols = st.columns(3)
    for i, (key, (label, lo, hi, step)) in enumerate(GATE_LABELS.items()):
        gates[key] = gcols[i % 3].number_input(label, lo, hi, float(wf.DEFAULT_GATES[key]), step, format="%.4f")

est = len(frame_ids) * int(n_seeds)
st.caption(
    f"≈ {est} edits with {gen_spec.title} (+ model load, and scene analysis for frames not analysed yet).",
)
if st.button("Queue batch job", type="primary", disabled=not frame_ids or not gen_status.usable):
    try:
        spec = build_spec(
            dataset=state["dataset"],
            hardware=state["hardware"],
            frame_ids=frame_ids,
            mode=mode,
            preset=preset,
            slots=slots,
            prompt=custom_prompt,
            negative_prompt=negative,
            generator=gen_spec.id,
            params=gen_params,
            input_kind=input_kind,
            refresh_scene=refresh_scene,
            seed_offsets=list(range(int(n_seeds))),
            use_vlm=use_vlm,
            gates=gates,
            auto_accept=auto_accept,
            label=f"batch_{preset or 'custom'}_{gen_spec.id}",
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.success(f"Queued job {queue.submit(spec)}.")

st.divider()
st.subheader("4 · Worker and jobs")


@st.fragment(run_every=3)
def jobs_panel() -> None:
    """Live worker status and job table (refreshes every few seconds)."""
    workers = queue.live_workers()
    w1, w2 = st.columns([3, 1])
    if workers:
        w = workers[0]
        w1.success(f"Worker running (pid {w['pid']}) · current job: {w['current_job'] or 'idle'}")
    else:
        w1.warning("No worker running. Queued jobs wait until one starts.")
        if w2.button("Start worker", width="stretch"):
            if engine.comparer_loaded:
                engine.unload()  # avoid two Klein copies in RAM
            log = engine.root / "worker.log"
            with log.open("a") as fh:
                subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                    [sys.executable, str(WORKER), "--dataset", state["dataset"], "--hardware", state["hardware"]],
                    cwd=str(WORKER.parent),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=os.environ.copy(),
                )
            time.sleep(1.0)
            st.rerun(scope="fragment")
    jobs = queue.recent(limit=20)
    if not jobs:
        st.caption("No jobs yet.")
        return
    for job in jobs:
        p = job.progress
        total = p.get("frames_total") or len(job.spec.get("frame_ids", []))
        done = p.get("frames_done", 0)
        with st.container(border=True):
            a, b = st.columns([4, 1])
            a.markdown(
                f"**Job {job.id}** · `{job.status}` · {total} frames × {len(job.spec.get('seed_offsets', []))} seeds · "
                f"{job.spec.get('mode')}: {job.spec.get('preset') or 'custom'} · "
                f"{job.spec.get('generator', gens.DEFAULT_GENERATOR)}",
            )
            if job.status in ("queued", "running") and not job.cancel_requested:
                if b.button("Cancel", key=f"cancel_{job.id}", width="stretch"):
                    queue.request_cancel(job.id)
                    st.rerun(scope="fragment")
            elif job.cancel_requested and job.status == "running":
                b.caption("cancelling…")
            if job.status in ("running", "done", "cancelled", "failed") and total:
                a.progress(min(1.0, done / total), text=p.get("message", ""))
            a.caption(
                f"candidates {p.get('candidates', 0)} · passed gates {p.get('passed', 0)} · "
                f"auto-accepted {p.get('accepted', 0)}",
            )
            if p.get("frame_errors"):
                a.error("Frame errors: " + "; ".join(f"{k}: {v}" for k, v in p["frame_errors"].items()))
            if job.error:
                a.error(job.error)


jobs_panel()
st.caption(f"Worker log: `{engine.root / 'worker.log'}` · review results on the **Review** page.")
