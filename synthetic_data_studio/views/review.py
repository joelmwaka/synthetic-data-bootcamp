"""Review: compare candidates with the source, fix box visibility, and decide."""

from __future__ import annotations

import streamlit as st
from aieng.syn_data.image.workflow import generators as gens
from aieng.syn_data.image.workflow import night as wf
from PIL import Image
from studio.overlays import colourise, draw_overlay
from studio.ui import candidate_image, fmt, get_analysis, metrics_row, sidebar


DECISION_ICON = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}

engine = sidebar()
st.title("Review candidates")
runs = engine.store.list_runs()
runs = [r for r in runs if r.get("candidates")]
if not runs:
    st.info("No candidates yet — generate some on the **Generate** or **Batch** page.")
    st.stop()

state = st.session_state
frame_id = state.get("frame_id")
preferred = state.get("review_run") or state.get("run_by_frame", {}).get(frame_id or "")
run_ids = [r["run_id"] for r in runs]
index = run_ids.index(preferred) if preferred in run_ids else 0
by_id = {r["run_id"]: r for r in runs}


def _run_label(run_id: str) -> str:
    run = by_id[run_id]
    cands = run["candidates"]
    pending = sum(c["decision"] == "pending" for c in cands)
    return f"{run['frame']['frame_id']} · {run['kind']} · {len(cands)} cand. ({pending} pending) · {run['created_at']}"


c1, c2 = st.columns([3, 1])
run_id = c1.selectbox("Run", run_ids, index=index, format_func=_run_label)
state["review_run"] = run_id
show = c2.multiselect("Show", ["pending", "accepted", "rejected"], default=["pending", "accepted", "rejected"])

run = engine.store.load_run(run_id)
pool = [wf.Candidate.from_dict(c) for c in run["candidates"]]
ranked = sorted(pool, key=lambda c: -wf.rank_key(c, pool))
visible_pool = [c for c in ranked if c.decision in show]
if not visible_pool:
    st.info("No candidates match the filter.")
    st.stop()

st.dataframe([metrics_row(c, pool) for c in visible_pool], hide_index=True)

cand_ids = [c.candidate_id for c in visible_pool]
cand_key = f"review_cand_{run_id}"
next_cand = state.pop(f"review_next_{run_id}", None)
if next_cand in cand_ids:
    state[cand_key] = next_cand
if state.get(cand_key) not in cand_ids:
    state[cand_key] = cand_ids[0]
by_cand = {c.candidate_id: c for c in visible_pool}
cand = by_cand[
    st.selectbox(
        "Candidate",
        cand_ids,
        key=cand_key,
        format_func=lambda i: (
            f"{DECISION_ICON[by_cand[i].decision]} {by_cand[i].label} · seed +{by_cand[i].seed_offset} "
            f"· rank {wf.rank_key(by_cand[i], pool):.3f}"
        ),
    )
]

frame = None
try:
    frame = engine.frame(run["frame"]["frame_id"])
except KeyError:
    st.warning("The source frame is no longer available in the dataset; showing the stored record only.")

generated = candidate_image(engine, run_id, cand)
src_w, src_h = run["frame"]["size"]
scale = (generated.size[0] / src_w, generated.size[1] / src_h)
boxes = cand.metrics.get("boxes") or []
flags = cand.visible_flags()

cols = st.columns(3)
if frame is not None:
    source = frame.load_image().resize(generated.size, Image.Resampling.LANCZOS)
    cols[0].image(
        draw_overlay(source, boxes=boxes, box_scale=scale, show_labels=False),
        caption="source frame",
        width="stretch",
    )
    analysis = get_analysis(engine, frame, run_if_missing=False)
    if analysis is not None and analysis.work_size == generated.size:
        cols[1].image(
            colourise(analysis.irradiance, log=True),
            caption="headlight irradiance (physics)",
            width="stretch",
        )
    else:
        cols[1].caption("Physics map unavailable (scene not analysed with the current settings).")
cols[2].image(
    draw_overlay(generated, boxes=boxes, box_scale=scale, visible=flags),
    caption="candidate",
    width="stretch",
)

left, right = st.columns([2, 1])
with left:
    st.markdown(
        "**Gate:** " + ("✅ passes" if cand.gate.get("passed") else "⚠️ " + "; ".join(cand.gate.get("reasons", []))),
    )
    crit = cand.critique or {}
    if crit.get("scores"):
        st.markdown(
            f"**VLM {fmt(cand.vlm_overall, '.1f')}** — {crit.get('critique')}  \nfix → {crit.get('prompt_fix')}",
        )
        st.dataframe([crit["scores"]], hide_index=True)
    elif crit.get("error"):
        st.caption(f"VLM unscored: {crit['error'][:200]}")
    gen_spec = gens.GENERATORS.get(cand.generator)
    st.markdown(f"**Generator:** {gen_spec.title if gen_spec else cand.generator}")
    with st.expander("Prompt and generator settings"):
        st.write(cand.prompt)
        if cand.negative_prompt:
            st.caption(f"Negative: {cand.negative_prompt}")
        if cand.slots:
            st.json(cand.slots, expanded=False)
        if cand.generator_params:
            st.json(cand.generator_params, expanded=False)
with right, st.form(f"review_{run_id}_{cand.candidate_id}"):
    st.markdown("**Label visibility at night**")
    new_flags = []
    for i, box in enumerate(boxes):
        default = flags[i] if i < len(flags) else bool(box.get("visible"))
        help_text = f"irradiance {fmt(box.get('irradiance'), '.2f')}, structure {fmt(box.get('structure'), '.2f')}"
        new_flags.append(st.checkbox(f"{i}: {box['label']} visible", value=default, help=help_text))
    if not boxes:
        st.caption("No labelled boxes for this frame.")
    decision = st.radio(
        "Decision",
        ["pending", "accepted", "rejected"],
        index=["pending", "accepted", "rejected"].index(cand.decision),
        horizontal=True,
    )
    note = st.text_input("Note", value=cand.note)
    if st.form_submit_button("Save review", type="primary"):
        engine.store.update_candidate(
            run_id,
            cand.candidate_id,
            decision=decision,
            note=note,
            box_visible=new_flags if boxes else None,
        )
        remaining = [
            c.candidate_id for c in visible_pool if c.decision == "pending" and c.candidate_id != cand.candidate_id
        ]
        if decision != "pending" and remaining:
            state[f"review_next_{run_id}"] = remaining[0]
        st.toast(f"Saved: {decision}")
        st.rerun()
