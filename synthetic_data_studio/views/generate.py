"""Generate: edit slot prompts, generate seeds, score, and refine with the VLM."""

from __future__ import annotations

import streamlit as st
from aieng.syn_data.image.data.frames import CAP_CALIBRATION
from aieng.syn_data.image.generate import night
from aieng.syn_data.image.judge.night_critic import rewrite_prompt
from aieng.syn_data.image.workflow import night as wf
from studio.engine import GpuBusyError, vlm_key_available
from studio.overlays import draw_overlay
from studio.ui import candidate_image, fmt, get_analysis, metrics_row, require_frame, sidebar


engine = sidebar()
frame = require_frame(engine)
if frame is None:
    st.stop()
st.title(f"Generate · {frame.frame_id}")
if CAP_CALIBRATION not in frame.capabilities:
    st.info(
        "Night-headlight synthesis needs camera calibration (Labels4Rail frames). "
        "This frame/dataset has none — pick an `l4r_nlb` frame.",
    )
    st.stop()

analysis = get_analysis(engine, frame, run_if_missing=False)
if analysis is None:
    st.caption("Scene geometry is not computed yet for this frame.")
    if st.button("Run scene analysis", type="primary"):
        analysis = get_analysis(engine, frame, run_if_missing=True)
if analysis is None:
    st.stop()

state = st.session_state
fid = frame.frame_id
presets = wf.prompt_presets(analysis.facts)
slot_key = {name: f"slot_{fid}_{name}" for name in night.PROMPT_SLOTS}
raw_key = f"raw_{fid}"
input_key = f"input_{fid}"
runs_by_frame: dict[str, str] = state.setdefault("run_by_frame", {})


def load_preset(name: str) -> None:
    """Fill the editor with a preset (call before widgets render)."""
    preset = presets[name]
    state[input_key] = preset["input"]
    state[f"mode_{fid}"] = "slots" if preset["slots"] else "raw"
    if preset["slots"]:
        for slot, text in preset["slots"].items():
            state[slot_key[slot]] = text
    state[raw_key] = preset["prompt"]
    state[f"label_{fid}"] = name


if f"mode_{fid}" not in state:
    load_preset("hybrid_prior")
pending_input = state.pop(f"pending_input_{fid}", None)
if pending_input:
    state[input_key] = pending_input
pending = state.pop(f"pending_slots_{fid}", None)
if pending:
    for slot, text in pending.items():
        state[slot_key[slot]] = text
    state[f"mode_{fid}"] = "slots"

c_img, c_prompt = st.columns([2, 3])
with c_img:
    st.image(
        draw_overlay(analysis.prior, boxes=list(frame.boxes), box_scale=analysis.box_scale, show_labels=False),
        caption="physics prior (input when 'prior' is selected)",
        width="stretch",
    )
    st.caption(
        f"Facts used in prompts: {', '.join(analysis.facts['environment']) or '–'} · "
        f"{analysis.facts['n_tracks']} tracks · {analysis.facts['n_switches']} switches · "
        f"beam reach {analysis.facts['beam_reach_m']} m",
    )
with c_prompt:
    p1, p2 = st.columns([3, 1])
    preset_name = p1.selectbox(
        "Preset",
        list(presets),
        index=list(presets).index(state.get(f"label_{fid}", "hybrid_prior"))
        if state.get(f"label_{fid}") in presets
        else 3,
        format_func=lambda n: f"{n} — {wf.PRESET_DESCRIPTIONS[n]}",
    )
    if p2.button("Load preset", width="stretch"):
        load_preset(preset_name)
        st.rerun()
    st.radio(
        "Input to edit",
        ["prior", "photo"],
        key=input_key,
        horizontal=True,
        help="'prior' refines the physics render (best on this data); 'photo' edits the dusk photo.",
    )
    mode = st.radio("Prompt editor", ["slots", "raw"], key=f"mode_{fid}", horizontal=True)
    if mode == "slots":
        st.caption("`task` and `preserve` are locked so prompt edits cannot drop the geometry contract.")
        for slot in night.PROMPT_SLOTS:
            if slot_key[slot] not in state:
                state[slot_key[slot]] = presets["hybrid_prior"]["slots"][slot]
            st.text_area(slot, key=slot_key[slot], height=68, disabled=slot in night.LOCKED_SLOTS)
        slots: dict[str, str] | None = {s: state[slot_key[s]] for s in night.PROMPT_SLOTS}
        prompt = night.compose_prompt(slots or {})
    else:
        st.text_area("Prompt", key=raw_key, height=200)
        slots = None
        prompt = state[raw_key]
    with st.expander(f"Composed prompt ({len(prompt.split())} words)"):
        st.write(prompt)

st.divider()
g1, g2, g3, g4 = st.columns([1, 1, 1, 2])
label = g1.text_input("Label", key=f"label_{fid}")
n_seeds = g2.number_input("Seeds", 1, 6, 1)
seed_start = g3.number_input("First seed offset", 0, 999, 0)
use_vlm = g4.checkbox(
    "Score with VLM critic",
    value=vlm_key_available(),
    disabled=not vlm_key_available(),
    help=None if vlm_key_available() else "No judge API key in the environment.",
)

if st.button("Generate", type="primary", disabled=not prompt.strip()):
    run_id = runs_by_frame.get(fid)
    if run_id is None:
        run_id = engine.store.create_run(
            frame,
            kind="interactive",
            settings={"hardware": state["hardware"], "rig_overrides": state.get("rig_overrides") or {}},
            analysis=analysis.meta(),
        )
        runs_by_frame[fid] = run_id
    offsets = list(range(int(seed_start), int(seed_start) + int(n_seeds)))
    with st.status("Generating…", expanded=True) as status:
        if not engine.comparer_loaded:
            st.write("Loading FLUX.2 Klein (first time ~1 min)…")
        try:
            made = engine.generate(
                run_id,
                frame,
                analysis,
                label=label or "custom",
                input_kind=state[input_key],
                prompt=prompt,
                slots=slots,
                seed_offsets=offsets,
                use_vlm=use_vlm,
                progress=st.write,
            )
        except GpuBusyError as exc:
            status.update(label="GPU busy", state="error")
            st.error(str(exc))
        else:
            status.update(label=f"Generated {len(made)} candidate(s)", state="complete")

run_id = runs_by_frame.get(fid)
if run_id is None:
    st.stop()
pool = engine.store.candidates(run_id)
if not pool:
    st.stop()

st.subheader(f"Candidates in this session's run · `{run_id}`")
st.dataframe([metrics_row(c, pool) for c in sorted(pool, key=lambda c: -wf.rank_key(c, pool))], hide_index=True)
ranked = sorted(pool, key=lambda c: -wf.rank_key(c, pool))
for start in range(0, len(ranked), 3):
    cols = st.columns(3)
    for col, cand in zip(cols, ranked[start : start + 3]):
        with col.container(border=True):
            st.image(
                draw_overlay(
                    candidate_image(engine, run_id, cand),
                    boxes=list(frame.boxes),
                    box_scale=analysis.box_scale,
                    visible=cand.visible_flags(),
                    show_labels=False,
                ),
                width="stretch",
            )
            gate = "✅ passes gates" if cand.gate.get("passed") else "⚠️ " + "; ".join(cand.gate.get("reasons", []))
            st.markdown(
                f"**{cand.label}** · seed +{cand.seed_offset} · {cand.input} · rank "
                f"{wf.rank_key(cand, pool):.3f}  \n{gate}",
            )
            crit = cand.critique or {}
            if crit.get("critique"):
                st.caption(f"VLM {fmt(cand.vlm_overall, '.1f')}: {crit['critique']}  \nfix → {crit.get('prompt_fix')}")
            elif crit.get("error"):
                st.caption(f"VLM unscored: {crit['error'][:160]}")
            if cand.slots and st.button("Edit from this prompt", key=f"use_{cand.candidate_id}"):
                state[f"pending_slots_{fid}"] = cand.slots
                state[f"pending_input_{fid}"] = cand.input
                st.rerun()

st.divider()
st.subheader("Refine with the VLM")
slot_cands = [c for c in pool if c.slots]
if not vlm_key_available():
    st.caption("Needs a judge API key in the environment.")
elif not slot_cands:
    st.caption("Generate at least one slot-structured candidate first.")
else:
    best = max(slot_cands, key=lambda c: wf.rank_key(c, pool))
    st.caption(
        f"Rewrites the slots of the best slot candidate (**{best.label}**, seed +{best.seed_offset}) from all reviews.",
    )
    if st.button("Ask VLM for a slot rewrite"):
        with st.spinner("Rewriting…"):
            work_image = frame.load_image().resize(analysis.work_size)
            judge = engine.cfg.judge
            result = rewrite_prompt(
                best.slots or {},
                [wf.history_entry(c) for c in pool],
                work_image,
                candidate_image(engine, run_id, best),
                model=str(judge.model_id),
                api_base_url=judge.get("api_base_url"),
                provider=judge.get("api_provider"),
            )
        if result.error:
            st.error(f"Rewrite rejected: {result.error}")
        else:
            state[f"rewrite_{fid}"] = {
                "base": best.slots,
                "slots": result.slots,
                "changed": result.changed,
                "rationale": result.rationale,
            }
    proposal = state.get(f"rewrite_{fid}")
    if proposal:
        st.markdown(f"**Rationale:** {proposal['rationale']}")
        for key in proposal["changed"]:
            st.markdown(f"- **{key}**  \n  ~~{proposal['base'][key]}~~  \n  {proposal['slots'][key]}")
        if st.button("Load rewrite into editor"):
            state[f"pending_slots_{fid}"] = proposal["slots"]
            state.pop(f"rewrite_{fid}")
            st.rerun()
