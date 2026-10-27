"""Runs & export: browse runs and export accepted data with provenance."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import streamlit as st
from aieng.syn_data.image.workflow import night as wf
from studio.ui import frame_thumb, sidebar


engine = sidebar()
state = st.session_state
st.title("Runs & export")
runs = engine.store.list_runs()
if not runs:
    st.info("No runs yet.")
    st.stop()

table = []
for run in runs:
    cands = [wf.Candidate.from_dict(c) for c in run.get("candidates", [])]
    decisions = [c.decision for c in cands]
    table.append(
        {
            "run": run["run_id"],
            "frame": run["frame"]["frame_id"],
            "kind": run.get("kind", ""),
            "candidates": len(cands),
            "accepted": decisions.count("accepted"),
            "rejected": decisions.count("rejected"),
            "pending": decisions.count("pending"),
            "passed gates": sum(bool(c.gate.get("passed")) for c in cands),
            "best physics": round(max((c.physics_score for c in cands), default=0.0), 3),
            "created": run.get("created_at", ""),
        },
    )
kinds = sorted({r["kind"] for r in table})
chosen_kinds = st.multiselect("Kind", kinds, default=kinds)
hide_empty = st.checkbox("Hide runs without candidates", value=True)
shown = [r for r in table if r["kind"] in chosen_kinds and (r["candidates"] or not hide_empty)]
st.dataframe(shown, hide_index=True, width="stretch")
totals = {k: sum(r[k] for r in table) for k in ("candidates", "accepted", "rejected", "pending")}
st.caption(" · ".join(f"{k}: {v}" for k, v in totals.items()))

if shown:
    pick = st.selectbox("Open run", [r["run"] for r in shown])
    run = engine.store.load_run(pick)
    cands = [wf.Candidate.from_dict(c) for c in run["candidates"]]
    cols = st.columns(4)
    for i, cand in enumerate(cands):
        with cols[i % 4]:
            st.image(frame_thumb(engine.store.image_path(pick, cand), 320), width="stretch")
            st.caption(f"{cand.decision} · {cand.label} · seed +{cand.seed_offset} · physics {cand.physics_score:.3f}")
    if cands and st.button("Review this run"):
        state["review_run"] = pick
        st.switch_page("views/review.py")

st.divider()
st.subheader("Export accepted candidates")
n_accepted = totals["accepted"]
st.caption(
    f"{n_accepted} accepted candidate(s). Exports copy images, write `labels.json` (boxes scaled to the image) "
    "and `provenance.json` (source frame, settings, prompt, metrics, review) into a new folder.",
)
c1, c2 = st.columns(2)
include_dark = c1.checkbox("Keep boxes judged dark at night (flagged visible_at_night=false)", value=False)
name = c2.text_input("Export name (optional)", placeholder="export_<timestamp>")
if st.button("Export", type="primary", disabled=n_accepted == 0):
    try:
        dest = engine.store.export_accepted(include_dark_boxes=include_dark, name=name.strip() or None)
    except (ValueError, FileExistsError) as exc:
        st.error(str(exc))
    else:
        st.success(f"Exported to `{dest}`")

exports = sorted((p for p in engine.store.exports_dir.glob("*") if p.is_dir()), reverse=True)
if exports:
    st.markdown("**Existing exports**")
    for exp in exports:
        labels_path = exp / "labels.json"
        n_images = len(json.loads(labels_path.read_text())) if labels_path.is_file() else 0
        a, b = st.columns([3, 1])
        a.markdown(f"`{exp.name}` · {n_images} images")
        zip_key = f"zip_{exp.name}"
        if zip_key in state:
            b.download_button(
                "Download zip",
                state[zip_key],
                file_name=f"{exp.name}.zip",
                mime="application/zip",
                key=f"dl_{exp.name}",
            )
        elif b.button("Prepare zip", key=f"prep_{exp.name}"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(exp.rglob("*")):
                    if path.is_file():
                        zf.write(path, Path(exp.name) / path.relative_to(exp))
            state[zip_key] = buf.getvalue()
            st.rerun()
