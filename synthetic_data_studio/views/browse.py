"""Browse: filter source frames by annotation tags and labels, then select one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from studio.ui import frame_table, frame_thumb, select_frame, sidebar


PAGE_SIZE = 24
COLUMNS = 4

engine = sidebar()
st.title("Browse source frames")
rows = frame_table(st.session_state["dataset"], st.session_state["hardware"])
if not rows:
    st.warning("No frames found for this dataset — prepare its data first (see the README).")
    st.stop()

tag_groups = sorted({group for r in rows for group in r["tags"]})
labels = sorted({label for r in rows for label in r["labels"]})

with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Frame id contains", "")
    max_boxes = max((r["n_boxes"] for r in rows), default=0)
    min_boxes = c2.number_input("Min. labelled boxes", 0, max(max_boxes, 0), 0)
    label_filter = c3.multiselect("Has label", labels)
    tag_filters: dict[str, list[str]] = {}
    if tag_groups:
        cols = st.columns(min(len(tag_groups), 4))
        for i, group in enumerate(tag_groups):
            values = sorted({v for r in rows for v in r["tags"].get(group, [])})
            if values:
                tag_filters[group] = cols[i % len(cols)].multiselect(group.replace("_", " ").title(), values)
    st.caption("Tag filters match frames having **any** selected value within a group, and **all** groups.")


def _keep(row: dict[str, Any]) -> bool:
    if query and query.lower() not in row["frame_id"].lower():
        return False
    if row["n_boxes"] < min_boxes:
        return False
    if label_filter and not set(label_filter) & set(row["labels"]):
        return False
    return all(not chosen or set(chosen) & set(row["tags"].get(group, [])) for group, chosen in tag_filters.items())


filtered = [r for r in rows if _keep(r)]
st.session_state["browse_filtered"] = [r["frame_id"] for r in filtered]
n_pages = max(1, -(-len(filtered) // PAGE_SIZE))
top = st.columns([3, 1])
top[0].markdown(f"**{len(filtered)}** of {len(rows)} frames match")
page = top[1].number_input("Page", 1, n_pages, 1, help=f"{n_pages} pages")

with st.expander("Table view"):
    st.dataframe(
        [
            {
                "frame": r["frame_id"],
                "boxes": r["n_boxes"],
                "labels": ", ".join(f"{k}×{v}" for k, v in r["labels"].items()),
                **{g: ", ".join(r["tags"].get(g, [])) for g in tag_groups},
            }
            for r in filtered
        ],
        hide_index=True,
        width="stretch",
    )

selected = st.session_state.get("frame_id")
chunk = filtered[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]
for start in range(0, len(chunk), COLUMNS):
    cols = st.columns(COLUMNS)
    for col, row in zip(cols, chunk[start : start + COLUMNS]):
        with col.container(border=True):
            st.image(frame_thumb(Path(row["image_path"])), width="stretch")
            badge = " ✅" if row["frame_id"] == selected else ""
            st.markdown(f"**{row['frame_id']}**{badge}")
            label_txt = ", ".join(f"{k}×{v}" for k, v in row["labels"].items()) or "no boxes"
            env = ", ".join(row["tags"].get("environment", []))
            st.caption(f"{label_txt}" + (f" · {env}" if env else ""))
            b1, b2 = st.columns(2)
            if b1.button("Inspect", key=f"inspect_{row['frame_id']}", width="stretch"):
                select_frame(row["frame_id"])
                st.switch_page("views/inspect.py")
            if b2.button("Generate", key=f"gen_{row['frame_id']}", width="stretch"):
                select_frame(row["frame_id"])
                st.switch_page("views/generate.py")
