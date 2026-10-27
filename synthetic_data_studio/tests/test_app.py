"""Streamlit AppTest smoke tests on local Labels4Rail data (skipped without it).

They render every page headlessly and assert no exceptions. No page here
triggers model loading: Inspect/Generate only read cached geometry.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"
IMPL = Path(__file__).resolve().parents[2] / "implementations" / "edge_case_image_generation"
RAW = IMPL / "data" / "l4r_nlb" / "raw" / "L4R_NLB_fall"
FRAME = "nlb_fall_000419"

pytestmark = pytest.mark.skipif(
    not (RAW / "images" / f"{FRAME}.png").is_file(),
    reason="local Labels4Rail data not available",
)


def _app(frame: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["dataset"] = "l4r_nlb"
    at.session_state["hardware"] = "gpu_l4"
    if frame:
        at.session_state["frame_id"] = frame
    return at


def _no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_browse_lists_and_filters_frames():
    at = _app().run()
    _no_errors(at)
    assert any("frames match" in m.value for m in at.markdown)
    env = next(ms for ms in at.multiselect if ms.label == "Environment")
    env.set_value(["station"]).run()
    _no_errors(at)
    summary = next(m.value for m in at.markdown if "frames match" in m.value)
    matched, total = int(summary.split("**")[1]), int(summary.split("of ")[1].split()[0])
    assert 0 < matched < total


def test_inspect_page_renders_annotations_and_calibration():
    at = _app(FRAME)
    at.switch_page("views/inspect.py").run()
    _no_errors(at)
    assert any("Ego-rail gauge" in s.value for s in at.success)


@pytest.mark.parametrize("page", ["views/generate.py", "views/review.py", "views/batch.py", "views/runs.py"])
def test_synthesis_pages_render(page):
    at = _app(FRAME)
    at.switch_page(page).run()
    _no_errors(at)
    if page == "views/generate.py":
        # Cached geometry → prompt editor with all nine slots, two of them locked.
        assert len(at.text_area) == 9
        assert sum(t.disabled for t in at.text_area) == 2
    if page == "views/review.py" and any(Path(IMPL / "outputs" / "l4r_nlb" / "studio" / "runs").glob("*/run.json")):
        assert any(s.label == "Run" for s in at.selectbox)


def test_pages_without_selection_prompt_for_a_frame():
    at = _app()
    at.switch_page("views/generate.py").run()
    _no_errors(at)
    assert any("Select a frame" in i.value for i in at.info)
