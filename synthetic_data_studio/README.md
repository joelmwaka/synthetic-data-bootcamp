# Synthetic Data Studio

A Streamlit app for synthetic image data generation on top of the
[edge-case image generation](../implementations/edge_case_image_generation) pipeline.
Pick a real source frame with its annotations, generate synthetic variants, score
them against a physics reference and an optional VLM critic, review them, and
export accepted images with labels and provenance.

The first supported synthesis task is **night travel under train headlights** on the
Labels4Rail `l4r_nlb` frames (see
[Notebook 4](../implementations/edge_case_image_generation/notebooks/04_dark_condition_simulation.ipynb)
for the method). Browsing and inspection work for every dataset package; night
synthesis needs camera calibration, which only `l4r_nlb` provides.

## Run it

From the repository root, install the image stack (Streamlit is part of that group):

```bash
uv sync --dev --group edge-case-image-generation
```

Then start the app **from this folder** so `.streamlit/config.toml` is applied:

```bash
cd synthetic_data_studio
uv run streamlit run app.py
```

The server binds to `localhost:8501` only (this machine is shared). Open it through
VS Code port forwarding. A GPU is needed for generation (an L4 works: Klein loads in
about a minute, each edit then takes about 20 s).

**Data:** Labels4Rail frames are read directly from
`implementations/edge_case_image_generation/data/l4r_nlb/raw/L4R_NLB_fall/`
(`images/`, `annotations/`, `masks/`, `camera/camera.yaml`). Other datasets are
read from `data/<dataset>/samples/` plus an optional `labels.json`.

**VLM critic:** set `OPENAI_API_KEY` (Vector proxy) in the environment or in
`implementations/edge_case_image_generation/.env`. Without a key the app runs on
physics metrics only; keys are never shown in the UI.

## Pages

| Page | What it does |
|---|---|
| **Browse** | Filter frames by scene tags (environment, weather, light, time of day, layout), labels and box count; thumbnail grid; select a frame. |
| **Inspect** | Overlays for switch boxes, track polylines, track mask, and the rail-plane horizon; tag and label tables; calibration check (ego-rail gauge); scene analysis (depth, sky/ground/objects, metric distance); headlight irradiance and the physics-prior render; adjustable headlight rig. |
| **Generate** | Load a prompt preset, edit the slot-structured prompt (`task` and `preserve` locked), choose the input (physics prior or photo) and seeds, generate, and compare candidates by rank, metrics, gate result and VLM critique. Ask the VLM to rewrite the slots from the run's review history and load the rewrite back into the editor. |
| **Review** | Side-by-side source, physics map and candidate; per-box "visible at night" flags; accept or reject with a note. |
| **Batch** | Queue a preset or a reviewed candidate's prompt over many frames (from the Browse filter, frames with boxes, or a manual pick), with seeds, accept gates, optional VLM scoring and optional auto-accept. Start the worker and follow progress; cancel jobs. |
| **Runs & export** | All runs with decision counts; open a run for review; export accepted candidates to a new folder with `images/`, `labels.json` and `provenance.json`, and download it as a zip. |

## How it is built

```text
synthetic_data_studio/
  app.py              Streamlit entry point (st.navigation)
  worker.py           batch worker process
  views/              one file per page
  studio/
    ui.py             Streamlit caching, sidebar, selection state
    engine.py         config, frame source, run store, lazy models, GPU lock
    batch.py          job specs and the per-job loop
    jobs.py           SQLite job queue with heartbeats and cancellation
    overlays.py       annotation overlays and false-colour maps
  tests/              unit tests and Streamlit AppTest smoke tests
```

The app is a thin UI. Synthesis logic lives in the shared library, also used by
Notebook 4:

- `aieng.syn_data.image.data.frames` — dataset adapters returning a `Frame`
  (image, boxes, tags, track annotation, mask, calibration, capability flags).
- `aieng.syn_data.image.workflow.night` — scene analysis with an on-disk geometry
  cache, prompt presets, generation, scoring, accept gates and ranking.
- `aieng.syn_data.image.workflow.store` — run records and export.

### Storage

Everything is written under
`implementations/edge_case_image_generation/outputs/<dataset>/studio/` (ignored by
git):

```text
runs/<run_id>/run.json             source frame, settings, analysis summary, candidates + reviews
runs/<run_id>/candidates/<id>.png  generated images (768×432 analysis grid)
cache/<frame_id>/<key>.npz         per-frame geometry (depth + segmentation results)
exports/<name>/                    images/, labels.json, provenance.json
jobs.sqlite                        batch queue
worker.log                         output of workers started from the app
```

`run.json` is updated under a file lock with an atomic replace, so the app, the
notebook and a worker can write to the same store safely. Export labels are scaled
from source-image to generated-image coordinates; boxes judged dark are dropped
unless you choose to keep them (flagged `visible_at_night: false`).

### GPU and memory

Model work (depth and segmentation on a geometry-cache miss, and Klein edits) runs
under a cross-process lock (`gpu.lock`), so the app and a worker never use the GPU at
the same time. If the worker holds it, interactive generation reports "GPU busy".
Each process that generates keeps its own copy of Klein in memory. Starting a worker
from the Batch page unloads the app's copy first. Use **Unload generator** in the
sidebar to free memory manually. The worker unloads after every job.

### Batch worker

Start it from the Batch page, or manually:

```bash
cd synthetic_data_studio
uv run python worker.py --dataset l4r_nlb --hardware gpu_l4          # keeps polling
uv run python worker.py --dataset l4r_nlb --hardware gpu_l4 --once   # drain the queue, then exit
```

One frame failing doesn't stop a job; its error is recorded in the job progress.
A worker that stops sending heartbeats has its running job marked failed by the
next worker that starts.

## Tests

```bash
cd synthetic_data_studio
uv run pytest tests -q
```

`tests/test_studio.py` covers the queue, batch logic and overlays without a GPU.
`tests/test_app.py` renders every page with Streamlit's `AppTest` against the local
Labels4Rail data and is skipped when that data isn't present. Workflow and adapter
tests live in `aieng-synthetic-data/tests/test_image_workflow.py`.

## Limitations

- One synthesis task (night headlights), and only for calibrated Labels4Rail frames.
- Boxes can be marked visible or dark, but not redrawn.
- Single-user design: one GPU lock and one shared output store per dataset.
- The physics prior has no occlusion shadows, and the scoring weights and gates are
  workshop defaults, not calibrated quality thresholds.
