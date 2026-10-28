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
VS Code port forwarding. A GPU is needed for the local generators (see
[Generators](#generators)).

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
| **Generate** | Pick a generator and its parameters, load a prompt preset (or the generator's recommended one), edit the slot-structured prompt (`task` and `preserve` locked) or a plain prompt plus negative prompt, choose the input (physics prior or photo) and seeds, generate, and compare candidates by rank, metrics, gate result and VLM critique. "Reuse these settings" restores a candidate's generator, parameters and prompt. Ask the VLM to rewrite the slots from the run's review history. |
| **Review** | Side-by-side source, physics map and candidate; per-box "visible at night" flags; accept or reject with a note. |
| **Batch** | Queue a preset or a reviewed candidate's prompt (and generator) over many frames (from the Browse filter, frames with boxes, or a manual pick), with seeds, accept gates, optional VLM scoring and optional auto-accept. Start the worker and follow progress; cancel jobs. |
| **Runs & export** | All runs with decision counts; open a run for review; export accepted candidates to a new folder with `images/`, `labels.json` and `provenance.json`, and download it as a zip. |

## Generators

Pick the image model on the Generate and Batch pages. Each generator is defined in
`aieng.syn_data.image.workflow.generators` (method, model ids, tunable parameters,
prompt style, requirements); the choice and its parameters are stored with every
candidate and in export provenance.

| Generator | Models | Prompt style | Notes |
|---|---|---|---|
| FLUX.2 Klein 4B — instruct edit (default) | `FLUX.2-klein-4B` | long instruction | ~15 GB weights; ~1 min load, then ~20–40 s per edit on an L4. |
| FLUX.2 Klein 4B — full-frame img2img | same weights (inpaint pipeline, full mask) | long instruction | `strength` sets how much of the input is re-noised. |
| SD 1.5 + ControlNet (depth + seg) | SD 1.5, `sd-controlnet-depth`, `sd-controlnet-seg` | caption + negative (77 CLIP tokens) | Geometry held by the scene's depth map and ADE20K segmentation. |
| SD 1.5 inpainting — full-frame | `stable-diffusion-inpainting` | caption + negative (77 CLIP tokens) | No structural control; can move or invent content. |
| InstructPix2Pix | `instruct-pix2pix` | short command (77 CLIP tokens) | Runs at 512 px. |
| Qwen-Image-Edit (~20B) | `Qwen-Image-Edit` | long instruction | Needs ≥64 GB of GPU memory — blocked on a single L4. |
| Nano Banana 2 (cloud) | `gemini-3.1-flash-image` via the Vector proxy | long instruction | Needs `OPENAI_API_KEY`; the frame is sent to the API; not seed-reproducible. |

The picker shows 🟢 ready (weights cached), 🟡 downloads weights on first use, or
⛔ cannot run here (with the reason). Switching between local generators unloads
the previous one; generators sharing weights (the two Klein options) reuse them.
For SD 1.5 generators the page counts CLIP tokens and warns when the prompt is cut
off; use the `caption` or `short_instruction` presets.

Measured on frame `nlb_fall_000419` (L4, recommended preset, seed 0; physics score
from `selection_score`, higher is better; gates as in `DEFAULT_GATES`):

| Generator | Input | Physics score | Gates |
|---|---|---|---|
| Klein 4B full-frame img2img (strength 0.75) | physics prior | 0.81 | pass |
| Klein 4B instruct | physics prior | 0.72 | pass |
| SD 1.5 inpainting (strength 0.5) | physics prior | 0.71 | pass, but changes the track layout |
| SD 1.5 + ControlNet (strength 0.5) | physics prior | 0.70 | pass |
| InstructPix2Pix | physics prior | 0.66 | pass, close to the prior |
| SD 1.5 + ControlNet (strength 0.75) | dusk photo | 0.11 | fail — stays dusk-bright |
| SD 1.5 inpainting (strength 0.7) | dusk photo | 0.02 | fail — invents a lit train |

SD 1.5 cannot take a dusk photo to real night darkness, so its presets edit the
physics render instead. One frame and one seed are not a benchmark; compare
generators on your own frames before choosing one for a batch.

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
Each process that generates keeps its own copy of the active generator in memory.
Starting a worker from the Batch page unloads the app's copy first. Cloud generation
doesn't take the GPU lock or unload a local generator. Use **Unload generator** in the
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
- Qwen-Image-Edit cannot run on a single L4; Nano Banana 2 needs an API key and is not
  covered by GPU tests here.
- Boxes can be marked visible or dark, but not redrawn.
- Single-user design: one GPU lock and one shared output store per dataset.
- The physics prior has no occlusion shadows, and the scoring weights and gates are
  workshop defaults, not calibrated quality thresholds.
