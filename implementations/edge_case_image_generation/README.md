# Edge-Case Image Generation

Synthetic long-tail / edge-case scene synthesis for perception models.

Part of the [Vector Institute synthetic-data-bootcamp](https://github.com/VectorInstitute/synthetic-data-bootcamp).

## Overview

Rare road conditions (a traffic cone in the lane, a trash bin at the curb) are exactly where perception models fail, and exactly what real datasets have few labeled examples of. This reference implementation edits **real** street photos to add those rare objects, keeps only the edits that pass a quality filter, and then tests whether they help a detector.

```text
 real scene → EDIT (insert rare object) → ANNOTATE (boxes) → JUDGE (multi-signal filter)
                                                              ├─ accept → training set
                                                              ├─ retry  → re-edit with next variation
                                                              └─ reject
 real-only vs real + accepted synth → train detector → evaluate on the SAME held-out real test
```

- **Why start from real scenes?** Real photos already have the right camera, lighting, and clutter, so the edit only has to add the rare object. Generating whole scenes from text is more likely to drift from the real domain.
- **What makes a synthetic sample good?**
  - **Fidelity:** stays faithful to the real data distribution and the intended edit.
  - **Novelty:** adds useful variation rather than repeating existing data.
  - **Usability:** usable for the downstream task; for detection that means a correct, well-placed box.
- **Multi-signal quality filtering with a VLM in the loop.** The VLM judge is one signal, not the whole filter. An edit is accepted only if it also passes:
  - real reference-image fidelity checks,
  - a CLIP embedding "safe zone" (close enough to real, not a near-duplicate),
  - annotation boxes,
  - placement gates.
- **The final test is on real data.** Notebook 3 trains on real-only, real + 50%, and real + 100% of the accepted synthetic images, then evaluates all three on the same held-out **real** test set. A pretty edit that does not move that number is a demo, not data.

## Configuration

Configuration is **dataset-first** under `configs/datasets/<name>/`, with shared defaults in `configs/default/` and hardware profiles in `configs/hardware/`.

| Want to change… | Where |
|-----------------|-------|
| Workshop size (download + NB2/NB3 counts) | `configs/config.yaml` → `scale` (`1.0` full run; `0.2` bootcamp small mode) |
| Dataset / your own local images | Copy `configs/datasets/_template/`; `data.yaml` → `kind: local`, images in `data/<id>/samples/` ([Notebook 0](notebooks/00_flight_precheck.ipynb) §1) |
| Anomalies (prompts, variations, masks, gates) | `configs/datasets/<id>/generation/anomalies/*.yaml` — [docs/anomaly_authoring.md](docs/anomaly_authoring.md) |
| Editor / depth / seg / detector models | `configs/default/*.yaml`, overridden per GPU in `configs/hardware/*.yaml` |
| Judge model, API provider, thresholds | `configs/default/judge.yaml` |
| Batch sizes, detector fine-tune | Knobs at the top of Notebooks 2 and 3 |

## Default dataset: Mapillary Vistas v2 (toy subset)

Open **street-level** scenes ([Mapillary Vistas](https://www.mapillary.com/dataset/vistas), CC BY-NC-SA).

Workshop anomalies (see `configs/datasets/mapillary_vistas/dataset.yaml`):

- **road_debris** (cardboard box on the lane)
- **traffic_cone**
- **trash_bin**
- **fog** (global; shows why local inpaint alone is not enough)

Notebooks 2–3 use **traffic_cone** and **trash_bin**.

We do **not** download the full ~29 GB HF zip. Toy images land in `data/mapillary_vistas/samples/`.

**Preferred:** run [Notebook 0](notebooks/00_flight_precheck.ipynb) — it checks the cache and downloads the toy subset with a progress bar (plus interactive HF login if needed). You do **not** need a separate CLI extract for the workshop path.

Optional CLI (same extract as Notebook 0):

```bash
# requires: accept Mapillary terms on HF once
# (https://huggingface.co/datasets/candylion/mapillary-vistas-v2), then huggingface-cli login
uv run python scripts/extract_mapillary_toy.py
```

Other dataset packages: `rdd2022`, `nordland_hf`, `l4r_nlb` (Labels4Rail railway frames, see below). Copy `configs/datasets/_template/` to add your own.

### Labels4Rail switch sample preparation

With the fall images, annotations, masks, and camera calibration under
`data/l4r_nlb/raw/L4R_NLB_fall/`, run:

```bash
uv run python implementations/edge_case_image_generation/scripts/prepare_l4r_samples.py
```

Run from the repository root. The script selects 16 switch-bearing frames using
box prominence, switch count, and a minimum frame-ID gap of 20. It creates
relative image symlinks under `data/l4r_nlb/samples/`, filename-keyed `labels.json`
boxes in original-image pixel coordinates, and `source_meta.json` with source
paths, scene tags, switch attributes, and selection details. A boxed contact sheet
is saved as `data/l4r_nlb/switch_contact_sheet.jpg`. Raw files stay unchanged, and
the script refuses to overwrite an existing samples directory.

Boxes are normalized from the two annotation marks; exact duplicates and entries
without two marks are excluded and counted in the metadata. These are positive
switch examples named `scene_*` for use as editing seeds, **not** verified empty
backgrounds or a train/test split. Frame-ID spacing does not guarantee independent
scenes.

### Labels4Rail night simulation (Notebook 4)

`configs/datasets/l4r_nlb/` defines one global anomaly, `night_headlights`: night
travel lit only by the train's headlights, which sit **below** the cab camera.
[Notebook 4](notebooks/04_dark_condition_simulation.ipynb) develops its prompt on
one frame:

1. **Geometry** — the L4R `camera.yaml` gives a calibrated rail plane (checked by
   measuring the annotated ego-rail gauge at several distances); Depth Anything is
   aligned to it on track-mask pixels to get metric depth, and SegFormer masks the sky.
2. **Physics prior** — a headlight irradiance map (beam lobes, inverse-square falloff,
   Lambert cosine; no occlusion shadows) and a physically based night render.
3. **Prompt development** — slot-structured prompts grounded in annotation tags and
   the computed beam reach; an ablation (naive / named headlights / effect-only /
   refine-the-prior), physics metrics (`beam_corr`, `phantom_light`, `lit_structure`,
   `clipped_lit`, darkness) plus a VLM critic, VLM slot rewrites with `task` and
   `preserve` locked, and a seed-robustness check.
4. **Labels** — switch boxes carry over and are flagged `visible_at_night` by beam
   irradiance and preserved local structure.

Outputs go to `outputs/l4r_nlb/night/<frame>/` (`prompt_development.json`, every
candidate, the selected frame, and `labels_night.json`). Library code:
`aieng.syn_data.image.generate.night` and `aieng.syn_data.image.judge.night_critic`;
the shared workflow (scene analysis with a geometry cache, scoring, gates, run
records) is `aieng.syn_data.image.workflow`. Every candidate the notebook scores is
also saved to `outputs/l4r_nlb/studio/`, where the
[Synthetic Data Studio](../../synthetic_data_studio/README.md) app can review, batch,
and export it. Notebooks 1–3 are not yet configured for railway classes.

## Hardware + dataset selection

In `configs/config.yaml` (or notebook overrides):

```python
load_config(overrides=["dataset_name=mapillary_vistas", "hardware=cpu"])
load_config(overrides=["dataset_name=mapillary_vistas", "hardware=gpu_l4"])
```

**L4:** FLUX.2-klein-4B for inpaint + instruct (default generator); SD 1.5 ControlNet depth+seg.
**Judge:** API vision chat by default (`configs/default/judge.yaml`); optional local Qwen for offline.

## Setup

Dependencies live in the **monorepo root** `pyproject.toml` under the `edge-case-image-generation` group. Library code ships with the workspace package **`aieng-synthetic-data`** as `aieng.syn_data.image` (same pattern as `aieng.syn_data.text` / `.synbench`).

```bash
# From the synthetic-data-bootcamp repository root
uv sync --dev --group edge-case-image-generation

# Optional local overrides (Vector key is usually already in the workshop env):
cp implementations/edge_case_image_generation/.env.example \
   implementations/edge_case_image_generation/.env
# edit only if you want your own API keys / base URLs, or HF_TOKEN=…
```

Or run `implementations/edge_case_image_generation/scripts/setup_notebook_env.sh` to create `.venv` at the repo root and register the Jupyter kernel.

**Workshop images** (Mapillary toy subset — not the full ~29 GB zip):

- Prefer [Notebook 0](notebooks/00_flight_precheck.ipynb) (progress bar + interactive HF login), or
- CLI: `uv run python implementations/edge_case_image_generation/scripts/extract_mapillary_toy.py`

Notebooks load `.env` automatically on startup when present — no `export` in the terminal needed. If the workshop machine already exports `OPENAI_API_KEY` / proxy URL, you can skip creating `.env`.

Imports use the installed package (no `sys.path` hacks):

```python
from aieng.syn_data.image.config import load_config
from aieng.syn_data.image.bootstrap import bootstrap_project_root
```

Package layout under `aieng-synthetic-data/aieng/syn_data/image/`:

```text
config/   data/   generate/   judge/   batch/   train/   viz/
preflight.py   bootstrap.py
```

Flat import paths (`aieng.syn_data.image.pipeline`, `.batch_runner`, …) still work via thin re-exports.

This folder keeps **configs**, **data**, **notebooks**, and **scripts** only.

## Notebooks

- `notebooks/00_flight_precheck.ipynb` — env / API / GPU / **data download** / model cache check
- `notebooks/00.5_method_comparison.ipynb` — educational bake-off of edit methods; pick `instruct` for NB1
- `notebooks/01_sample_data_generation.ipynb` — synthesis loop + fidelity / novelty / usability (multi-signal judge)
- `notebooks/02_batch_dataset_generation.ipynb` — stratified split, batch synth + judge until a target number of **accepted** images per class (with an attempt budget), export for NB3
- `notebooks/03_training_and_evaluation.ipynb` — YOLOv8n real-only vs real+synth on held-out real test
- `notebooks/04_dark_condition_simulation.ipynb` — Labels4Rail night simulation: calibrated headlight physics prior + prompt development with physics metrics and a VLM critic

Supporting docs: `docs/citations.md`, `docs/figures.md`, `docs/anomaly_authoring.md`.
