# Deploying to HuggingFace Spaces

## Local run

Install Streamlit into the conda environment (one-time):

```bash
conda activate FB_twin
pip install streamlit
```

Then launch:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## HuggingFace Spaces deployment

HF Spaces needs its own repository with a `README.md` containing a YAML frontmatter block.
The simplest workflow is to keep the project repo separate from the Space repo.

### Step 1 — Create the Space

Go to https://huggingface.co/spaces/new and fill in:

| Field | Value |
|-------|-------|
| Space name | `fluid-bed-coating-twin` (or any name you like) |
| SDK | **Streamlit** |
| Visibility | Public or Private |

### Step 2 — Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<your-hf-username>/fluid-bed-coating-twin
cd fluid-bed-coating-twin
```

### Step 3 — Copy the required files

Copy these files/directories from this project into the cloned Space repo:

```
app.py
requirements.txt
src/
```

The `src/fluid_bed/` package is imported via `sys.path` in `app.py`, so no `pip install` of the
local package is needed — HF Spaces will find it automatically.

### Step 4 — Add HF Spaces metadata to README.md

The Space repo's `README.md` must start with this YAML block:

```yaml
---
title: Fluid Bed Coating Digital Twin
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
short_description: Physics-based ODE digital twin for fluid bed particle coating
---
```

Replace the version with the latest available Streamlit on HF Spaces if needed.

### Step 5 — Push

```bash
git add .
git commit -m "Add Streamlit digital twin app"
git push
```

HF Spaces will automatically build and deploy. Build logs are visible on the Space page.

---

## What runs on the Space

- **Sliders** in the sidebar control 16 process parameters.
- **Simulation** is run on every slider change and cached — moving the Sample time slider
  re-plots the dissolution curve instantly without re-running the ODE.
- **No data files** are needed: all physics is embedded in the `fluid_bed` package.

## Updating the Space

Push new commits to the HF Space repo (same `git push`). HF rebuilds automatically.
To sync updates from this dev branch:

```bash
cp app.py requirements.txt /path/to/space-repo/
cp -r src/ /path/to/space-repo/
cd /path/to/space-repo && git add . && git commit -m "Update" && git push
```
