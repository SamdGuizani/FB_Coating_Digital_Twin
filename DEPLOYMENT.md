# Deploying to HuggingFace Spaces

The Space is deployed with the **Docker SDK** and runs the Plotly rendering
variant (`app_plotly.py`). The Matplotlib variant (`app.py`) is kept in the
repo as a fallback.

## Local run

Install Streamlit into the conda environment (one-time):

```bash
conda activate FB_twin
pip install streamlit plotly
```

Then launch the Plotly app (the one deployed to the Space):

```bash
streamlit run app_plotly.py
```

The Matplotlib variant still works the same way:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## HuggingFace Spaces deployment

HF Spaces needs its own repository with a `README.md` containing a YAML
frontmatter block. The Space repo is kept **separate** from this project repo
(it lives in `../Fluid Bed Coating HF App/fluid-bed-coating-twin`). Files are
copied from this project into the Space repo, then pushed.

### Step 1 — Create the Space

Go to https://huggingface.co/spaces/new and fill in:

| Field | Value |
|-------|-------|
| Space name | `fluid-bed-coating-twin` (or any name you like) |
| SDK | **Docker** |
| Visibility | Public or Private |

### Step 2 — Clone the Space repo

```bash
git clone https://huggingface.co/spaces/<your-hf-username>/fluid-bed-coating-twin
cd fluid-bed-coating-twin
```

### Step 3 — Copy the required files

Copy these files/directories from this project into the cloned Space repo:

```
app_plotly.py        # entrypoint (Plotly rendering)
plotly_figures.py    # Plotly figure builders imported by app_plotly.py
app.py               # Matplotlib fallback (optional)
requirements.txt
src/
```

The `src/fluid_bed/` package is imported via `sys.path` in `app_plotly.py`, so
no `pip install` of the local package is needed — the import resolves at runtime.
`plotly_figures.py` must sit next to `app_plotly.py` (it is imported as a
top-level module).

### Step 4 — Dockerfile

The Space builds from a `Dockerfile` (committed in the Space repo). Its
entrypoint must launch the Plotly app:

```dockerfile
ENTRYPOINT ["streamlit", "run", "app_plotly.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

The Dockerfile does `COPY . .`, so both root-level files (`app_plotly.py` and
`plotly_figures.py`) land in the image automatically.

### Step 5 — requirements.txt

Must include `plotly` alongside the existing dependencies:

```
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
streamlit>=1.37
plotly>=5.20
```

### Step 6 — Add HF Spaces metadata to README.md

The Space repo's `README.md` must start with this YAML block (note `sdk: docker`
and `app_port`, not `sdk: streamlit`):

```yaml
---
title: Fluid Bed Coating Twin
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Physics-based ODE digital twin for fluid bed coating
license: gpl-2.0
---
```

### Step 7 — Push

```bash
git add .
git commit -m "Deploy Plotly digital twin app"
git push
```

HF Spaces will automatically build the Docker image and deploy. Build logs are
visible on the Space page.

---

## What runs on the Space

- **Sliders** in the sidebar control 16 process parameters.
- **Charts** are interactive Plotly figures (`plotly_figures.py`).
- **Simulation** is run on every slider change and cached — moving the Sample
  time slider re-plots the dissolution curve instantly without re-running the ODE.
- **No data files** are needed: all physics is embedded in the `fluid_bed` package.

## Updating the Space

Push new commits to the HF Space repo (same `git push`). HF rebuilds
automatically. To sync updates from this dev branch:

```bash
cp app_plotly.py plotly_figures.py app.py requirements.txt /path/to/space-repo/
cp -r src/ /path/to/space-repo/
cd /path/to/space-repo && git add . && git commit -m "Update" && git push
```
