Overvak Action Detection Take‑Home
=================================

This repository contains a **minimal end‑to‑end action‑detection pipeline** for kids’ activities in unconstrained YouTube videos, designed for the Overvak founding AI engineer take‑home.

### What this project does
- **Ingests a YouTube URL** and downloads the video.
- **Samples frames** from the video and feeds them to a **pre‑trained 3D CNN** (`r3d_18` from `torchvision`, trained on Kinetics‑400).
- **Maps generic Kinetics actions → child‑activity labels** (e.g. *climbing*, *running*, *laughing*).
- **Outputs detected actions + confidences**.
- If ground‑truth labels are available for that video (from a CSV like the example in the prompt), it **computes Precision and Recall**.
- Exposes everything through a **simple Streamlit UI** where you paste a YouTube URL and click **“Process”**.

### Quickstart

1. **Create and activate a virtualenv (recommended)**

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **(Optional) Add your label CSV**

Put a CSV file at `data/labels.csv` with at least:

- `video_id` – YouTube video ID
- One column per action label, e.g. `climbing,crying,falling,hitting,jumping,laughing,pushing,running` (values 0/1).

An example header row is included in `data/labels_example.csv`.

4. **Run the UI**

```bash
streamlit run ui.py
```

Then open the printed local URL in your browser, paste a YouTube link, and click **Process**.

### High‑level approach (very short)

- **Model**: `torchvision.models.video.r3d_18` with Kinetics‑400 pretrained weights for robust generic action recognition without custom training.
- **Inference**: uniformly sample a small number of frames (e.g. 16) from the video, apply the official Kinetics transforms, average logits over clips, and map top‑K Kinetics classes to child‑centric labels via simple keyword rules.
- **Metrics**: treat each target action as present/absent per video and compute **micro Precision and Recall** using the ground‑truth CSV when available.

For a fuller write‑up (problem framing, risks, and assumptions), see the comments and docstrings in `pipeline.py`, `metrics.py`, and the Streamlit app in `ui.py`.

