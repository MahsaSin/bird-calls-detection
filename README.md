# Bird Calls Detection Project

A machine learning project that detects bird calls from audio.  
Built with **Python 3.12**, structured for readability and easy deployment to an **API**, and **Dockerized**.

Simple wrapper around a scikit-learn classifier with `librosa` feature extraction.  
Returns multi-label predictions plus a joined `labels` field.

---

## Project Structure

notebooks/         → Exploration & experiments  
src/inference/     → Core logic (Predictor class, FastAPI app)  
models/            → Trained model files (e.g., `svm_ovr.joblib`)  
artifacts/         → Bundle (`metadata.json`) describing features/targets/model path  
pyproject.toml     → Metadata & dependencies  
uv.lock            → Locked deps for reproducibility  
Dockerfile         → Container build (Python 3.12 slim, ffmpeg, uv)  
README.md          → You’re here

> The Dockerfile copies `src/inference` into `/app/inference`, and sets `PYTHONPATH=.` so `from inference ...` works.

---

## Overview

This project predicts bird species from audio files. It:

- Loads audio with **librosa**
- Extracts features (spectral flux, RMSE, bandwidth, flatness, spectral contrast, poly, tonnetz, MFCCs, ZCR, tempo)
- Optionally maps `Location` if the model expects it (the API fixes it internally to `"CAL"`)
- Runs a trained **scikit-learn** model and returns multi-label outputs

---

## Getting Started

### Clone

```bash
git clone <your-repo-url>.git
cd bird-calls-detection
```

## Running with Docker

The project includes a `Dockerfile` based on Python 3.12 slim, pre-configured with dependencies like `ffmpeg` and `uv`.

**Build the Docker image:**
```bash
docker build -t my-inference-app .
```

**Run the container:**
```bash
docker run -p 8000:8000 my-inference-app
```

**Once running, the API will be accessible at:**
```bash
http://localhost:8000
```

