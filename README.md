# Project Prederetion: A Proactive Road Safety Intelligence System
---

A real-time road safety system for Singapore that combines dynamic risk indexing with edge-based collision detection, integrated into a live user-facing prototype.

Find it here: https://dlw2026-prederetion-frontend-production.up.railway.app/

## System Overview

Prederetion operates via two integrated components:

**Component 1. Prevention (Dynamic Risk Indexing)**
An unsupervised Autoencoder model continuously pulls data from LTA's public API (speed limits, road type, incidents, time of day) and computes a real-time Risk Index (0–100) for each road segment based on reconstruction error anomaly scoring.

**Component 2. Detection-Reaction (Edge-Based Incident Response)**
A supervised ResNet-18 CNN model trained on 5,399 labeled traffic images performs binary classification (accident / normal) on live road camera feeds, triggering alerts and dynamically updating the local Risk Index.

| Risk Level | Index Range | Proposed Action |
|---|---|---|
| 🔴 High | 76 – 100 | Cautionary alert on OBU |
| 🟠 Moderate | 51 – 75 | Standard display on OBU |
| 🟢 Low | 1 – 50 | Hidden from display |

---

## Project Structure

```
/
├── model-training/        # Training notebooks for both models
│   ├── risk_index_model   # Autoencoder (Component 1)
│   └── collision_detector # ResNet-18 (Component 2)
├── testbench/             # Judge evaluation materials
│   ├── TESTBENCH_README.md
│   └── judges_testing.ipynb
├── frontend/              # Vanilla JS, Leaflet.js, Nominatim, OSRM
├── node-server/           # Node.js + Express
└── python_inf/            # Python server, Flask (model serving)
```

---

## Tech Stack

- **Frontend:** Vanilla JS, Leaflet.js, Nominatim (OpenStreetMap), OSRM
- **Backend:** Node.js + Express, node-fetch, dotenv
- **Models:** Python, PyTorch, Flask (HTTP wrapper), Pickle (model serialisation)
- **Deployment:** Railway (PaaS)

---

## Model Training

Training notebooks for both models are located under `/model-training`.

- The **Risk Index model** notebook covers data collection via LTA's API, dataset construction, Autoencoder training, and Z-score/Risk Index computation.
- The **Collision Detector model** notebook covers dataset preparation, ResNet-18 fine-tuning, and model export to pickle.

---

## Testing

Refer to [`testbench/TESTBENCH_README.md`](testbench/TESTBENCH_README.md) for full judge evaluation instructions.

**Quick summary:**

| Component | How to Test |
|---|---|
| Component 1 — Risk Index | Directly via the live website |
| Component 2 — Collision Detector | Via the image bank in `testbench/judges_testing.ipynb` |

---
