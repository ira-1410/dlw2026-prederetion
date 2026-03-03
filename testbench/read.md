# Testbench README
## Vehicle Collision Detector — Judge Evaluation Guide

---

## Overview

This project contains two models: one for risk index and one for vehicle collision detection. Each model has a dedicated testing method. Judges should follow the relevant section for the model they wish to evaluate.

---

## Model 1 — Web Interface Testing

**Testing Method:** Live website — no setup required

Model 1 can be tested directly on the web interface. 

### Steps

1. Navigate to the [project website](https://dlw2026-prederetion-frontend-production.up.railway.app/)
2. Select a start and end destination and click 'Get Route'
3. Risk Indices are classified into three classes ('Normal', 'Medium', 'High') and used to highlight the route.


---

## Model 2 — Jupyter Notebook Testing

**Testing Method:** `judges_testing.ipynb` with a pre-trained pickle file and image bank

Since the LTA API uses live data, the probably of seeing a collision is low. We created a notebook for judges to test the model directly by inputting images to verify the model can predict both classes. 

The notebook loads the pre-trained model directly from a public pickle file and runs an interactive judge testing session — no training or downloading data is required.

### Prerequisites

- Google Colab or a local Jupyter environment
- Internet connection (to download the pickle file and fetch live LTA camera data)

### Steps

1. Open `judges_testing.ipynb` in Google Colab or Jupyter.
2. Run all cells in order from top to bottom.
3. When prompted, enter a **camera number (1–30)** to select an LTA traffic camera location.
4. When prompted, enter an **image number (1–30)** to select a test image from the image bank.
5. The model will output the collision detection result, confidence score, and a correct/incorrect verdict against the true label.

### Image Bank

The notebook includes a built-in image bank of 30 test images:

| Images | Label |
|--------|-------|
| 1 – 15 | 🚨 Accident |
| 16 – 30 | ✅ No Accident |

> **Note:** The pickle file is downloaded automatically in Cell 1 via a public Google Drive link. No manual file transfer is needed.
