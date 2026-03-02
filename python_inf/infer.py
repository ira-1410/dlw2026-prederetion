"""
infer.py
─────────────────
Standalone inference module for the LTA Road Risk Index.
Called by the Node.js server via child_process or a persistent HTTP wrapper.

Usage (single score):
    python risk_inference.py '{"speed_band":4,"min_speed":30,...}'

Usage (batch):
    python risk_inference.py '[{"link_id":"123",...}, {"link_id":"456",...}]'

Returns JSON to stdout:
    Single: {"link_id": "123", "risk_index": 42.7, "reconstruction_error": 0.18}
    Batch:  [{"link_id": "123", ...}, {"link_id": "456", ...}]

Required files in ./lta_data/:
    lstm_autoencoder.pt      — model weights
    scaler.pkl               — fitted StandardScaler
    baseline_tracker.pkl     — BaselineTracker with per-(segment,hour) stats
"""

import sys
import json
import math
import pickle
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image
import io

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "model"
MODEL_PATH  = DATA_DIR / "lstm_autoencoder.pt"
SCALER_PATH = DATA_DIR / "scaler.pkl"
BASELINE_PATH = DATA_DIR / "baseline_tracker.pkl"
CNN_MODEL_PATH = DATA_DIR / "accident_classifier_checkpoint.pkl" 

# ── Feature columns — must exactly match training order ──────────────────
FEATURE_COLS = [
    # Speed (3)
    "speed_band", "min_speed", "max_speed",
    # Speed anomaly (2)
    "speed_deviation", "speed_drop_rate",
    # Incidents (7)
    "incident_score", "incident_count", "nearest_incident_dist",
    "flag_accident", "flag_breakdown", "flag_weather", "flag_roadwork",
    # Traffic lights (3)
    "faulty_light_score", "blackout_count", "flashing_count",
    # VMS (2)
    "emas_breakdown_score", "emas_rain_score",
    # Flood (2)
    "flood_severity_score", "flood_nearby",
    # Road category one-hot (7)
    "road_cat_1", "road_cat_2", "road_cat_3", "road_cat_4",
    "road_cat_5", "road_cat_6", "road_cat_8",
    # Time (6)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_peak_hour", "is_late_night",
]
FEATURE_DIM = len(FEATURE_COLS)   # 32

# ── Domain multipliers ────────────────────────────────────────────────────
DOMAIN_MULTIPLIERS = {
    "active_accident":       2.0,
    "flood_extreme":         1.8,
    "flood_severe":          1.5,
    "flood_moderate":        1.3,
    "flood_minor":           1.1,
    "faulty_light_blackout": 1.6,
    "faulty_light_flashing": 1.3,
    "late_night":            1.3,
    "peak_hour":             1.1,
    "active_roadworks":      1.2,
    "emas_breakdown":        1.4,
    "emas_rain":             1.2,
}

RISK_BIAS         = -2.0
RISK_SCALE_FACTOR = 2.5

# ── CNN Definition ────────────────────────────────────────────────────────
def build_cnn_model(num_classes=2):
    # ResNet18 architecture must match the training notebook
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

cnn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Model definition (must match training exactly) ────────────────────────
class FeedforwardAutoencoder(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM,
                 hidden_dims: list = [32, 16, 8],
                 dropout: float = 0.2):
        super().__init__()

        enc, in_dim = [], feature_dim
        for h in hidden_dims:
            enc += [nn.Linear(in_dim, h), nn.ReLU(),
                    nn.BatchNorm1d(h), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*enc)

        dec = []
        for h in reversed(hidden_dims[:-1]):
            dec += [nn.Linear(in_dim, h), nn.ReLU(),
                    nn.BatchNorm1d(h), nn.Dropout(dropout)]
            in_dim = h
        dec.append(nn.Linear(in_dim, feature_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        with torch.no_grad():
            return ((x - self(x)) ** 2).mean(dim=1)


# ── Baseline tracker (copied from notebook) ───────────────────────────────
class BaselineTracker:
    def __init__(self):
        self._bins   = defaultdict(lambda: deque(maxlen=500))
        self._global = deque(maxlen=10000)

    def update(self, link_id: str, hour: int, error: float):
        self._bins[(link_id, hour)].append(error)
        self._global.append(error)

    def get_stats(self, link_id: str, hour: int, min_samples: int = 10):
        bin_data = list(self._bins[(link_id, hour)])
        if len(bin_data) >= min_samples:
            return np.mean(bin_data), max(np.std(bin_data), 1e-6)
        global_data = list(self._global)
        if len(global_data) >= min_samples:
            return np.mean(global_data), max(np.std(global_data), 1e-6)
        return 0.5, 0.5

    @classmethod
    def load(cls, path):
        bt = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        for k, v in d["bins"].items():
            bt._bins[k] = deque(v, maxlen=500)
        bt._global = deque(d["global_"], maxlen=10000)
        return bt


# ── Risk scoring helpers ──────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _domain_multiplier(feat: dict) -> float:
    m = 1.0
    if feat.get("flag_accident"):
        m *= DOMAIN_MULTIPLIERS["active_accident"]
    sev = feat.get("flood_severity_score", 0)
    if   sev >= 4: m *= DOMAIN_MULTIPLIERS["flood_extreme"]
    elif sev >= 3: m *= DOMAIN_MULTIPLIERS["flood_severe"]
    elif sev >= 2: m *= DOMAIN_MULTIPLIERS["flood_moderate"]
    elif sev >= 1: m *= DOMAIN_MULTIPLIERS["flood_minor"]
    if feat.get("blackout_count", 0) > 0:
        m *= DOMAIN_MULTIPLIERS["faulty_light_blackout"]
    elif feat.get("flashing_count", 0) > 0:
        m *= DOMAIN_MULTIPLIERS["faulty_light_flashing"]
    if feat.get("is_late_night"):
        m *= DOMAIN_MULTIPLIERS["late_night"]
    elif feat.get("is_peak_hour"):
        m *= DOMAIN_MULTIPLIERS["peak_hour"]
    if feat.get("flag_roadwork"):
        m *= DOMAIN_MULTIPLIERS["active_roadworks"]
    if feat.get("emas_breakdown_score", 0) > 1:
        m *= DOMAIN_MULTIPLIERS["emas_breakdown"]
    if feat.get("emas_rain_score", 0) > 1:
        m *= DOMAIN_MULTIPLIERS["emas_rain"]
    return m


def _error_to_risk(error: float, link_id: str, hour: int,
                   feat: dict, baseline: BaselineTracker) -> float:
    mu, sigma  = baseline.get_stats(link_id, hour)
    z          = (error - mu) / sigma
    dm         = _domain_multiplier(feat)
    raw        = float(np.clip((z * dm + RISK_BIAS) / RISK_SCALE_FACTOR, -20.0, 20.0))
    return round(100.0 * _sigmoid(raw), 2)


# ── Lazy-loaded globals (loaded once on first call) ───────────────────────
_model    = None
_scaler   = None
_baseline = None
_device   = torch.device("cpu")   # CPU-only for inference server
_cnn_model = None
CNN_CLASSES = ['accident', 'normal'] #
IMG_SIZE = None

def _load_artifacts():
    global _model, _scaler, _baseline, _cnn_model
    if _model is not None:
        return   # already loaded

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"Baseline not found: {BASELINE_PATH}")

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    _baseline = BaselineTracker.load(BASELINE_PATH)

    _model = FeedforwardAutoencoder(feature_dim=FEATURE_DIM).to(_device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.eval()

    if _cnn_model is None:
        _cnn_model = build_cnn_model(num_classes=2).to(_device)
        with open(CNN_MODEL_PATH, 'rb') as f:
            checkpoint = pickle.load(f)

        _cnn_model.load_state_dict(checkpoint['model_state_dict'])
        CNN_CLASSES = checkpoint['class_names']
        IMG_SIZE    = checkpoint['img_size']
        print(IMG_SIZE)
        _cnn_model.eval()
        print("✓ CNN Accident Classifier loaded")


# ── Core scoring function ─────────────────────────────────────────────────
def score(feat_dict: dict) -> dict:
    """
    Score a single 32-dim feature dict.

    Args:
        feat_dict: dict with keys matching FEATURE_COLS.
                   Missing keys default to 0.
                   Must include 'link_id' and 'timestamp' (ISO string or datetime).

    Returns:
        dict with keys: link_id, risk_index, reconstruction_error
    """
    _load_artifacts()

    link_id = str(feat_dict.get("link_id", "UNKNOWN"))
    ts      = feat_dict.get("timestamp", datetime.now().isoformat())
    hour    = datetime.fromisoformat(str(ts)).hour if isinstance(ts, str) else ts.hour

    # Build feature vector in exact training order — missing keys → 0
    vec = np.array(
        [float(feat_dict.get(c, 0.0)) for c in FEATURE_COLS],
        dtype=np.float32
    ).reshape(1, -1)

    # Scale + clip (match FlatFeatureDataset preprocessing)
    vec = _scaler.transform(vec)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    vec = np.clip(vec, -5.0, 5.0)

    tensor = torch.tensor(vec, dtype=torch.float32).to(_device)

    _model.eval()
    error    = float(_model.reconstruction_error(tensor).item())
    risk_idx = _error_to_risk(error, link_id, hour, feat_dict, _baseline)

    # Update baseline online so it improves over time
    _baseline.update(link_id, hour, error)

    return {
        "link_id":              link_id,
        "risk_index":           risk_idx,
        "reconstruction_error": round(error, 6),
    }


def score_batch(feat_dicts: list) -> list:
    """
    Score a list of feature dicts in one forward pass.
    More efficient than calling score() in a loop for large batches.
    """
    _load_artifacts()

    if not feat_dicts:
        return []

    link_ids = [str(d.get("link_id", "UNKNOWN")) for d in feat_dicts]
    hours    = []
    for d in feat_dicts:
        ts = d.get("timestamp", datetime.now().isoformat())
        hours.append(datetime.fromisoformat(str(ts)).hour if isinstance(ts, str) else ts.hour)

    # Build (N, 32) matrix
    mat = np.array(
        [[float(d.get(c, 0.0)) for c in FEATURE_COLS] for d in feat_dicts],
        dtype=np.float32
    )
    mat = _scaler.transform(mat)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, -5.0, 5.0)

    tensor = torch.tensor(mat, dtype=torch.float32).to(_device)
    _model.eval()
    errors = _model.reconstruction_error(tensor).cpu().numpy().tolist()

    results = []
    for feat, lid, hour, error in zip(feat_dicts, link_ids, hours, errors):
        risk_idx = _error_to_risk(float(error), lid, hour, feat, _baseline)
        _baseline.update(lid, hour, float(error))
        results.append({
            "link_id":              lid,
            "risk_index":           risk_idx,
            "reconstruction_error": round(float(error), 6),
        })

    return results

def predict_accident(image_bytes: bytes) -> dict:
    _load_artifacts()

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = cnn_transforms(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _cnn_model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    prediction = CNN_CLASSES[idx.item()]

    return {
        "prediction":    prediction,
        "label":         "Collision detected" if prediction == "accident" else "No collision detected",
        "confidence":    round(conf.item(), 4),
        "probabilities": {
            CNN_CLASSES[i]: round(probs[i].item(), 4)
            for i in range(len(CNN_CLASSES))
        },
    }


# ── CLI entry point (called by Node.js child_process) ────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided. Pass a JSON string as argv[1]."}))
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        sys.exit(1)

    try:
        if isinstance(payload, list):
            result = score_batch(payload)
        else:
            result = score(payload)
        print(json.dumps(result))
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Inference failed: {e}"}))
        sys.exit(1)