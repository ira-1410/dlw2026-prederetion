import os
import gdown

def download_models():
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    files = {
        "accident_classifier_checkpoint.pkl": "1jc3iPMAJveEG4pjX1TzFHcVlrXdTl9cQ",
        "baseline_tracker.pkl":               "1hyzT-qNRlpuPaGXKH5_20UqvwuR9yyEO",
        "scaler.pkl":                         "1dVuyk10AGasdM3SR1ICAaZfQ2qhcg1Zj",
        "autoencoder.pt":                     "1lfU1VmUiuwPnuNLIqgKrGbybfVajjl1b",
    }

    for filename, file_id in files.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)
        else:
            print(f"{filename} already exists, skipping.")