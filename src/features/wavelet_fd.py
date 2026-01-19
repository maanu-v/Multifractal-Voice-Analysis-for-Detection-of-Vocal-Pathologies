import numpy as np
import pandas as pd
import librosa
import pywt
from pathlib import Path
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import (
    AUDIO_DIR, LABELS_CSV, WAVELET_FD_FEATURES_CSV,
    TARGET_SR, WAVELET_NAME, WAVELET_LEVEL
)

from src.features.higuchi_fd import higuchi_fd

logger = get_logger(__name__)

# -----------------------------
# Configuration
# -----------------------------
WAVELET = WAVELET_NAME
LEVEL = WAVELET_LEVEL

# -----------------------------
# Wavelet FD function
# -----------------------------
def wavelet_fd(signal: np.ndarray):
    """
    Compute wavelet-based FD features using DWT + Higuchi FD on sub-bands.
    Returns FD for Approximation (A3) and Detail coefficients (D3, D2).
    """
    try:
        # Perform Discrete Wavelet Transform
        # Output for level 3: [cA3, cD3, cD2, cD1]
        coeffs = pywt.wavedec(signal, wavelet=WAVELET, level=LEVEL)
        
        # We focus on the deeper levels as requested/implied by "FD_A3, FD_D3, FD_D2"
        # cA3: Approximation at level 3 (Low freq)
        # cD3: Detail at level 3
        # cD2: Detail at level 2
        
        A3 = coeffs[0]
        D3 = coeffs[1]
        D2 = coeffs[2]
        
        return {
            "FD_A3": higuchi_fd(A3),
            "FD_D3": higuchi_fd(D3),
            "FD_D2": higuchi_fd(D2),
        }
    except Exception as e:
        logger.error(f"DWT failed: {e}")
        return {
            "FD_A3": np.nan,
            "FD_D3": np.nan,
            "FD_D2": np.nan,
        }

# -----------------------------
# Main loop
# -----------------------------
def main():
    if not LABELS_CSV.exists():
        logger.error(f"Labels file not found at {LABELS_CSV}")
        return

    labels = pd.read_csv(LABELS_CSV)
    features = []

    logger.info(f"Extracting Wavelet FD from {len(labels)} files...")
    logger.info(f"Wavelet: {WAVELET}, Level: {LEVEL}")

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        wav_path = AUDIO_DIR / row["filename"]

        try:
            y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            
            # Extract features
            fd_bands = wavelet_fd(y)
            
            # Check for NaNs
            if np.isnan(fd_bands["FD_A3"]):
                logger.warning(f"NaN in Wavelet FD for {row['filename']}")

            features.append({
                "filename": row["filename"],
                "category": row["category"],
                "pathology": row["pathology"],
                "speaker_id": row["speaker_id"],
                **fd_bands
            })

        except Exception as e:
            logger.warning(f"Wavelet FD extraction failed for {row['filename']}: {e}")

    df = pd.DataFrame(features)
    WAVELET_FD_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WAVELET_FD_FEATURES_CSV, index=False)

    logger.info("Wavelet FD extraction complete.")
    logger.info(f"Saved to: {WAVELET_FD_FEATURES_CSV}")

    print("\n--- Wavelet FD Summary ---")
    print(df.groupby("category")[["FD_A3", "FD_D3", "FD_D2"]].describe())

if __name__ == "__main__":
    main()
