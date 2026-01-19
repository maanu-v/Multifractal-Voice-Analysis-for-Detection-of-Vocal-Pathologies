import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import numba

from src.utils.logger import get_logger
from src.utils.config import (
    AUDIO_DIR, LABELS_CSV, HIGUCHI_FD_FEATURES_CSV, 
    TARGET_SR, HIGUCHI_K_MAX
)

logger = get_logger(__name__)

# -----------------------------
# Higuchi FD function (Numba Optimized)
# -----------------------------
@numba.jit(nopython=True)
def _higuchi_fd_calc(signal: np.ndarray, kmax: int) -> float:
    """
    Core calculation of Higuchi Fractal Dimension using Numba.
    """
    N = len(signal)
    L = np.zeros(kmax)
    x = np.arange(1, kmax + 1)
    
    for k in range(1, kmax + 1):
        Lk = 0.0
        for m in range(k):
            Lmk = 0.0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            
            non = (N - 1) / (int((N - m) / k) * k ** 2)
            Lk += Lmk * non
        
        L[k - 1] = Lk / k

    # Linear fit in log-log space (manual regression for numba compatibility or just return L)
    # Numba supports some numpy, but polyfit is safer done outside or manually implemented.
    # To keep it simple in numba, we'll implement simple linear regression or return arrays.
    
    return L

def higuchi_fd(signal: np.ndarray, kmax: int = HIGUCHI_K_MAX) -> float:
    """
    Compute Higuchi Fractal Dimension of a 1D signal.
    """
    try:
        # Pre-check for signal length
        if len(signal) < kmax * 2:
            return np.nan

        # Numba calculation returns L values
        L = _higuchi_fd_calc(signal, kmax)
        k_values = np.arange(1, kmax + 1)
        
        # Log-log transformation
        log_k = np.log(1.0 / k_values)
        log_L = np.log(L)
        
        # Linear regression
        slope, _ = np.polyfit(log_k, log_L, 1)
        return slope
        
    except Exception as e:
        logger.error(f"Error in HFD calculation: {e}")
        return np.nan

# -----------------------------
# Main loop
# -----------------------------
def main():
    if not LABELS_CSV.exists():
        logger.error(f"Labels file not found at {LABELS_CSV}")
        return

    labels = pd.read_csv(LABELS_CSV)
    features = []

    logger.info(f"Extracting Higuchi FD from {len(labels)} files...")

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        wav_path = AUDIO_DIR / row["filename"]

        try:
            # Load audio (mono)
            y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            
            # Normalize if not already (should be done in preprocess, but safe to verify/redo for robustness)
            # Preprocess already does peak normalization to [-1, 1]
            
            fd = higuchi_fd(y, kmax=HIGUCHI_K_MAX)

            # Sanity Check
            if not np.isnan(fd):
                if fd < 1.0 or fd > 2.2: # Relaxed slightly for outliers, but warn
                    logger.warning(f"Abnormal FD value for {row['filename']}: {fd:.4f}")
                
                # Check for extremely invalid global values check later
            else:
                 logger.warning(f"NaN FD for {row['filename']}")

            features.append({
                "filename": row["filename"],
                "category": row["category"],
                "pathology": row["pathology"],
                "speaker_id": row["speaker_id"],
                "FD_full": fd
            })

        except Exception as e:
            logger.warning(f"FD failed for {row['filename']}: {e}")

    df = pd.DataFrame(features)
    HIGUCHI_FD_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HIGUCHI_FD_FEATURES_CSV, index=False)

    logger.info("Higuchi FD extraction complete.")
    logger.info(f"Saved to: {HIGUCHI_FD_FEATURES_CSV}")

    # Summary Statistics
    desc = df.groupby("category")["FD_full"].describe()
    logger.info("\n--- FD Summary by Category ---\n" + desc.to_string())

if __name__ == "__main__":
    main()
