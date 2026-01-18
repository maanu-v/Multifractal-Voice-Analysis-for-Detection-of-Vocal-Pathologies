import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Union
import logging

# -----------------------------
# Configuration
# -----------------------------
TARGET_SR = 16000
N_MFCC = 13
N_FFT = int(0.025 * TARGET_SR)   # 25 ms -> 400 samples
HOP_LENGTH = int(0.010 * TARGET_SR)  # 10 ms -> 160 samples

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent.parent.parent
AUDIO_DIR = BASE_DIR / "data" / "processed" / "audio"
LABELS_CSV = BASE_DIR / "data" / "processed" / "labels.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "mfcc_features.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# MFCC extraction function
# -----------------------------
def extract_mfcc(wav_path: Union[str, Path], sr: int = TARGET_SR, n_mfcc: int = N_MFCC) -> Dict[str, float]:
    """
    Extracts MFCC features (mean and std) from an audio file.
    Drops the 0th MFCC coefficient (energy).
    """
    try:
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
    except Exception as e:
        logger.error(f"Failed to load audio {wav_path}: {e}")
        return {}

    # Extract MFCCs
    try:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Drop MFCC 0 (energy)
        # Result shape: (n_mfcc-1, time_steps)
        mfcc = mfcc[1:, :]
        
        features = {}
        for i in range(mfcc.shape[0]):
            # 1-indexed names (mfcc1 to mfcc12)
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
            
        return features
        
    except Exception as e:
        logger.error(f"MFCC computation failed for {wav_path}: {e}")
        return {}

# -----------------------------
# Main loop
# -----------------------------
def main():
    if not LABELS_CSV.exists():
        logger.error(f"Labels file not found: {LABELS_CSV}")
        return

    labels = pd.read_csv(LABELS_CSV)
    all_features = []

    logger.info(f"Extracting MFCC features from {len(labels)} files...")

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Processing"):
        wav_path = AUDIO_DIR / row["filename"]

        if not wav_path.exists():
            logger.warning(f"File not found: {wav_path}")
            continue

        mfcc_feat = extract_mfcc(wav_path)
        
        if mfcc_feat:
            mfcc_feat.update({
                "filename": row["filename"],
                "category": row["category"],
                "pathology": row["pathology"],
                "speaker_id": row["speaker_id"]
            })
            all_features.append(mfcc_feat)
        else:
            # Handle failure appropriately - maybe record NaN row?
            # For now, skipping failed extractions to keep dataset clean, 
            # or we could append a row of NaNs.
            # Let's append if strictly needed, but empty dict implies skipping.
            # User's previous script just printed error and continued.
            pass

    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Check for NaNs
    if not df.empty:
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning("Some features contain NaNs:")
            print(nan_counts[nan_counts > 0])
            
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("MFCC feature extraction complete.")
    logger.info(f"Saved to: {OUTPUT_CSV}")
    if not df.empty:
        print("\n--- Summary by Category ---")
        print(df.groupby("category").size())

if __name__ == "__main__":
    main()
