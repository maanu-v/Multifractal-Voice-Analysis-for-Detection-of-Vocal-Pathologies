import parselmouth
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Union, Any
import logging

from src.config import (
    AUDIO_DIR, LABELS_CSV, BASELINE_FEATURES_CSV,
    PRAAT_PITCH_FLOOR, PRAAT_PITCH_CEILING, 
    PRAAT_SILENCE_THRESHOLD, PRAAT_VOICING_THRESHOLD
)

# -----------------------------
# Configuration
# -----------------------------
PITCH_FLOOR = PRAAT_PITCH_FLOOR
PITCH_CEILING = PRAAT_PITCH_CEILING
TIME_STEP = None  # None for auto
MAX_CANDIDATES = 15
SILENCE_THRESHOLD = PRAAT_SILENCE_THRESHOLD
VOICING_THRESHOLD = PRAAT_VOICING_THRESHOLD
OCTAVE_COST = 0.01
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14

# -----------------------------
# Paths
# -----------------------------
OUTPUT_CSV = BASELINE_FEATURES_CSV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_praat_features(wav_path: Union[str, Path]) -> Dict[str, float]:
    """
    Extracts acoustic features (F0, Jitter, Shimmer, HNR) using Parselmouth (Praat).
    
    Args:
        wav_path: Path to the .wav audio file.
        
    Returns:
        Dictionary containing extracted features or NaNs if extraction fails.
    """
    try:
        snd = parselmouth.Sound(str(wav_path))
    except Exception as e:
        logger.error(f"Failed to load audio {wav_path}: {e}")
        return {k: np.nan for k in ["mean_f0", "std_f0", "jitter_local", "shimmer_local", "hnr"]}

    # 1. Pitch (F0) Extraction
    # Important: Use consistent parameters across Pulse and Pitch objects
    pitch = snd.to_pitch(
        time_step=TIME_STEP,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING
    )
    
    f0_values = pitch.selected_array['frequency']
    # Filter out unvoiced segments (0 Hz)
    f0_values = f0_values[f0_values > 0]

    mean_f0 = np.mean(f0_values) if len(f0_values) > 0 else np.nan
    std_f0 = np.std(f0_values) if len(f0_values) > 0 else np.nan

    # 2. Jitter & Shimmer (Requires PointProcess)
    # We use the cross-correlation method (cc) which is generally more robust
    try:
        point_process = parselmouth.praat.call(
            snd, 
            "To PointProcess (periodic, cc)", 
            PITCH_FLOOR, 
            PITCH_CEILING
        )
        
        # Jitter (local)
        # range: 0 to 0.02s ideally covers period-to-period variations
        jitter = parselmouth.praat.call(
            point_process, 
            "Get jitter (local)", 
            0, 0, 0.0001, 0.02, 1.3
        )
        
        # Shimmer (local)
        shimmer = parselmouth.praat.call(
            [snd, point_process], 
            "Get shimmer (local)", 
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        # If PointProcess fails (e.g. valid pitch not detected), return NaN
        jitter = np.nan
        shimmer = np.nan

    # 3. Harmonics-to-Noise Ratio (HNR)
    try:
        # Use explicit Praat call for robust HNR extraction
        # Parameters: time step (0.01), min pitch (75), silence threshold (0.1), periods per window (1.0)
        harmonicity = parselmouth.praat.call(
            snd, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0
        )
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    except Exception as e:
        logger.error(f"HNR Extraction failed: {e}")
        hnr = np.nan

    return {
        "mean_f0": mean_f0,
        "std_f0": std_f0,
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "hnr": hnr
    }

# -----------------------------
# Main loop
# -----------------------------
def main():
    if not LABELS_CSV.exists():
        logger.error(f"Labels file not found: {LABELS_CSV}")
        return

    labels = pd.read_csv(LABELS_CSV)
    features = []

    logger.info(f"Extracting Praat baseline features from {len(labels)} files...")

    # Using tqdm for progress bar
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Processing"):
        wav_path = AUDIO_DIR / row["filename"]
        
        if not wav_path.exists():
            logger.warning(f"File not found: {wav_path}")
            continue

        feat = extract_praat_features(wav_path)
        
        # Merge metadata with features
        feat_entry = {
            "filename": row["filename"],
            "category": row["category"],
            "pathology": row["pathology"],
            "speaker_id": row["speaker_id"],
            **feat
        }
        features.append(feat_entry)

    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    # Check for NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning("Some features contain NaNs (extraction failed for some samples):")
        print(nan_counts[nan_counts > 0])

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    
    logger.info("Baseline feature extraction complete.")
    logger.info(f"Saved to: {OUTPUT_CSV}")
    print("\n--- Summary by Category ---")
    print(df.groupby("category")[["mean_f0", "jitter_local", "hnr"]].mean())

if __name__ == "__main__":
    main()
