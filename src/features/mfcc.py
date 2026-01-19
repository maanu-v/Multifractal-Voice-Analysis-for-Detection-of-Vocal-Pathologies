import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import librosa
import librosa.display
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict

from src.utils.logger import get_logger
from src.utils.config import (
    AUDIO_DIR, LABELS_CSV, MFCC_FEATURES_CSV, MFCC_REPORTS_DIR,
    TARGET_SR, MFCC_N_MFCC, MFCC_N_FFT, MFCC_HOP_LENGTH
)

logger = get_logger(__name__)

# -----------------------------
# Configuration
# -----------------------------
TARGET_SR = TARGET_SR
N_MFCC = MFCC_N_MFCC
N_FFT = MFCC_N_FFT
HOP_LENGTH = MFCC_HOP_LENGTH

# -----------------------------
# Paths
# -----------------------------
OUTPUT_CSV = MFCC_FEATURES_CSV
REPORTS_DIR = MFCC_REPORTS_DIR
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
# Visualization Functions
# -----------------------------
def plot_mfcc_distributions(df: pd.DataFrame):
    """
    Plots boxplots of the first 3 MFCC means by category to show spread.
    """
    logger.info("Generating MFCC distribution plots...")
    
    # Select first 3 MFCC means for visualization
    features_to_plot = ["mfcc1_mean", "mfcc2_mean", "mfcc3_mean"]
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features_to_plot):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x="category", y=feature, hue="category", data=df, palette="Set2", legend=False)
        plt.title(f"Distribution of {feature}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    output_path = REPORTS_DIR / "mfcc_distributions.png"
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")

def plot_sample_mfccs(df: pd.DataFrame, n_samples: int = 3):
    """
    Plots MFCC spectrograms for n_samples random files from each category.
    """
    logger.info(f"Generating sample MFCC plots ({n_samples} per category)...")
    
    categories = df['category'].unique()
    
    for category in categories:
        subset = df[df['category'] == category]
        samples = subset.sample(min(n_samples, len(subset)))
        
        plt.figure(figsize=(15, 4 * len(samples)))
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            wav_path = AUDIO_DIR / row['filename']
            try:
                y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                mfcc = librosa.feature.mfcc(
                    y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
                )
                mfcc = mfcc[1:, :] # Drop 0th coeff
                
                plt.subplot(len(samples), 1, idx+1)
                librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis='time')
                plt.colorbar()
                plt.title(f"MFCC: {row['filename']} ({category})")
                plt.ylabel('MFCC Coeff')
                plt.tight_layout()
                
            except Exception as e:
                logger.error(f"Failed to plot MFCC for {row['filename']}: {e}")
                
        output_path = REPORTS_DIR / f"mfcc_samples_{category}.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved sample plots to {output_path}")

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
            pass

    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Check for NaNs
    if not df.empty:
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning("Some features contain NaNs:")
            print(nan_counts[nan_counts > 0])
            
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info("MFCC feature extraction complete.")
    logger.info(f"Saved to: {OUTPUT_CSV}")
    if not df.empty:
        print("\n--- Summary by Category ---")
        print(df.groupby("category").size())
        
        # Generate Visualizations
        plot_mfcc_distributions(df)
        plot_sample_mfccs(df, n_samples=3)

if __name__ == "__main__":
    main()
