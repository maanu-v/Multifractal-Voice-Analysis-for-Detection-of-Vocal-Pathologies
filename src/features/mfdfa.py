import numpy as np
import pandas as pd
import librosa
import scipy.signal
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union

from src.utils.logger import get_logger
from src.utils.config import (
    AUDIO_DIR, LABELS_CSV, MFDFA_FEATURES_CSV, TARGET_SR
)

logger = get_logger(__name__)

# -----------------------------
# MF-DFA core (Vectorized)
# -----------------------------
def mfdfa(signal: np.ndarray, q_vals: List[float], scales: List[int], m: int = 1) -> Dict[float, float]:
    """
    Perform MF-DFA and return h(q) for given q values using vectorized operations.
    
    Args:
        signal: Input time series (1D array)
        q_vals: List of q orders to compute
        scales: List of scales (window sizes)
        m: Polynomial order for detrending (1 = linear)
        
    Returns:
        Dictionary mapping q to h(q)
    """
    # 1. Integrate the signal (Profile)
    # Ensure zero mean before integration
    signal = signal - np.mean(signal)
    profile = np.cumsum(signal)
    N = len(profile)
    
    hq = {}
    
    # Check strict requirements
    if N < min(scales):
         logger.warning(f"Signal length {N} is shorter than minimum scale {min(scales)}")
         return {q: np.nan for q in q_vals}

    # Pre-calculate F_q(s) for all scales
    F_qs = {q: [] for q in q_vals}
    valid_scales = []

    for s in scales:
        n_seg = N // s
        if n_seg < 2:
            continue
            
        valid_scales.append(s)
        
        # Reshape profile into (n_seg, s)
        # We discard the remainder of the signal
        segments = profile[:n_seg*s].reshape(n_seg, s)
        
        # Detrending
        if m == 1:
            # Linear detrending (fastest with scipy)
            # scipy.signal.detrend subtracts the linear fit
            detrended = scipy.signal.detrend(segments, axis=1, type='linear')
        else:
            # For m != 1, we would need polynomial fitting
            # Implementing vectorized polyfit for generic m if needed, 
            # but for now we stick to m=1 optimization.
            # Fallback to loop if m != 1 (though we assume m=1 for this implementation)
            x = np.arange(s)
            detrended = np.zeros_like(segments)
            for i in range(n_seg):
                coeffs = np.polyfit(x, segments[i], m)
                trend = np.polyval(coeffs, x)
                detrended[i] = segments[i] - trend
        
        # Calculate Variance F^2(v, s)
        # Mean of squared residuals for each segment
        F2 = np.mean(detrended**2, axis=1)
        
        # Calculate F_q(s)
        for q in q_vals:
            if q == 0:
                # F_0(s) = exp( 0.5 * mean( ln(F^2) ) )
                # Add small epsilon to avoid log(0) if perfectly linear (rare in noise)
                f_q = np.exp(0.5 * np.mean(np.log(F2 + 1e-12)))
            else:
                # F_q(s) = ( mean( (F^2)^(q/2) ) )^(1/q)
                f_q = (np.mean(F2 ** (q / 2))) ** (1 / q)
            
            F_qs[q].append(f_q)

    # Calculate h(q) via log-log regression
    log_scales = np.log(valid_scales)
    
    if len(valid_scales) < 3:
        # Insufficient scales for fitting
        return {q: np.nan for q in q_vals}

    for q in q_vals:
        log_F = np.log(F_qs[q])
        # Fit line: log(F(s)) = h(q) * log(s) + C
        slope, _ = np.polyfit(log_scales, log_F, 1)
        hq[q] = slope

    return hq


def extract_multifractal_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Wrapper to extract specific MF-DFA features.
    """
    q_vals = [-5, -3, -1, 0, 1, 3, 5]
    # Scales: logarithmic spacing is better, but user provided linear-ish/geometric mix
    # We'll stick to a geometric progression roughly: 16, 32, ..., 512
    # Ensure scales are integers
    scales = [16, 32, 64, 128, 256, 512]
    
    hq = mfdfa(signal, q_vals, scales, m=1)
    
    features = {}

    # Store h(q) for all q
    for q in q_vals:
        label = f"h_q_neg{-q}" if q < 0 else f"h_q_{q}"
        features[label] = hq[q]

    # Calculate Delta Alpha (Multifractal Spectrum Width)
    # alpha = h(q) + q * h'(q)
    # We estimate h'(q) numerically
    
    q_arr = np.array(q_vals)
    h_arr = np.array([hq[q] for q in q_vals])
    
    # Check for NaNs
    if np.any(np.isnan(h_arr)):
        features["delta_alpha"] = np.nan
    else:
        dhdq = np.gradient(h_arr, q_arr)
        alpha = h_arr + q_arr * dhdq
        delta_alpha = np.max(alpha) - np.min(alpha)
        features["delta_alpha"] = delta_alpha

    return features

# -----------------------------
# Main loop
# -----------------------------
def main():
    if not LABELS_CSV.exists():
        logger.error(f"Labels file not found at {LABELS_CSV}")
        return

    labels = pd.read_csv(LABELS_CSV)
    features = []

    logger.info(f"Extracting MF-DFA features from {len(labels)} files...")
    logger.info(f"Target Output: {MFDFA_FEATURES_CSV}")

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        wav_path = AUDIO_DIR / row["filename"]
        
        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}")
            continue

        try:
            # Load audio
            y, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)

            # Extract MF-DFA
            mf_feats = extract_multifractal_features(y)

            # Append metadata
            mf_feats.update({
                "filename": row["filename"],
                "category": row["category"],
                "pathology": row["pathology"],
                "speaker_id": row["speaker_id"]
            })
            
            features.append(mf_feats)

        except Exception as e:
            logger.warning(f"MF-DFA failed for {row['filename']}: {e}")

    # Save Results
    df = pd.DataFrame(features)
    
    # Ensure directory exists
    MFDFA_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MFDFA_FEATURES_CSV, index=False)

    logger.info("MF-DFA extraction complete.")
    logger.info(f"Saved to {MFDFA_FEATURES_CSV}")

    # Sanity Check / Summary
    if not df.empty:
        print("\n--- MF-DFA Summary (Comparison) ---")
        cols = ["delta_alpha", "h_q_neg5", "h_q_0", "h_q_5"]
        # Filter cols that exist
        cols = [c for c in cols if c in df.columns]
        print(df.groupby("category")[cols].describe())

if __name__ == "__main__":
    main()
