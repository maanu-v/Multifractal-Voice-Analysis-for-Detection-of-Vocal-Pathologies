import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import (
    HIGUCHI_FD_FEATURES_CSV,
    WAVELET_FD_FEATURES_CSV,
    MFDFA_FEATURES_CSV,
    FD_MFDFA_FEATURES_CSV
)

logger = get_logger(__name__)

def main():
    logger.info("Merging Fractal Features (Higuchi + Wavelet + MF-DFA)...")

    # Load feature sets
    if not HIGUCHI_FD_FEATURES_CSV.exists():
        logger.error(f"Higuchi features not found at {HIGUCHI_FD_FEATURES_CSV}")
        return
    
    if not WAVELET_FD_FEATURES_CSV.exists():
        logger.error(f"Wavelet features not found at {WAVELET_FD_FEATURES_CSV}")
        return

    if not MFDFA_FEATURES_CSV.exists():
        logger.error(f"MF-DFA features not found at {MFDFA_FEATURES_CSV}")
        return

    hfd_df = pd.read_csv(HIGUCHI_FD_FEATURES_CSV)
    wfd_df = pd.read_csv(WAVELET_FD_FEATURES_CSV)
    mfdfa_df = pd.read_csv(MFDFA_FEATURES_CSV)

    logger.info(f"Higuchi Features: {hfd_df.shape}")
    logger.info(f"Wavelet Features: {wfd_df.shape}")
    logger.info(f"MF-DFA Features: {mfdfa_df.shape}")

    # Merge Keys
    merge_keys = ["filename", "category", "pathology", "speaker_id"]

    # Sequential Merge
    merged = pd.merge(hfd_df, wfd_df, on=merge_keys, how="inner")
    merged = pd.merge(merged, mfdfa_df, on=merge_keys, how="inner")

    logger.info(f"Merged Shape: {merged.shape}")
    
    # Save
    FD_MFDFA_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FD_MFDFA_FEATURES_CSV, index=False)
    logger.info(f"Saved merged fractal features to {FD_MFDFA_FEATURES_CSV}")

if __name__ == "__main__":
    main()
