import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import (
    CLASSIC_FEATURES_CSV, 
    HIGUCHI_FD_FEATURES_CSV, 
    WAVELET_FD_FEATURES_CSV, 
    BASELINE_FD_FEATURES_CSV
)

logger = get_logger(__name__)

def main():
    logger.info("Loading feature files...")

    if not CLASSIC_FEATURES_CSV.exists() or not HIGUCHI_FD_FEATURES_CSV.exists() or not WAVELET_FD_FEATURES_CSV.exists():
        logger.error("One or more feature files are missing.")
        return

    df_base = pd.read_csv(CLASSIC_FEATURES_CSV)
    df_fd = pd.read_csv(HIGUCHI_FD_FEATURES_CSV)
    df_wfd = pd.read_csv(WAVELET_FD_FEATURES_CSV)

    logger.info(f"Classic: {df_base.shape}")
    logger.info(f"Higuchi FD: {df_fd.shape}")
    logger.info(f"Wavelet FD: {df_wfd.shape}")

    # Merge Higuchi FD
    df = df_base.merge(
        df_fd[["filename", "FD_full"]],
        on="filename",
        how="inner"
    )

    # Merge Wavelet FD
    df = df.merge(
        df_wfd[["filename", "FD_A3", "FD_D3", "FD_D2"]],
        on="filename",
        how="inner"
    )

    logger.info(f"Merged samples: {len(df)}")
    logger.info(f"Final shape: {df.shape}")
    
    # Save
    BASELINE_FD_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASELINE_FD_FEATURES_CSV, index=False)
    logger.info(f"Saved merged feature set to {BASELINE_FD_FEATURES_CSV}")

if __name__ == "__main__":
    main()
