import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import BASELINE_FEATURES_CSV, MFCC_FEATURES_CSV, CLASSIC_FEATURES_CSV

logger = get_logger(__name__)

def main():
    logger.info("Loading feature files...")

    if not BASELINE_FEATURES_CSV.exists():
        logger.error(f"Baseline features not found at {BASELINE_FEATURES_CSV}")
        return
    
    if not MFCC_FEATURES_CSV.exists():
        logger.error(f"MFCC features not found at {MFCC_FEATURES_CSV}")
        return

    baseline = pd.read_csv(BASELINE_FEATURES_CSV)
    mfcc = pd.read_csv(MFCC_FEATURES_CSV)

    logger.info(f"Baseline shape: {baseline.shape}")
    logger.info(f"MFCC shape: {mfcc.shape}")

    # Merge on filename
    # Identify common columns (keys + potential duplicate metadata)
    # The extraction scripts both save metadata columns. We merge on them to ensure alignment.
    merge_keys = ["filename", "category", "pathology", "speaker_id"]
    
    merged = pd.merge(
        baseline,
        mfcc,
        on=merge_keys,
        how="inner"
    )

    logger.info(f"Merged shape: {merged.shape}")

    # Sanity checks
    if len(merged) != len(baseline):
         logger.warning(f"Row count mismatch! Baseline: {len(baseline)}, Merged: {len(merged)}")
    
    merged.to_csv(CLASSIC_FEATURES_CSV, index=False)
    logger.info(f"Saved merged features to {CLASSIC_FEATURES_CSV}")

    # Final visual check of columns
    logger.info(f"Columns: {list(merged.columns)}")

if __name__ == "__main__":
    main()
