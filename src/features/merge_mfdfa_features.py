import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import (
    BASELINE_FD_FEATURES_CSV, 
    MFDFA_FEATURES_CSV, 
    MULTIFRACTAL_FEATURES_CSV
)

logger = get_logger(__name__)

def main():
    logger.info("Merging features...")

    # Load existing feature sets
    if not BASELINE_FD_FEATURES_CSV.exists():
        logger.error(f"Base features not found at {BASELINE_FD_FEATURES_CSV}")
        return
    
    if not MFDFA_FEATURES_CSV.exists():
        logger.error(f"MF-DFA features not found at {MFDFA_FEATURES_CSV}")
        return

    base_df = pd.read_csv(BASELINE_FD_FEATURES_CSV)
    mfdfa_df = pd.read_csv(MFDFA_FEATURES_CSV)

    logger.info(f"Base Features (Classic+FD): {base_df.shape}")
    logger.info(f"MF-DFA Features: {mfdfa_df.shape}")

    # Merge
    # Ensure strict inner merge to keep only samples successful in all steps
    merge_keys = ["filename", "category", "pathology", "speaker_id"]
    
    merged = pd.merge(
        base_df,
        mfdfa_df,
        on=merge_keys,
        how="inner"
    )

    logger.info(f"Merged Shape: {merged.shape}")
    
    # Sanity Check
    if len(merged) != len(base_df):
        logger.warning(f"Dropped {len(base_df) - len(merged)} samples during merge.")

    # Save
    MULTIFRACTAL_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MULTIFRACTAL_FEATURES_CSV, index=False)
    logger.info(f"Saved multifractal feature set to {MULTIFRACTAL_FEATURES_CSV}")

if __name__ == "__main__":
    main()
