from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"
LABELS_CSV = PROCESSED_DIR / "labels.csv"

# Reports
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MFCC_REPORTS_DIR = REPORTS_DIR / "mfcc"
BASELINE_MODEL_DIR = REPORTS_DIR / "baseline_model"

# Feature Files
BASELINE_FEATURES_CSV = PROCESSED_DIR / "baseline_features.csv"
MFCC_FEATURES_CSV = PROCESSED_DIR / "mfcc_features.csv"
CLASSIC_FEATURES_CSV = PROCESSED_DIR / "classic_features.csv"

# -----------------------------
# Audio Settings
# -----------------------------
TARGET_SR = 16000

# -----------------------------
# Feature Extraction Settings
# -----------------------------
# Praat (Baseline)
PRAAT_PITCH_FLOOR = 75.0
PRAAT_PITCH_CEILING = 600.0
PRAAT_SILENCE_THRESHOLD = 0.03
PRAAT_VOICING_THRESHOLD = 0.45

# MFCC
MFCC_N_MFCC = 13
MFCC_N_FFT = int(0.025 * TARGET_SR)  # 25 ms
MFCC_HOP_LENGTH = int(0.010 * TARGET_SR)  # 10 ms

# Higuchi FD
HIGUCHI_K_MAX = 10
FD_FEATURES_CSV = PROCESSED_DIR / "fd_features.csv"

# Wavelet FD
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3
WAVELET_FD_FEATURES_CSV = PROCESSED_DIR / "wavelet_fd_features.csv"
