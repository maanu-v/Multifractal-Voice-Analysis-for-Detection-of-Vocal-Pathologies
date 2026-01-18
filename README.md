# Multifractal Analysis–Based Multiclass Classification of Vocal Pathologies

## 1. Overview
This project investigates **how different types of vocal disorders affect the speech signal** and whether those differences can be **captured and classified** using advanced signal processing techniques. Specifically, it explores the combination of:
*   **MFCC (Mel-frequency cepstral coefficients)**: capturing spectral and perceptual information.
*   **MF-DFA (Multifractal Detrended Fluctuation Analysis)**: capturing nonlinear, multiscale irregularities.

Instead of treating every disease separately, this project identifies **physiological mechanisms** of voice production to classify pathologies into broader, clinically meaningful categories.

## 2. Motivation
Voice disorders arise from various causes, but many automatic systems simply classify a voice as "normal" or "pathological." This project aims to answer a deeper question:

> **Do structural and neurological voice disorders leave different multiscale signatures in speech signals?**

If successful, this approach can:
*   Improve early screening.
*   Enhance clinical interpretation.
*   Increase the relevance of automated analysis in medical settings.

## 3. Methodology

### 3.1. Classification Evaluation
The project classifies voice signals into three distinct classes based on physiological mechanisms:

#### 🟢 Healthy
*   Normal vocal fold structure.
*   Normal neural control and regular vibration.

#### 🔵 Structural Pathologies
*   **Cause**: Physical changes to vocal folds (e.g., nodules, polyps, edema).
*   **Effect**: Turbulent airflow and noisy phonation.
*   **Examples**: Phonationsknötchen, Stimmlippenpolyp, Reinke Ödem.

#### 🔴 Neurological Pathologies
*   **Cause**: Impaired neural/motor control (e.g., paralysis, Parkinson’s, spasms).
*   **Effect**: Instability, tremor, and timing irregularities.
*   **Examples**: Rekurrensparese, Morbus Parkinson, Spasmodische Dysphonie.

> **Note**: Functional disorders are excluded due to their high acoustic variability and lack of a consistent physiological mechanism.

### 3.2. Feature Extraction
We utilize a hybrid feature set to capture different aspects of the voice signal:

| Feature | Type | Purpose |
| :--- | :--- | :--- |
| **MFCC** | Spectral / Linear | Captures the "timbre" or what the voice *sounds* like (resonance, noise). Serves as a strong baseline. |
| **MF-DFA** | Temporal / Nonlinear | Captures how the voice *behaves* across time scales (irregularity, instability). Ideal for separating neurological from structural issues. |

## 4. Dataset
We use the **Saarbruecken Voice Database (SVD)**, a widely recognized, clinically annotated dataset containing high-quality recordings.

*   **Input Data**: Sustained vowel `/a/` recordings with normal phonation (`*-a_n.nsp`).
*   **Strategy**: One file per speaker to ensure consistency and reduce variability.

## 5. Workflow

### 5.1. Data Preparation
1.  **Select Pathologies**: specific disorders are mapped to the three target classes.
2.  **Organize Data**: Flatten signals into class-wise folders to prevent leakage.
3.  **Preprocess**: Silence trimming and amplitude normalization (raw data is preserved).

### 5.2. Analysis & Classification
1.  **Feature Extraction**: Compute MFCC statistics (mean, std) and MF-DFA parameters (Hurst exponents, multifractal spectrum width/peak).
2.  **Feature Analysis**: Compare feature distributions per class to ensure statistical significance and interpretability.
3.  **Modeling**: Train Multiclass SVM or Random Forest classifiers.
4.  **Evaluation**: Assess using Accuracy, Precision/Recall, and Confusion Matrices.

## 6. Installation

This project relies on [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repository_url>
cd multifractal-voicepathology

# Sync dependencies (creates .venv)
uv sync
```

## 7. Usage

To run the main application within the environment:

```bash
# Activate the environment
source .venv/bin/activate

# Run the application
uv run main.py
```

Or simply:

```bash
uv run main.py
```

## 8. Expected Outcomes
*   Demonstrate that structural and neurological pathologies exhibit **different multifractal patterns**.
*   Show that MF-DFA adds discriminative power beyond standard MFCCs.
*   Prove that mechanism-based classification is both feasible and interpretable.

---
*This is a signal-analysis-driven medical AI study, focusing on understanding pathology mechanisms rather than just achieving high leaderboard scores.*
