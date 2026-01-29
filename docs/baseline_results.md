# Baseline Classification Results

## 1. Objective

The goal of this baseline experiment is to evaluate how well **classical acoustic features** can discriminate between three voice categories:

- **Healthy**
- **Neurological pathologies**
- **Structural pathologies**

This baseline serves as a reference point against which more advanced **multifractal-based features** will be compared.

---

## 2. Dataset Summary

- **Total samples:** 1082  
- **One sample per speaker** (validated)
- **Class distribution:**

| Category        | Samples |
|-----------------|---------|
| Healthy         | 687     |
| Neurological    | 270     |
| Structural      | 125     |

The dataset is clearly **imbalanced**, reflecting real-world clinical prevalence.

---

## 3. Feature Set (Baseline)

Each audio file is represented using a combination of **standard clinical and spectral features**:

### 3.1 Praat-based clinical features
Extracted using Parselmouth (Praat backend):

- Mean fundamental frequency (F0)
- Standard deviation of F0
- Local jitter
- Local shimmer
- Harmonics-to-noise ratio (HNR)

### 3.2 MFCC features
- 12 MFCC coefficients (excluding MFCC0)
- Mean and standard deviation per coefficient

**Total features per sample:** 29

---

## 4. Models Evaluated

Two commonly used baseline classifiers were evaluated:

1. **Logistic Regression**
   - Linear classifier
   - Standardized features
   - Hyperparameter tuning using grid search

2. **Random Forest**
   - Nonlinear ensemble model
   - Default and tuned configurations evaluated

All models were trained using a **stratified train–test split** to preserve class proportions.

---

## 5. Results

### 5.1 Logistic Regression

#### Default model
- **Accuracy:** 70.05%

#### Tuned model
- **Accuracy:** 71.89%

**Classification report (tuned):**

| Class         | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| Healthy       | 0.76      | 0.93   | 0.84     |
| Neurological  | 0.56      | 0.43   | 0.48     |
| Structural    | 0.62      | 0.20   | 0.30     |
| **Macro Avg** | 0.65      | 0.52   | 0.54     |

**Observation:**
- Healthy voices are classified reliably.
- Neurological voices show moderate separability.
- Structural voices are poorly recalled.

---

### 5.2 Random Forest

#### Default model
- **Accuracy:** 73.73%

#### Tuned model
- **Accuracy:** 73.73% (no improvement)

**Classification report (tuned):**

| Class         | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| Healthy       | 0.78      | 0.96   | 0.86     |
| Neurological  | 0.60      | 0.50   | 0.55     |
| Structural    | 0.00      | 0.00   | 0.00     |
| **Macro Avg** | 0.46      | 0.49   | 0.47     |

**Observation:**
- The model completely fails to predict the structural class.
- Even with nonlinear decision boundaries, structural pathologies remain indistinguishable using baseline features.

---

## 6. Key Insights

1. **Healthy voices are easily separable** using classical acoustic and spectral features.
2. **Neurological pathologies show partial separability**, likely due to increased temporal irregularities captured by jitter, shimmer, and MFCC variance.
3. **Structural pathologies are consistently misclassified**, often confused with healthy or neurological samples.
4. Increasing model complexity (Random Forest vs Logistic Regression) **does not resolve structural misclassification**.

---

## 7. Interpretation and Motivation

The inability of baseline features to characterize structural voice disorders suggests that:

- Short-term spectral features (MFCCs) are insufficient.
- Cycle-level perturbation measures (jitter/shimmer) capture only local irregularities.
- Structural disorders likely introduce **heterogeneous, multi-scale, nonlinear dynamics** that are not represented in conventional features.

This limitation directly motivates the use of **multifractal analysis**, which is designed to capture:

- Long-range correlations
- Scale-dependent fluctuations
- Nonlinear temporal structure

---

## 8. Conclusion (Baseline)

The baseline experiments establish that while traditional features provide reasonable performance for healthy voice detection, they fail to adequately represent structural voice pathologies.

These results justify the exploration of **multifractal detrended fluctuation analysis (MF-DFA)** as a complementary feature extraction approach to improve discrimination, particularly for structurally induced voice disorders.

The baseline results will serve as a reference for evaluating the effectiveness of multifractal features in subsequent experiments.
