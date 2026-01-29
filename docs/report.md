# Multifractal Voice Analysis for Detection of Vocal Pathologies

**Aksaya Kanthan K S**  
Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India.  
cb.ai.u4aid23003@cb.students.amrita.edu  

**Kiruthik Nandhan Murthi**  
Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India.  
cb.ai.u4aid23018@cb.students.amrita.edu  

**Manasa V**  
Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India.  
cb.ai.u4aid23022@cb.students.amrita.edu  

**Rithvik Rajesh**  
Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India.  
cb.ai.u4aid23032@cb.students.amrita.edu  

---

**Abstract**—The reliable detection of voice pathologies remains a significant challenge in speech processing, as structural and neurological disorders often manifest in subtle, nonlinear irregularities that traditional acoustic features fail to capture fully. This project proposes a Multifractal Voice Analysis framework to distinguish between healthy voices, structural disorders (e.g., polyps, nodules), and neurological disorders (e.g., Parkinson's, paralysis). By integrating baseline acoustic features with fractal dimension (FD) and Multifractal Detrended Fluctuation Analysis (MF-DFA), we aim to characterize the complex, multiscale dynamics of pathological speech. Preliminary results on the Saarbrücken Voice Database (SVD) demonstrate that pathological voices exhibit higher fractal complexity and wider multifractal spectra compared to healthy voices, suggesting that these nonlinear features offer robust discriminative power for multi-class pathology detection.

**Index Terms**—Voice Pathology Detection, Multifractal Analysis, Fractal Dimension, MF-DFA, Saarbrücken Voice Database, Nonlinear Speech Analysis.

---

## I. INTRODUCTION

The human voice is produced through a complex interaction of airflow, vocal fold vibration, and vocal tract movement. Voice disorders, particularly structural and neurological pathologies, disturb this delicate process, introducing subtle irregularities in speech production. Recent advancements in speech processing have sought to automate the detection of these disorders, yet distinguishing between specific types of pathologies remains difficult.

Traditional automatic speech recognition (ASR) and analysis systems often rely on linear acoustic features such as pitch, jitter, shimmer, and Mel-frequency cepstral coefficients (MFCCs). While these features are effective for general distinction between healthy and unhealthy voices, they capture only short-term or local changes and often fail to reflect the deeper, nonlinear irregularities characteristic of complex pathologies. Structural and neurological disorders can sound remarkably similar, leading to high misclassification rates in conventional systems.

Since speech is inherently a nonlinear and multiscale signal, this project explores the application of fractal and multifractal analysis to better describe its complex behavior. We propose a framework that leverages Higuchi Fractal Dimension (FD) and Multifractal Detrended Fluctuation Analysis (MF-DFA) to extract features representing the long-range correlations and singularity spectra of voice signals. By combining these nonlinear measures with baseline acoustic features, we aim to develop a robust approach for the multi-class classification of vocal pathologies.

## II. METHODOLOGY

The proposed methodology integrates data preparation, baseline feature extraction, and advanced fractal analysis to characterize voice signals.

### A. Dataset Description
The study utilizes the **Saarbrücken Voice Database (SVD)**, a clinically validated dataset widely used in voice pathology research. The data is organized into three major categories:
1.  **Healthy voices**: Speakers without diagnosed voice disorders (687 samples).
2.  **Structural disorders**: Conditions affecting the physical structure of the vocal folds, including *Phonationsknötchen* (vocal nodules), *Stimmlippenpolyp* (vocal polyps), and *Reinke Ödem* (125 samples).
3.  **Neurological disorders**: Conditions affecting neural control, including *Rekurrensparese* (recurrent laryngeal nerve paralysis), *Morbus Parkinson*, and *Spasmodische Dysphonie* (270 samples).

The final dataset consists of 1082 sustained vowel /a/ recordings. All signals are resampled to 16 kHz, converted to mono, and amplitude-normalized to ensure consistency.

![Class Distribution](../reports/data_distribution/class_distribution_bar.png)
*Fig. 1. Distribution of samples across Healthy, Structural, and Neurological classes.*

### B. Baseline Feature Extraction
To establish a strong performance baseline, classical acoustic features are extracted:
*   **Fundamental frequency (F0)**: Measures the pitch of the voice.
*   **Jitter and Shimmer**: Quantify cycle-to-cycle variations in frequency and amplitude, respectively.
*   **Harmonics-to-Noise Ratio (HNR)**: Assesses the purity of the voice signal.
*   **MFCCs**: Represents the spectral envelope and perceptual properties.

### C. Fractal Dimension (FD) Analysis
Fractal analysis provides a measure of signal complexity:
*   **Higuchi Fractal Dimension**: Computed on the time-domain signal to quantify overall waveform complexity.
*   **Wavelet-based Multiresolution FD**: Performed using a 3-level Discrete Wavelet Transform to capture complexity across specific frequency bands.

FD features describe long-range and scale-dependent irregularities, which are expected to be more pronounced in pathological speech.

### D. Multifractal Detrended Fluctuation Analysis (MF-DFA)
MF-DFA is applied to the steady vowel segment to analyze scale-dependent fluctuations. Unlike simple fractal analysis, which assumes a single scaling exponent, MF-DFA characterizes signals that exhibit multiple scaling behaviors. Key features extracted include:
*   **Singularity Spectrum Width (Δα)**: Represents the range of fractal exponents; a wider spectrum indicates greater multifractality and heterogeneity.
*   **Generalized Hurst Exponents h(q)**: Calculated at multiple moment orders ($q$) to capture different fluctuation strengths.

## III. RESULTS AND ANALYSIS

### A. Baseline vs. Nonlinear Features
Descriptive statistics reveal systematic differences between the classes. Healthy voices exhibit higher average F0 and lower jitter, indicating stable phonation. In contrast, pathological voices show higher jitter and lower HNR, reflecting increased vocal irregularity and noise.

The distribution of Jitter, Shimmer, and HNR across the three voice categories is illustrated in Fig. 2.

````carousel
![Jitter Distribution](../reports/plot/baseline/jitter_boxplot.png)
<!-- slide -->
![Shimmer Distribution](../reports/plot/baseline/shimmer_boxplot.png)
<!-- slide -->
![HNR Distribution](../reports/plot/baseline/hnr_boxplot.png)
````
*Fig. 2. Boxplots of Jitter, Shimmer, and HNR features across Healthy, Structural, and Neurological categories.*

The spectral envelopes of representative samples from each category can be visualized through their MFCC coefficients. Fig. 3 displays the heatmap of the first 12 MFCC means, highlighting distinct spectral signatures for different pathologies.

![MFCC Heatmap](../reports/plot/baseline/mfcc_heatmap.png)
*Fig. 3. Heatmap of MFCC coefficients for representative Healthy, Structural, and Neurological voice samples.*

Higuchi Fractal Dimension (FD) analysis reveals consistent trends:
*   **Healthy voices**: Show lower average FD values and smaller variability.
*   **Pathological voices**: Structural and neurological voices show higher FD values and a wider spread, indicating increased signal complexity.
*   **Neurological voices**: Exhibit the greatest FD variability, consistent with the unstable neural control mechanisms involved.

The distribution of Higuchi Fractal Dimension values across the categories is shown in Fig. 7.

![Higuchi FD Distribution](../reports/plot/fd/higuchi_fd_violin.png)
*Fig. 8. Violin plot of Higuchi Fractal Dimension (FD) across Healthy, Structural, and Neurological categories.*

The analysis was further extended to multiresolution wavelet scales (A3, D3, D2). Fig. 9 compares these features, showing how complexity varies across different frequency bands for each category.

![Wavelet FD Comparison](../reports/plot/fd/wavelet_fd_comparison.png)
*Fig. 9. Comparison of Wavelet Fractal Dimensions (A3, D3, D2) across voice categories.*

### B. Multifractal Analysis
The multifractal analysis provides deeper insights into the dynamics of the voice signals.

**Singularity Spectrum Width (Δα)**:
**Singularity Spectrum Width (Δα)**:
As shown in Fig. 4, the singularity spectrum width ($\Delta\alpha$) acts as a discriminative feature.
*   **Healthy voices** generally exhibit narrower spectra, indicating more uniform signal dynamics (monofractal-like behavior).
*   **Structural and Neurological voices** display wider spectra, reflecting higher heterogeneity and stronger multifractality.

![Delta Alpha Violin Plot](../reports/figures/mfdfa_delta_alpha_violin.png)
*Fig. 4. Violin plot of Singularity Spectrum Width ($\Delta\alpha$) across voice classes. Wider distributions in pathological voices indicate stronger multifractality.*

The superposition of singularity spectra $f(\alpha)$ for representative samples further illustrates this behavior. As seen in Fig. 5, the healthy voice exhibits a narrower spectrum (closer to monofractal), while pathological voices show broader, more asymmetric spectra.

![Singularity Spectrum](../reports/figures/mfdfa_singularity_spectrum.png)
*Fig. 5. Representative Multifractal Singularity Spectra $f(\alpha)$ vs $\alpha$.*

**Generalized Hurst Exponents**:
The variation of $h(q)$ across different moments resonates with the degree of multifractality. Fig. 5 illustrates the fluctuation behavior for negative and positive orders ($q$).

![H(q) Fluctuations](../reports/figures/mfdfa_hq_comparison.png)
*Fig. 6. Hurst Exponents h(q) for Small (q=-5) vs Large (q=+5) Fluctuations. The significant difference between negative and positive moments confirms the multifractal nature of the signals.*

### C. Correlation of Features
Fig. 6 shows the relationship between Delta Alpha and Fractal Dimension. The scatter plot highlights the separability of classes when combining overall complexity (FD) with multifractal width ($\Delta\alpha$). Pathological samples tend to cluster in regions of higher complexity and multifractality compared to healthy samples.

![Delta Alpha vs FD](../reports/plot/mfdfa/delta_alpha_vs_fd_scatter.png)
*Fig. 7. Scatter plot of Delta Alpha vs. Fractal Dimension (FD) colored by class.*

## IV. CONCLUSION

This mid-project review presents a comprehensive framework for Multifractal Voice Analysis. We have successfully completed data preparation, preprocessing, and the extraction of baseline, fractal, and multifractal features. Our preliminary analysis confirms that:
1.  **Pathological voices are more complex**: Demonstrated by higher Fractal Dimension values.
2.  **Pathological voices exhibit multifractality**: Evidenced by wider singularity spectra ($\Delta\alpha$) compared to healthy voices.
3.  **Nonlinear features are complementary**: They capture aspects of voice irregularity that traditional linear acoustic measures may miss.

Moving forward, the project will focus on training machine learning classifiers using combinations of these feature sets to quantify the performance improvement offered by the neuro-symbolic and multifractal approach. We aim to specifically address the challenge of distinguishing between structural and neurological disorders.

## REFERENCES

[1] H. Kantz and T. Schreiber, *Nonlinear Time Series Analysis*, Cambridge University Press, 2004.  
[2] T. Higuchi, “Approach to an irregular time series on the basis of the fractal theory,” *Physica D: Nonlinear Phenomena*, vol. 31, no. 2, pp. 277–283, 1988.  
[3] J. W. Kantelhardt et al., “Multifractal detrended fluctuation analysis of nonstationary time series,” *Physica A: Statistical Mechanics and its Applications*, vol. 316, no. 1–4, pp. 87–114, 2002.  
[4] D. A. Torres, L. A. Da Silva, and M. A. Batista, “Detection of voice pathology using fractal dimension in a multiresolution analysis,” *Journal of Medical Systems*, vol. 39, no. 12, pp. 1–9, 2015.  
[5] P. Abásolo et al., “Nonlinear analysis of voice signals: Application to pathological voices,” *Computers in Biology and Medicine*, vol. 40, no. 2, pp. 132–139, 2010.
