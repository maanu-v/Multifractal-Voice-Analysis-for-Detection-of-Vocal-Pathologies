
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import os
from pathlib import Path
from src.features.mfdfa import mfdfa

# -------------------------------------------------------------
# 1. Helper for Singularity Spectrum Computation
# -------------------------------------------------------------
def compute_singularity_spectrum(signal, q_vals, scales):
    """
    Computes alpha and f(alpha) from the signal using MF-DFA.
    """
    hq_dict = mfdfa(signal, q_vals, scales, m=1)
    
    # Extract h(q) values
    h_arr = np.array([hq_dict[q] for q in q_vals])
    q_arr = np.array(q_vals)
    
    # Handle NaNs
    if np.any(np.isnan(h_arr)):
        return None, None

    # Calculate tau(q) = q * h(q) - 1
    tau = q_arr * h_arr - 1
    
    # Calculate alpha = d(tau)/dq
    # Use numerical differentiation
    alpha = np.gradient(tau, q_arr)
    
    # Calculate f(alpha) = q * alpha - tau
    f_alpha = q_arr * alpha - tau
    
    return alpha, f_alpha

# -------------------------------------------------------------
# 2. Main Plotting Function
# -------------------------------------------------------------
def generate_mfdfa_plots():
    # Setup directories
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    features_csv = "data/processed/features/mfdfa_features.csv"
    if not os.path.exists(features_csv):
        print("Error: Features CSV not found.")
        return
        
    df = pd.read_csv(features_csv)
    df['category'] = df['category'].str.capitalize()
    
    # =========================================================
    # PLOT 1: Violin Plot of Delta Alpha
    # =========================================================
    print("Generating Delta Alpha Violin Plot...")
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df,
        x="category",
        y="delta_alpha",
        inner="box",
        palette="Set2"
    )

    plt.title("Distribution of Singularity Spectrum Width ($\\Delta \\alpha$) by Voice Category")
    plt.xlabel("Voice Category")
    plt.ylabel("Singularity Spectrum Width ($\\Delta \\alpha$)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    output_violin = os.path.join(output_dir, "mfdfa_delta_alpha_violin.png")
    plt.savefig(output_violin, dpi=300)
    print(f"Saved: {output_violin}")
    plt.close()

    # =========================================================
    # PLOT 2: Singularity Spectrum f(alpha)
    # =========================================================
    print("Generating Singularity Spectrum Plot...")
    
    # Define representative files (Update paths as needed)
    audio_dir = Path("data/processed/audio")
    files = {
        'Healthy': str(audio_dir / "1046-a_n.wav"), 
        'Structural': str(audio_dir / "1052-a_n.wav"),
        'Neurological': str(audio_dir / "1749-a_n.wav")
    }
    
    q_vals = np.linspace(-5, 5, 101).tolist()
    scales = [16, 32, 64, 128, 256, 512]
    
    plt.figure(figsize=(7, 5))
    
    colors = {'Healthy': 'green', 'Structural': 'blue', 'Neurological': 'red'}
    
    for label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            # Fallback logic could go here
            continue
            
        try:
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            y, _ = librosa.effects.trim(y)
            
            alpha, f_alpha = compute_singularity_spectrum(y, q_vals, scales)
            
            if alpha is not None:
                plt.plot(alpha, f_alpha, label=label, linewidth=2, color=colors.get(label))
                
        except Exception as e:
            print(f"Error processing {label}: {e}")

    plt.xlabel("Singularity Strength $\\alpha$")
    plt.ylabel("Multifractal Spectrum $f(\\alpha)$")
    plt.title("Representative Multifractal Singularity Spectra")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    output_spectrum = os.path.join(output_dir, "mfdfa_singularity_spectrum.png")
    plt.savefig(output_spectrum, dpi=300)
    print(f"Saved: {output_spectrum}")
    plt.close()

    # =========================================================
    # PLOT 3: h(q) Comparison (Small vs Large Fluctuations)
    # =========================================================
    print("Generating h(q) Comparison Plot...")
    
    # Check if columns exist
    if "h_q_neg5" in df.columns and "h_q_5" in df.columns:
        plt.figure(figsize=(9, 5))

        df_long = df.melt(
            id_vars=["category"],
            value_vars=["h_q_neg5", "h_q_5"],
            var_name="Fluctuation Type",
            value_name="Hurst Exponent h(q)"
        )

        df_long["Fluctuation Type"] = df_long["Fluctuation Type"].map({
            "h_q_neg5": "Small Fluctuations (q = -5)",
            "h_q_5": "Large Fluctuations (q = +5)"
        })

        sns.boxplot(
            data=df_long,
            x="category",
            y="Hurst Exponent h(q)",
            hue="Fluctuation Type",
            palette="Set2"
        )

        plt.title("Hurst Exponents for Small vs Large Fluctuations")
        plt.xlabel("Voice Category")
        plt.ylabel("Hurst Exponent h(q)")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        output_hq = os.path.join(output_dir, "mfdfa_hq_comparison.png")
        plt.savefig(output_hq, dpi=300)
        print(f"Saved: {output_hq}")
        plt.close()
    else:
        print("Columns h_q_neg5 or h_q_5 not found, skipping h(q) plot.")

if __name__ == "__main__":
    generate_mfdfa_plots()
