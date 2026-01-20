
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Plotting Style for Publication
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)

# Configuration
INPUT_FILE = 'data/processed/features/fd_mfdfa_features.csv'
OUTPUT_DIR = 'reports/plot/mfdfa'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and preprocess data."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    
    # Capitalize category names for better labeling
    if 'category' in df.columns:
        df['category'] = df['category'].str.capitalize()
    
    return df

def plot_delta_alpha_violin(df):
    """1) Delta alpha (singularity spectrum width) - Box/Violin Plot."""
    plt.figure(figsize=(8, 6))
    
    # Violin plot with inner boxplot
    ax = sns.violinplot(
        data=df, 
        x='category', 
        y='delta_alpha', 
        hue='category',
        palette="viridis", 
        inner="box",
        legend=False
    )
    
    plt.title(r'Singularity Spectrum Width ($\Delta\alpha$) by Class')
    plt.xlabel('Pathology Class')
    plt.ylabel(r'Singularity Width $\Delta\alpha$')
    
    # Annotate: Healthy -> Narrower, Pathological -> Wider
    # Determine basic stats for positioning text (optional, or just title is enough)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'delta_alpha_violin.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_hq_fluctuations(df):
    """2) h(q) at two extremes (small vs large fluctuations) - Side-by-side Boxplots."""
    # Melt dataframe to long format for h(q=-5) and h(q=5)
    df_melted = df.melt(
        id_vars=['category'], 
        value_vars=['h_q_neg5', 'h_q_5'], 
        var_name='q_type', 
        value_name='h_q'
    )
    
    # Map column names to readable labels
    df_melted['q_label'] = df_melted['q_type'].map({
        'h_q_neg5': 'h(q = -5)\nSmall Fluctuations', 
        'h_q_5': 'h(q = +5)\nLarge Fluctuations'
    })
    
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=df_melted,
        x='category',
        y='h_q',
        hue='q_label',
        palette="RdBu_r"
    )
    
    plt.title('Hurst Exponent $h(q)$ for Small vs. Large Fluctuations')
    plt.xlabel('Pathology Class')
    plt.ylabel('Hurst Exponent $h(q)$')
    plt.legend(title='Fluctuation Scale')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hq_fluctuations_box.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_delta_alpha_vs_fd_scatter(df):
    """3) Multi-fractal (Delta alpha) vs Mono-fractal (FD) Scatter Plot."""
    if 'FD_full' not in df.columns:
        print("Warning: 'FD_full' column not found. Skipping scatter plot.")
        return

    plt.figure(figsize=(9, 7))
    
    sns.scatterplot(
        data=df,
        x='FD_full',
        y='delta_alpha',
        hue='category',
        style='category',
        s=80, # Marker size
        alpha=0.8,
        palette="deep"
    )
    
    plt.title(r'Complementarity: Multi-fractal ($\Delta\alpha$) vs. Mono-fractal (FD)')
    plt.xlabel('Fractal Dimension (FD)')
    plt.ylabel(r'Singularity Width $\Delta\alpha$')
    plt.legend(title='Category', loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'delta_alpha_vs_fd_scatter.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("Generating MF-DFA plots...")
    try:
        df = load_data()
        plot_delta_alpha_violin(df)
        plot_hq_fluctuations(df)
        plot_delta_alpha_vs_fd_scatter(df)
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
