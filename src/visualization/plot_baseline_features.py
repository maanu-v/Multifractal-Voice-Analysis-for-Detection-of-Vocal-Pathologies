
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def plot_baseline_features(input_path, output_dir):
    """
    Generates boxplots for Jitter, Shimmer, and HNR across voice categories.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter out any undefined categories if necessary
    # df = df[df['category'].isin(['healthy', 'structural', 'neurological'])]
    
    # Capitalize category names for better modification
    df['category'] = df['category'].str.capitalize()

    features = [
        ('jitter_local', 'Jitter (local)', 'jitter_boxplot.png'),
        ('shimmer_local', 'Shimmer (local)', 'shimmer_boxplot.png'),
        ('hnr', 'Harmonics-to-Noise Ratio (HNR)', 'hnr_boxplot.png')
    ]

    sns.set_theme(style="whitegrid")

    for col, title, filename in features:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataset. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        
        # Create boxplot
        sns.boxplot(x='category', y=col, data=df, palette="Set2")
        
        plt.title(f'Distribution of {title} by Voice Category', fontsize=16)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel(title, fontsize=12)
        
        output_file = os.path.join(output_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

if __name__ == "__main__":
    input_csv = "data/processed/features/classic_features.csv"
    output_directory = "reports/plot/baseline"
    
    plot_baseline_features(input_csv, output_directory)
