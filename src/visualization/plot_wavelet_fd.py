
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_wavelet_fd(input_path, output_dir):
    """
    Generates a grouped boxplot for Wavelet FD features (A3, D3, D2) across voice categories.
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

    # Capitalize category names
    df['category'] = df['category'].str.capitalize()

    # Features to compare
    features = ['FD_A3', 'FD_D3', 'FD_D2']
    
    # Melt the dataframe for easy plotting with seaborn
    # We want rows like: category | feature_type | value
    df_melted = df.melt(id_vars=['category'], value_vars=features, 
                        var_name='Wavelet Component', value_name='Fractal Dimension')

    plt.figure(figsize=(12, 7))
    
    # Create grouped boxplot
    sns.boxplot(x='category', y='Fractal Dimension', hue='Wavelet Component', 
                data=df_melted, palette="Set2")
    
    plt.title('Comparison of Wavelet Fractal Dimensions (A3, D3, D2) across Categories', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Fractal Dimension (FD)', fontsize=12)
    plt.legend(title='Wavelet Level')
    
    output_file = os.path.join(output_dir, 'wavelet_fd_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    input_csv = "data/processed/features/wavelet_fd_features.csv"
    output_directory = "reports/plot/fd"
    
    plot_wavelet_fd(input_csv, output_directory)
