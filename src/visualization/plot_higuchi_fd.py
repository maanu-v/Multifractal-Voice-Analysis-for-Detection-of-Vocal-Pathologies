
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_higuchi_fd(input_path, output_dir):
    """
    Generates a violin plot for Higuchi FD values across voice categories.
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

    # Capitalize category names for better display
    df['category'] = df['category'].str.capitalize()

    # Define order if needed, but alphabetical is usually fine or custom
    # category_order = ['Healthy', 'Structural', 'Neurological']

    plt.figure(figsize=(10, 6))
    
    # Create violin plot
    # "inner='box'" draws a miniature boxplot inside the violin
    sns.violinplot(x='category', y='FD_full', data=df, palette="muted", inner="box")
    
    plt.title('Distribution of Higuchi Fractal Dimension (FD) by Voice Category', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Higuchi FD', fontsize=12)
    
    output_file = os.path.join(output_dir, 'higuchi_fd_violin.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    input_csv = "data/processed/features/higuchi_fd_features.csv"
    output_directory = "reports/plot/fd"
    
    plot_higuchi_fd(input_csv, output_directory)
