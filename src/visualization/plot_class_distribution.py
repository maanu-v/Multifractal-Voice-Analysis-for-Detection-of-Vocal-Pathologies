import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Plotting Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)

# Configuration
INPUT_FILE = 'data/processed/features/fd_mfdfa_features.csv'
OUTPUT_DIR = 'reports/data_distribution'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and preprocess data."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    df = pd.read_csv(INPUT_FILE)
    if 'category' in df.columns:
        df['category'] = df['category'].str.capitalize()
    return df

def plot_class_distribution_pie(df):
    """Plot Class Distribution as a Pie Chart."""
    class_counts = df['category'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        class_counts, 
        labels=class_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("pastel"),
        textprops={'fontsize': 14}
    )
    plt.title('Pathology Class Distribution', fontsize=16)
    
    output_path = os.path.join(OUTPUT_DIR, 'class_distribution_pie.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def plot_class_distribution_bar(df):
    """Plot Class Distribution as a Bar Chart."""
    plt.figure(figsize=(8, 6))
    
    ax = sns.countplot(
        x='category', 
        data=df, 
        hue='category',
        legend=False,
        palette="viridis",
        order=df['category'].value_counts().index
    )
    
    plt.title('Number of Samples per Pathology Class', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'class_distribution_bar.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print("Generating Class Distribution plots...")
    try:
        df = load_data()
        plot_class_distribution_pie(df)
        plot_class_distribution_bar(df)
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
