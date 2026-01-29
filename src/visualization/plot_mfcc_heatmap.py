
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_mfcc_heatmap(input_path, output_dir):
    """
    Generates a heatmap of MFCC coefficients for representative samples.
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

    # Select representative samples
    # Strategy: Select the first sample from each key pathology/category
    # We want: Healthy, Structural (e.g., Polyp), Neurological (e.g., Parkinson)
    
    categories_of_interest = {
        'healthy': 'Healthy',
        'structural': 'Structural', 
        'neurological': 'Neurological'
    }
    
    samples = []
    sample_labels = []

    # Try to pick specific pathologies if possible for better representation
    target_pathologies = {
        'healthy': None, # Any healthy
        'structural': ['Stimmlippenpolyp', 'Phonationsknötchen'],
        'neurological': ['Morbus Parkinson', 'Rekurrensparese']
    }

    for cat_key, cat_label in categories_of_interest.items():
        subset = df[df['category'] == cat_key]
        
        if subset.empty:
            continue
            
        # Try to find specific pathology
        target = target_pathologies.get(cat_key)
        
        selected_row = None
        if target:
            for path_name in target:
                path_subset = subset[subset['pathology'] == path_name]
                if not path_subset.empty:
                    selected_row = path_subset.iloc[0]
                    label = f"{cat_label} ({path_name})"
                    break
        
        if selected_row is None:
            # Fallback to first available in category
            selected_row = subset.iloc[0]
            label = f"{cat_label}"
            
        samples.append(selected_row)
        sample_labels.append(label)

    if not samples:
        print("No samples found for heatmap.")
        return

    # Extract MFCC means (mfcc1_mean to mfcc12_mean)
    mfcc_cols = [f'mfcc{i}_mean' for i in range(1, 13)]
    
    data_matrix = []
    valid_labels = []
    
    for row, label in zip(samples, sample_labels):
        # check if all columns exist
        if all(col in row for col in mfcc_cols):
             data_matrix.append([row[col] for col in mfcc_cols])
             valid_labels.append(label)
    
    if not data_matrix:
        print("MFCC columns missing.")
        return

    data_matrix = np.array(data_matrix)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"MFCC-{i}" for i in range(1, 13)],
                yticklabels=valid_labels)
    
    plt.title('MFCC Feature Heatmap for Representative Samples', fontsize=16)
    plt.xlabel('MFCC Coefficients', fontsize=12)
    plt.ylabel('Voice Category / Pathology', fontsize=12)
    plt.xticks(rotation=45)
    
    output_file = os.path.join(output_dir, 'mfcc_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

if __name__ == "__main__":
    input_csv = "data/processed/features/mfcc_features.csv"
    output_directory = "reports/plot/baseline"
    
    plot_mfcc_heatmap(input_csv, output_directory)
