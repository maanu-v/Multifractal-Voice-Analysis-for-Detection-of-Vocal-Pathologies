import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
import random

def check_signal_stats(audio_dir, labels_file, num_samples=3):
    """
    Performs basic sanity checks on the audio dataset.
    """
    base_dir = Path(__file__).parent.parent.parent
    figures_dir = base_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    df = pd.read_csv(labels_file)
    print(f"Loaded {len(df)} entries from {labels_file}")
    
    # Check 10: Speaker Uniqueness
    unique_speakers = df['speaker_id'].nunique()
    total_files = len(df)
    print(f"\n--- Speaker Uniqueness Check ---")
    print(f"Total Files: {total_files}")
    print(f"Unique Speakers: {unique_speakers}")
    if unique_speakers == total_files:
        print("✅ PASS: One file per speaker.")
    else:
        print(f"❌ FAIL: {total_files - unique_speakers} duplicate speakers found!")

    # Check 9: Class Balance
    print(f"\n--- Class Balance ---")
    print(df['category'].value_counts())
    
    # Sample files for detailed inspection
    categories = df['category'].unique()
    
    print(f"\n--- Detailed Signal Inspection (Sample of {num_samples} per class) ---")
    
    for category in categories:
        print(f"\nCategory: {category}")
        subset = df[df['category'] == category]
        samples = subset.sample(min(num_samples, len(subset)))
        
        for _, row in samples.iterrows():
            filename = row['filename']
            file_path = audio_dir / filename
            
            try:
                sample_rate, data = wavfile.read(file_path)
                
                # Check 1: Sampling Rate
                # Check 2: Bit Depth / Dtype
                # Check 3: Duration
                duration = len(data) / sample_rate
                
                # Check 6: Amplitude Stats
                min_val = np.min(data)
                max_val = np.max(data)
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                # Check 7: Silence Proportion (Simple energy threshold)
                # Normalize to 0-1 for energy calc
                if data.dtype == np.int16:
                    norm_data = data / 32768.0
                else:
                    norm_data = data
                
                energy = norm_data ** 2
                silence_thresh = 0.001 # Arbitrary low energy threshold
                silence_prop = np.mean(energy < silence_thresh)
                
                print(f"  File: {filename}")
                print(f"    SR: {sample_rate} Hz, Type: {data.dtype}")
                print(f"    Duration: {duration:.2f}s")
                print(f"    Range: [{min_val}, {max_val}], Mean: {mean_val:.2f}, Std: {std_val:.2f}")
                print(f"    Silence Prop: {silence_prop:.2%}")
                
                # Check 4 & 8: Plots
                plt.figure(figsize=(10, 4))
                
                # Waveform
                plt.subplot(1, 2, 1)
                time = np.linspace(0, duration, len(data))
                plt.plot(time, data, alpha=0.7)
                plt.title(f"Waveform: {filename}")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                
                # Spectrogram
                plt.subplot(1, 2, 2)
                plt.specgram(data, NFFT=1024, Fs=sample_rate, noverlap=512)
                plt.title(f"Spectrogram: {filename}")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                
                plt.tight_layout()
                plot_filename = figures_dir / f"analysis_{category}_{filename}.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"    Plot saved to: {plot_filename}")
                
            except Exception as e:
                print(f"    Error analyzing {filename}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    audio_dir = base_dir / "data" / "processed" / "audio"
    labels_file = base_dir / "data" / "processed" / "labels.csv"
    
    check_signal_stats(audio_dir, labels_file)
