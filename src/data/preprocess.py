import os
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import numpy as np

def resample_and_normalize(audio_dir, target_sr=16000):
    """
    Resamples all .wav files in the directory to the target sampling rate using polyphase filtering
    and performs peak normalization to [-1, 1].
    Overwrites the original files.
    """
    print(f"Scanning {audio_dir} for resampling to {target_sr} Hz and normalization...")
    
    files = list(Path(audio_dir).glob("*.wav"))
    count = 0
    
    for file_path in files:
        try:
            # Read metadata first to check SR without loading everything? 
            # wavfile.read loads everything. simpler to just load.
            sr, data = wavfile.read(file_path)
            
            # Resample if needed
            if sr != target_sr:
                import math
                gcd = math.gcd(sr, target_sr)
                up = target_sr // gcd
                down = sr // gcd
                
                # Convert to float for processing
                data = data.astype(np.float32)
                
                # Resample
                data = signal.resample_poly(data, up, down)
            else:
                # Ensure float32 for normalization
                data = data.astype(np.float32)
            
            # Peak Normalization
            # x_norm = x / max(|x|)
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
            
            # Save as float32
            wavfile.write(file_path, target_sr, data)
            
            print(f"Processed {file_path.name}: {sr} -> {target_sr} Hz, Normalized (Float32)")
            count += 1
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    print(f"\nProcessing complete. Processed {count} files.")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    audio_dir = base_dir / "data" / "processed" / "audio"
    
    if audio_dir.exists():
        resample_and_normalize(audio_dir)
    else:
        print(f"Audio directory not found: {audio_dir}")
