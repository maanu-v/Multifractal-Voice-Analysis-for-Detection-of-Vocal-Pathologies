import os
import shutil
import csv
import nspfile
from scipy.io import wavfile
from collections import Counter
from pathlib import Path

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    raw_data_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    audio_output_dir = processed_dir / "audio"
    labels_file = processed_dir / "labels.csv"

    # Create output directories
    if audio_output_dir.exists():
        shutil.rmtree(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    extracted_data = []

    print(f"Scanning {raw_data_dir}...")

    # Traverse the directory structure
    for category_path in raw_data_dir.iterdir():
        if not category_path.is_dir():
            continue
        
        category = category_path.name
        
        # Handle 'healthy' category which has a different structure:
        # data/raw/healthy/<SpeakerID>/vowels/<filename>
        if category == "healthy":
            pathology = "healthy" # Use 'healthy' as pathology name for healthy voices
            for speaker_path in category_path.iterdir():
                if not speaker_path.is_dir():
                    continue
                
                speaker_id = speaker_path.name
                vowels_path = speaker_path / "vowels"
                
                if not vowels_path.exists():
                    continue
                
                # Filter for *-a_n.nsp files
                for file_path in vowels_path.glob("*-a_n.nsp"):
                    filename_stem = file_path.stem
                    filename = f"{filename_stem}.wav"
                    target_path = audio_output_dir / filename
                    
                    try:
                        # Read NSP file
                        sample_rate, data = nspfile.read(str(file_path))
                        
                        # Write WAV file
                        wavfile.write(target_path, sample_rate, data)
                        
                        # Append metadata
                        extracted_data.append({
                            "filename": filename,
                            "category": category,
                            "pathology": pathology,
                            "speaker_id": speaker_id
                        })
                        print(f"Converted: {file_path.name} -> {filename} ({category} / {pathology})")
                    except Exception as e:
                        print(f"Error converting {file_path.name}: {e}")
    
        # Handle other categories (structural, neurological)
        # data/raw/<Category>/<Pathology>/<SpeakerID>/vowels/<filename>
        else:
            for pathology_path in category_path.iterdir():
                if not pathology_path.is_dir():
                    continue
                
                pathology = pathology_path.name
                
                for speaker_path in pathology_path.iterdir():
                    if not speaker_path.is_dir():
                        continue
                    
                    speaker_id = speaker_path.name
                    vowels_path = speaker_path / "vowels"
                    
                    if not vowels_path.exists():
                        continue
                    
                    # Filter for *-a_n.nsp files
                    for file_path in vowels_path.glob("*-a_n.nsp"):
                        filename_stem = file_path.stem
                        filename = f"{filename_stem}.wav"
                        target_path = audio_output_dir / filename
                        
                        try:
                            # Read NSP file
                            sample_rate, data = nspfile.read(str(file_path))
                            
                            # Write WAV file
                            wavfile.write(target_path, sample_rate, data)
                            
                            # Append metadata
                            extracted_data.append({
                                "filename": filename,
                                "category": category,
                                "pathology": pathology,
                                "speaker_id": speaker_id
                            })
                            print(f"Converted: {file_path.name} -> {filename} ({category} / {pathology})")
                        except Exception as e:
                            print(f"Error converting {file_path.name}: {e}")

    # Write labels.csv
    if extracted_data:
        try:
            with open(labels_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "category", "pathology", "speaker_id"])
                writer.writeheader()
                writer.writerows(extracted_data)
            print(f"\nSuccessfully processed {len(extracted_data)} files.")
            print(f"Audio files saved to: {audio_output_dir}")
            print(f"Labels saved to: {labels_file}")

            # Print statistics
            print("\n--- Extraction Statistics ---")
            category_counts = Counter(item['category'] for item in extracted_data)
            for category, count in category_counts.items():
                print(f"  - {category}: {count} files")
            print(f"Total: {len(extracted_data)} files")
        except IOError as e:
            print(f"Error writing csv file: {e}")
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()
