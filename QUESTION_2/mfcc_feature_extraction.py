import os
import librosa
import numpy as np

dataset_path = '/scratch/data/m23mac004/audio_dataset/Language Detection Dataset'

# Loop through each item in the dataset_path directory
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        print(f"{folder_name}: {len(files)} files")




def load_audio_fixed_length(file_path, sr=16000, duration=5):

    desired_length = sr * duration
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    
    if len(y) > desired_length:
        # Truncate the audio signal to the desired length
        y = y[:desired_length]
    elif len(y) < desired_length:
        # Pad the audio signal with zeros to reach the desired length
        y = np.pad(y, (0, desired_length - len(y)), mode='constant')
    
    return y

def extract_mfcc_fixed(file_path, sr=16000, duration=5, n_mfcc=13):
    y = load_audio_fixed_length(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

# Define the input and output directories
ROOT_DIR = dataset_path
OUTPUT_DIR = "/scratch/data/m23mac004/audio_dataset/mfcc_feature"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Iterate through each language folder in the dataset
for language in os.listdir(ROOT_DIR):
    language_path = os.path.join(ROOT_DIR, language)
    
    if os.path.isdir(language_path):
        print(f"Processing language: {language}")
        # Create a corresponding folder to save MFCCs for this language
        language_output_dir = os.path.join(OUTPUT_DIR, language)
        os.makedirs(language_output_dir, exist_ok=True)
        
        # Process each audio file in the language folder
        for filename in os.listdir(language_path):
            if filename.lower().endswith(".mp3") or filename.lower().endswith(".wav"):
                file_path = os.path.join(language_path, filename)
                try:
                    # Extract MFCC features from a 5-second fixed audio
                    mfcc = extract_mfcc_fixed(file_path, sr=16000, duration=5, n_mfcc=13)
                    
                    # Create an output file name (e.g., "0.npy" for "0.mp3")
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(language_output_dir, base_name + ".npy")
                    
                    # Save the MFCC array to disk
                    np.save(output_file, mfcc)
                    # print(f"  - Processed {filename}: MFCC shape = {mfcc.shape} | Saved to {output_file}")
                
                except Exception as e:
                    print(f"  - Error processing {filename}: {e}")

print("MFCC extraction and saving complete.")
