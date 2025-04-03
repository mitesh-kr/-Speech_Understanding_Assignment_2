import os
import csv
import argparse
import pandas as pd
import random
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

def mix_two_sources(src1, src2):
    # Truncate both sources to the length of the shorter one and sum them.
    min_len = min(len(src1), len(src2))
    return src1[:min_len] + src2[:min_len]

def process_mixture(row1, row2, out_dir, mix_id, target_sr):
    # Read the two audio files.
    audio1, sr1 = sf.read(row1['filepath'], dtype='float32')
    audio2, sr2 = sf.read(row2['filepath'], dtype='float32')
    
    # Resample if needed.
    if sr1 != target_sr:
        audio1 = resample_poly(audio1, target_sr, sr1)
    if sr2 != target_sr:
        audio2 = resample_poly(audio2, target_sr, sr2)
    
    # Mix the two sources.
    mixture = mix_two_sources(audio1, audio2)
    
    # Create a unique filename using the full path (folder + filename)
    speaker1_path = row1['filepath']
    speaker2_path = row2['filepath']
    
    # Extract folder and file parts (excluding file extension)
    speaker1_folder = os.path.basename(os.path.dirname(speaker1_path))
    speaker2_folder = os.path.basename(os.path.dirname(speaker2_path))
    
    speaker1_filename = os.path.splitext(os.path.basename(speaker1_path))[0]
    speaker2_filename = os.path.splitext(os.path.basename(speaker2_path))[0]
    
    # Construct the output file name in the desired format
    mix_filename = f"{row1['speaker']}_{speaker1_folder}_{speaker1_filename}_{row2['speaker']}_{speaker2_folder}_{speaker2_filename}_{mix_id}.wav"
    
    # Save the mixture
    out_path = os.path.join(out_dir, mix_filename)
    sf.write(out_path, mixture, target_sr)
    return out_path

def main(metadata_file, out_dir, target_sr, n_mix, M_per_pair):
    # Load the metadata.
    md = pd.read_csv(metadata_file)
    os.makedirs(out_dir, exist_ok=True)
    
    speakers = md['speaker'].unique()
    mix_count = 0
    
    # Create mixtures by pairing utterances from two different speakers.
    for i, spk1 in enumerate(speakers):
        for spk2 in speakers[i+1:]:
            srcs1 = md[md['speaker'] == spk1]
            srcs2 = md[md['speaker'] == spk2]
            if len(srcs1) > 0 and len(srcs2) > 0:
                # Generate M_per_pair mixtures for this speaker pair.
                for m in range(M_per_pair):
                    if mix_count >= n_mix:
                        print("Created", mix_count, "mixtures.")
                        return
                    row1 = srcs1.sample(n=1).iloc[0]
                    row2 = srcs2.sample(n=1).iloc[0]
                    # Generate a unique mix ID (optional)
                    mix_id = f"{mix_count}"
                    process_mixture(row1, row2, out_dir, mix_id, target_sr)
                    mix_count += 1
                    if mix_count % 100 == 0:
                        print(f"Created {mix_count} mixtures.")
    print("Created", mix_count, "mixtures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_file', type=str, required=True,
                        help='Path to the VoxCeleb2 metadata CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the mixtures')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='Target sampling rate for mixtures')
    parser.add_argument('--n_mix', type=int, default=1000,
                        help='Number of mixtures to create')
    parser.add_argument('--M_per_pair', type=int, default=2,
                        help='Number of mixtures to create per speaker pair')
    args = parser.parse_args()
    main(args.metadata_file, args.output_dir, args.target_sr, args.n_mix, args.M_per_pair)
