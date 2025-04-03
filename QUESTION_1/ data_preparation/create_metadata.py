#!/usr/bin/env python3
import os
import csv
import argparse

def main(vox2_dir, out_dir):
    # List all speaker directories that start with "id" and sort them.
    all_ids = sorted([d for d in os.listdir(vox2_dir)
                      if d.startswith("id") and os.path.isdir(os.path.join(vox2_dir, d))])
    
    if len(all_ids) < 100:
        raise ValueError("Expected at least 100 identities in the VoxCeleb2 dataset.")
    
    # First 50 for training, next 50 for testing.
    train_ids = all_ids[:50]
    test_ids  = all_ids[50:100]
    
    # Define metadata file paths.
    train_md_path = os.path.join(out_dir, "train_voxceleb2_metadata.csv")
    test_md_path  = os.path.join(out_dir, "test_voxceleb2_metadata.csv")
    
    # Write training metadata.
    with open(train_md_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["speaker", "filepath"])
        for spk in train_ids:
            spk_path = os.path.join(vox2_dir, spk)
            for root, _, files in os.walk(spk_path):
                for f in files:
                    if f.endswith(".wav"):
                        writer.writerow([spk, os.path.join(root, f)])
                        
    # Write testing metadata.
    with open(test_md_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["speaker", "filepath"])
        for spk in test_ids:
            spk_path = os.path.join(vox2_dir, spk)
            for root, _, files in os.walk(spk_path):
                for f in files:
                    if f.endswith(".wav"):
                        writer.writerow([spk, os.path.join(root, f)])
                        
    print("Metadata files created:")
    print("  Training:", train_md_path)
    print("  Testing :", test_md_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vox2_dir', type=str, required=True,
                        help='Path to the VoxCeleb2 root directory')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory to save metadata files')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args.vox2_dir, args.out_dir)
