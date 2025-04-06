#!/usr/bin/env python3
"""
Evaluation script for SepFormer with WavLM speaker classification.
This script loads a trained model and evaluates it on test data.

Usage:
    python evaluate.py hparams/sepformer-wavlm-speaker.yaml --test_data /path/to/test.csv
"""

import os
import sys
import torch
import logging
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from train import SeparationAndClassification, dataio_prep

logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Check for test_data in overrides
    if "--test_data" not in " ".join(sys.argv):
        print("Error: You must specify --test_data /path/to/test.csv")
        sys.exit(1)
        
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create output directory if it doesn't exist
    os.makedirs(hparams["output_folder"], exist_ok=True)
    
    # Initialize ddp
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Load test data
    if not os.path.exists(hparams["test_data"]):
        print(f"Error: Test data file {hparams['test_data']} not found")
        sys.exit(1)
        
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    
    # Add dynamic items to test data
    sb.dataio.dataset.add_dynamic_item(
        [test_data], 
        lambda x: sb.dataio.dataio.read_audio(x["mix_wav"]), 
        "mix_sig"
    )
    
    sb.dataio.dataset.add_dynamic_item(
        [test_data], 
        lambda x: sb.dataio.dataio.read_audio(x["s1_wav"]), 
        "s1_sig"
    )
    
    sb.dataio.dataset.add_dynamic_item(
        [test_data], 
        lambda x: sb.dataio.dataio.read_audio(x["s2_wav"]), 
        "s2_sig"
    )
    
    # Add class labels if classifier is used
    if hparams.get("use_classifier", False):
        @sb.utils.data_pipeline.takes("s1_class", "s2_class")
        @sb.utils.data_pipeline.provides("class_labels")
        def class_pipeline(s1_class, s2_class):
            """Convert speaker class labels to tensor."""
            try:
                # Convert to integers, handling different input types
                def safe_parse_class(cls_val):
                    if isinstance(cls_val, str):
                        try:
                            return int(float(cls_val))
                        except ValueError:
                            return -1
                    elif isinstance(cls_val, (int, float)):
                        return int(cls_val)
                    else:
                        return -1
                
                s1_label = safe_parse_class(s1_class)
                s2_label = safe_parse_class(s2_class)
                
                return torch.tensor([s1_label, s2_label])
            
            except Exception as e:
                print(f"Error processing class labels: {e}")
                return torch.tensor([-1, -1])

        sb.dataio.dataset.add_dynamic_item([test_data], class_pipeline)
        
        # Set output keys with class labels
        sb.dataio.dataset.set_output_keys(
            [test_data], ["id", "mix_sig", "s1_sig", "s2_sig", "class_labels"]
        )
    else:
        # Set output keys without class labels
        sb.dataio.dataset.set_output_keys(
            [test_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
        )
    
    # Initialize the Brain class
    separator = SeparationAndClassification(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # Load latest checkpoint (automatically done by SpeechBrain's checkpointer)
    if not separator.checkpointer.find_checkpoint():
        print("No checkpoint found. Please train the model first.")
        sys.exit(1)
        
    print(f"Loaded checkpoint from {separator.checkpointer.checkpoints_dir}")
    
    # Run evaluation
    separator.evaluate(test_data, min_key="si-snr")
    
    # Save detailed test results to CSV
    separator.save_results(test_data)
    
    print("Evaluation completed. Results saved to:", 
          os.path.join(hparams["output_folder"], "test_results.csv"))

if __name__ == "__main__":
    main()
