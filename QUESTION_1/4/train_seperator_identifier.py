

#!/usr/bin/env/python3
"""
Modified SepFormer train script to integrate with WavLM classifier.
This script adds speaker classification capabilities to the
SepFormer speech separation system.

To run this recipe, do the following:
> python train.py hparams/sepformer-wavlm-speaker.yaml --data_folder /your_path/whamr
"""
# print("CHEKING PRINT STATEMENT")  # Removed print statement

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import numpy as np
import csv
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from contextlib import nullcontext

def create_csv_logger(csv_file):
    """Creates a CSV logger to store all metrics during training with enhanced error handling."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Create CSV file with headers
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 
            'stage',
            'loss', 
            'si-snr', 
            'class_loss', 
            'accuracy',
            'lr'
        ])
    
    def log_metrics(epoch, stage, metrics, lr=None):
        """Log metrics to the CSV file with robust error handling."""
        try:
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # Safe metric extraction
                def safe_extract(metrics_dict, key, default=None):
                    # Handle different input types
                    if not isinstance(metrics_dict, dict):
                        return default
                    
                    value = metrics_dict.get(key, default)
                    
                    # Convert torch tensors to scalars
                    if isinstance(value, torch.Tensor):
                        try:
                            value = value.item()
                        except:
                            value = str(value)
                    
                    return value if value is not None else default
                
                # Extract metrics
                loss = safe_extract(metrics, 'loss')
                si_snr = safe_extract(metrics, 'si-snr')
                class_loss = safe_extract(metrics, 'class_loss')
                accuracy = safe_extract(metrics, 'accuracy')
                
                # Write row to CSV
                writer.writerow([
                    epoch,
                    stage,
                    loss,
                    si_snr,
                    class_loss,
                    accuracy,
                    lr
                ])
                
                # Optional: print logged metrics for debugging
                # print(f"Logged metrics: Epoch {epoch}, Stage {stage}")
                # print(f"  Loss: {loss}")
                # print(f"  SI-SNR: {si_snr}")
                # print(f"  Class Loss: {class_loss}")
                # print(f"  Accuracy: {accuracy}")
                # print(f"  Learning Rate: {lr}")
        
        except Exception as e:
            # print(f"Error logging metrics to {csv_file}: {e}")
            # Optionally log to an error file
            with open(csv_file.replace('.csv', '_error.log'), 'a') as error_log:
                error_log.write(f"Logging error at Epoch {epoch}, Stage {stage}: {str(e)}\n")
    
    return log_metrics


class SeparationAndClassification(sb.Brain):
    def __init__(self, *args,num_spks=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_ctx = nullcontext() 
        self.hparams.num_spks = num_spks

    def on_train_start(self, *args, **kwargs):
        super().on_train_start(*args, **kwargs)
        self.training_ctx = self.auto_mix_prec

    def compute_forward(self, mix, targets, stage, noise=None, target_classes=None):
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        speaker_signals = [targets[i][0] for i in range(self.hparams.num_spks)]
        max_len = max([sig.shape[-1] for sig in speaker_signals])
        padded_signals = [F.pad(sig, (0, max_len - sig.shape[-1])) for sig in speaker_signals]
        targets = torch.stack(padded_signals, dim=-1).to(self.device)

        # Add speech distortions (same as the original code)
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    if "whamr" in self.hparams.data_folder and hasattr(self.hparams, "reverb"):
                        targets_rev = [
                            self.hparams.reverb(targets[:, :, i], None)
                            for i in range(self.hparams.num_spks)
                        ]
                        targets_rev = torch.stack(targets_rev, dim=-1)
                        mix = targets_rev.sum(-1)
                    else:
                        mix = targets.sum(-1)

                    noise = noise.to(self.device)
                    len_noise = noise.shape[1]
                    len_mix = mix.shape[1]
                    min_len = min(len_noise, len_mix)

                    # add the noise
                    mix = mix[:, :min_len] + noise[:, :min_len]

                    # fix the length of targets also
                    targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.drop_chunk(mix, mix_lens)
                    mix = self.hparams.drop_freq(mix)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)


        # Generate the separation masks and apply them
        mix_w = self.modules.encoder(mix)
        est_mask = self.modules.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask
        est_source = torch.cat(
            [
                self.modules.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )
        
        # Adjust length if needed
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        # Now perform classification if enabled
        if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier and \
        (stage != sb.Stage.TRAIN or self.hparams.classify_during_training):
            class_logits = []
            embeddings = []
            
            # Extract data from PaddedData if necessary
            if target_classes is not None:
                if hasattr(target_classes, 'data'):
                    speaker_labels_tensor = target_classes.data
                else:
                    speaker_labels_tensor = target_classes
            else:
                speaker_labels_tensor = None
                
            for i in range(self.hparams.num_spks):
                source = est_source[:, :, i]
                
                # Resample if needed
                if hasattr(self.hparams, "wavlm_sample_rate") and \
                self.hparams.wavlm_sample_rate != self.hparams.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        self.hparams.sample_rate, self.hparams.wavlm_sample_rate
                    ).to(self.device)
                    source = resampler(source)
                    
                # Fixed duration adjustments
                if hasattr(self.hparams, "fixed_duration"):
                    fixed_samples = int(self.hparams.wavlm_sample_rate * self.hparams.fixed_duration)
                    if source.shape[1] > fixed_samples:
                        source = source[:, :fixed_samples]
                    elif source.shape[1] < fixed_samples:
                        source = F.pad(source, (0, fixed_samples - source.shape[1]))
                        
                source = source.detach().clone()
                
                try:
                    if speaker_labels_tensor is not None:
                        # Get labels for current speaker
                        speaker_labels = speaker_labels_tensor[:, i].to(self.device)
                        
                        # Call classifier with labels
                        logits, embedding = self.modules.wavlm_classifier(source, speaker_labels)
                    else:
                        # Call classifier without labels
                        embedding = self.modules.wavlm_classifier(source)
                        logits = self.modules.wavlm_classifier.arcface(embedding)
                    
                    class_logits.append(logits)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    # Create fallback tensors
                    batch_size = source.shape[0]
                    embedding_dim = self.modules.wavlm_classifier.embedding_dim
                    num_classes = self.modules.wavlm_classifier.num_classes
                    
                    embedding = torch.zeros((batch_size, embedding_dim), device=self.device)
                    logits = torch.zeros((batch_size, num_classes), device=self.device)
                    
                    class_logits.append(logits)
                    embeddings.append(embedding)
            
            class_logits = torch.stack(class_logits, dim=1)  
            embeddings = torch.stack(embeddings, dim=1)  
            return est_source, targets, class_logits, embeddings
            
        return est_source, targets
    
    def compute_classification_loss(self, class_logits, speaker_labels):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        device = class_logits.device
        speaker_labels = speaker_labels.to(device)
        
        for i in range(self.hparams.num_spks):
            logits = class_logits[:, i]
            labels = speaker_labels[:, i]
            
            # Robust label filtering
            valid_indices = (labels >= 0) & (labels < self.hparams.num_speakers)
            valid_logits = logits[valid_indices]
            valid_labels = labels[valid_indices]
            
            if len(valid_labels) == 0:
                continue
            
            # Compute cross-entropy loss
            spk_loss = F.cross_entropy(valid_logits, valid_labels)
            
            # Predictions and accuracy
            predictions = torch.argmax(valid_logits, dim=1)
            correct = (predictions == valid_labels).sum().item()
            
            total_loss += spk_loss
            total_correct += correct
            total_samples += len(valid_labels)
        
        # Compute average loss and accuracy
        avg_loss = total_loss / self.hparams.num_spks if self.hparams.num_spks > 0 else total_loss
        accuracy = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
 
    def compute_objectives(self, predictions, targets):
        """Comprehensive objectives computation with robust classification handling."""
        alpha = self.hparams.classification_weight

        # Unpack predictions
        if isinstance(predictions, tuple) and len(predictions) == 4:
            est_source, targets, class_logits, embeddings = predictions
            
            # Separation loss computation
            separation_loss = self.hparams.separation_loss(targets, est_source)

            # Classification check
            if not hasattr(self.hparams, "use_classifier") or not self.hparams.use_classifier:
                return {"loss": separation_loss, "si-snr": separation_loss}

            # Get speaker labels
            speaker_labels = self.get_speaker_labels()

            # Validate labels
            if speaker_labels is None or speaker_labels.numel() == 0:
                return {"loss": separation_loss, "si-snr": separation_loss}

            try:
                # Compute classification loss
                class_loss, accuracy = self.compute_classification_loss(class_logits, speaker_labels)
                
                # Combine losses
                combined_loss = (1 - alpha) * separation_loss + alpha * class_loss
                
                return {
                    "loss": combined_loss, 
                    "si-snr": separation_loss, 
                    "class_loss": class_loss,
                    "accuracy": accuracy
                }
            except Exception as e:
                return {"loss": separation_loss, "si-snr": separation_loss}
        else:
            # Standard separation scenario
            est_source, targets = predictions
            separation_loss = self.hparams.separation_loss(targets, est_source)
            return {"loss": separation_loss, "si-snr": separation_loss}
        
    def get_speaker_labels(self):
        """Extract speaker labels from the current batch with comprehensive error handling."""
        try:
            # Check if current_batch exists and is not None
            if not hasattr(self.hparams, "current_batch") or self.hparams.current_batch is None:
                return None

            batch = self.hparams.current_batch

            # Multiple ways to access class labels
            if hasattr(batch, "class_labels"):
                labels = batch.class_labels
                
                # Handle PaddedData or similar wrapper types
                if hasattr(labels, 'data'):
                    return labels.data
                
                return labels

            # Fallback methods
            if hasattr(batch, "s1_class") and hasattr(batch, "s2_class"):
                s1_class = batch.s1_class
                s2_class = batch.s2_class
                
                # Convert to tensor if not already
                return torch.tensor([
                    int(float(s1_class)) if isinstance(s1_class, (str, float)) else s1_class,
                    int(float(s2_class)) if isinstance(s2_class, (str, float)) else s2_class
                ])

            return None

        except Exception as e:
            return None

    def fit_batch(self, batch):
        self.hparams.current_batch = batch

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        noise = batch.noise_sig[0]
        target_classes = None
        if hasattr(batch, 'class_labels'):
            target_classes = batch.class_labels
        
        with self.training_ctx:
            predictions = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN, noise, target_classes
            )
            loss = self.compute_objectives(predictions, targets)

            # Store all metrics for logging later
            self.train_metrics = loss if isinstance(loss, dict) else {"loss": loss, "si-snr": loss}
            
            # Extract scalar loss value for backprop
            if isinstance(loss, dict):
                loss_value = loss["loss"]
            else:
                loss_value = loss
                
            # Hard threshold the easy data items if required
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_value = loss_value[loss_value > th]
                if loss_value.nelement() > 0:
                    loss_value = loss_value.mean()
            else:
                # Ensure the loss is reduced to a scalar (mean of all elements if it's a tensor)
                loss_value = loss_value.mean()

        # Ensure loss is a tensor and finite before passing it
        if torch.isfinite(loss_value) and loss_value.nelement() > 0 and loss_value < self.hparams.loss_upper_lim:
            # Backpropagation and gradient step
            self.scaler.scale(loss_value).backward()
            
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Handle non-finite or invalid loss
            self.nonfinite_count += 1
            logger.info(
                f"infinite loss or empty loss! It happened {self.nonfinite_count} times so far - skipping this batch"
            )
            loss_value.data = torch.tensor(0.0).to(self.device)  # Set loss to zero to avoid affecting gradients
        
        # Zero gradients for the next step
        self.optimizer.zero_grad()

        # Clear current batch reference
        self.hparams.current_batch = None

        # Return the loss value as a tensor for SpeechBrain compatibility
        return loss_value

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        
        # Store current batch for label access 
        self.hparams.current_batch = batch
        
        # Get batch data
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        with torch.no_grad():
            predictions = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Handle audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            # Extract the first element (separated sources) from predictions if it's a tuple
            est_source = predictions[0] if isinstance(predictions, tuple) else predictions
            
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, est_source)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, est_source)

        # Clear current batch reference
        self.hparams.current_batch = None

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Modified to handle comprehensive metrics logging"""
        
        # For training, use metrics collected during training
        if stage == sb.Stage.TRAIN and hasattr(self, 'train_metrics'):
            stage_stats = self.train_metrics
        # For other stages, convert to dict if needed
        elif isinstance(stage_loss, dict):
            stage_stats = stage_loss
        else:
            stage_stats = {"si-snr": stage_loss}
            
        # Store training stats
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
            # Log training metrics to CSV
            self.hparams.log_metrics(
                epoch=epoch,
                stage="train",
                metrics=stage_stats,
                lr=self.optimizer.param_groups[0]["lr"]
            )
            
            # Save checkpoint after each training epoch
            self.checkpointer.save_checkpoint(
                meta={"si-snr": stage_stats.get("si-snr", 0), "epoch": epoch}
            )

        # Validation stage processing
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                # Use SI-SNR for LR scheduling
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_stats["si-snr"]
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log validation metrics to CSV
            self.hparams.log_metrics(
                epoch=epoch,
                stage="valid",
                metrics=stage_stats,
                lr=current_lr
            )
            
            # Store current validation stats for next epoch comparison
            self.prev_valid_stats = {}
            for k, v in stage_stats.items():
                if isinstance(v, torch.Tensor):
                    self.prev_valid_stats[k] = v.item()
                else:
                    self.prev_valid_stats[k] = v
            
            # Save checkpoint
            if (
                hasattr(self.hparams, "save_all_checkpoints")
                and self.hparams.save_all_checkpoints
            ):
                self.checkpointer.save_checkpoint(
                    meta={"si-snr": stage_stats["si-snr"]}
                )
            else:
                self.checkpointer.save_and_keep_only(
                    meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
                )
                    
        # Test stage processing
        elif stage == sb.Stage.TEST:
            # Log test metrics to CSV
            self.hparams.log_metrics(
                epoch=self.hparams.epoch_counter.current,
                stage="test",
                metrics=stage_stats
            )
        
        # Print a comprehensive summary of all metrics at the end of each epoch
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch} {stage.name} METRICS SUMMARY")
        print(f"{'=' * 70}")
        
        # Format learning rate info
        if stage == sb.Stage.TRAIN:
            lr = self.optimizer.param_groups[0]["lr"]
            lr_info = f"Learning rate: {lr:.6f}"
        elif stage == sb.Stage.VALID:
            lr = current_lr if 'current_lr' in locals() else "N/A"
            lr_info = f"Learning rate: {lr}"
        else:
            lr_info = ""
        
        if lr_info:
            print(lr_info)
        
        # Print all metrics in a formatted table
        print(f"\n{'Metric':<20} {'Value':<15}")
        print(f"{'-' * 20} {'-' * 15}")
        
        for metric, value in stage_stats.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, float):
                print(f"{metric:<20} {value:.4f}")
            else:
                print(f"{metric:<20} {value}")
        
        # For validation, show changes from previous epoch if available
        if stage == sb.Stage.VALID and hasattr(self, 'prev_valid_stats') and epoch > 1:
            print(f"\n{'Metric':<20} {'Change':<15}")
            print(f"{'-' * 20} {'-' * 15}")
            
            for metric in stage_stats:
                if metric in self.prev_valid_stats:
                    prev_value = self.prev_valid_stats[metric]
                    curr_value = stage_stats[metric]
                    
                    if isinstance(prev_value, torch.Tensor):
                        prev_value = prev_value.item()
                    if isinstance(curr_value, torch.Tensor):
                        curr_value = curr_value.item()
                        
                    if isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
                        change = curr_value - prev_value
                        print(f"{metric:<20} {change:+.4f}")
        
        print(f"{'=' * 70}\n")

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file, along with classification accuracy"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        all_class_accs = []  # Track classification accuracy
        
        # CSV columns - add classification accuracy
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]
        if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier:
            csv_columns.append("class_acc")

        # Create dataloader for test data
        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    
                    # Store batch for access to labels
                    self.hparams.current_batch = batch
                    
                    # Apply Separation and Classification
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # If classification was performed, unpack results
                    if isinstance(predictions, tuple) and len(predictions) == 4:
                        est_source, _, class_logits, _ = predictions
                    else:
                        est_source = predictions
                        class_logits = None

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(
                        (est_source, targets), targets
                    )
                    if isinstance(sisnr, dict):
                        sisnr = sisnr["si-snr"]

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        (mixture_signal, targets), targets
                    )
                    if isinstance(sisnr_baseline, dict):
                        sisnr_baseline = sisnr_baseline["si-snr"]
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        est_source[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()
                    
                    # Calculate classification accuracy if applicable
                    class_acc = 0
                    if class_logits is not None and hasattr(batch, "class_labels"):
                        correct = 0
                        total = 0
                        
                        speaker_labels = batch.class_labels.to(self.device)
                        for spk in range(self.hparams.num_spks):
                            preds = class_logits[:, spk].argmax(dim=1)
                            correct += (preds == speaker_labels[:, spk]).sum().item()
                            total += speaker_labels.shape[0]
                        
                        class_acc = correct / (total * self.hparams.num_spks) if total > 0 else 0
                        all_class_accs.append(class_acc)

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    
                    if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier:
                        row["class_acc"] = class_acc
                        
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())
                    
                    # Clear current batch reference
                    self.hparams.current_batch = None

                # Write average row
                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                
                if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier:
                    row["class_acc"] = np.array(all_class_accs).mean()
                    
                writer.writerow(row)

        # Print final test results summary
        print(f"\n{'=' * 70}")
        print("TEST RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"Mean SISNR: {np.array(all_sisnrs).mean():.4f}")
        print(f"Mean SISNRi: {np.array(all_sisnrs_i).mean():.4f}")
        print(f"Mean SDR: {np.array(all_sdrs).mean():.4f}")
        print(f"Mean SDRi: {np.array(all_sdrs_i).mean():.4f}")
        
        if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier:
            print(f"Mean Classification Accuracy: {np.array(all_class_accs).mean():.4f}")
        print(f"{'=' * 70}\n")

        logger.info(f"Mean SISNR is {np.array(all_sisnrs).mean()}")
        logger.info(f"Mean SISNRi is {np.array(all_sisnrs_i).mean()}")
        logger.info(f"Mean SDR is {np.array(all_sdrs).mean()}")
        logger.info(f"Mean SDRi is {np.array(all_sdrs_i).mean()}")
        
        if hasattr(self.hparams, "use_classifier") and self.hparams.use_classifier:
            logger.info(f"Mean Classification Accuracy is {np.array(all_class_accs).mean()}")
            
    # Keep the existing methods here:
    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""
        # Implementation same as original
        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)
    
    def save_audio(self, snt_id, mixture, targets, predictions):
        """Saves the test audio (mixture, targets, and estimated sources) on disk"""
        # Create output folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, f"item{snt_id}_source{ns+1}hat.wav"
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, f"item{snt_id}_source{ns+1}.wav"
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, f"item{snt_id}_mix.wav")
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline with classification labels."""

    # Load dataset
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    
    # Existing audio pipeline functions
    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        sig = sb.dataio.dataio.read_audio(mix_wav)
        return sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        sig = sb.dataio.dataio.read_audio(s1_wav)
        return sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        sig = sb.dataio.dataio.read_audio(s2_wav)
        return sig

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("noise_sig")
    def audio_pipeline_noise(_mix_path):
        mix_sig = sb.dataio.dataio.read_audio(_mix_path)
        noise = 0.01 * torch.randn_like(mix_sig)
        return noise

    # Classification pipeline with explicit takes and provides
    if hparams.get("use_classifier", False):
        @sb.utils.data_pipeline.takes("s1_class", "s2_class")
        @sb.utils.data_pipeline.provides("class_labels")
        def class_pipeline(s1_class, s2_class):
            """
            Convert speaker class labels to tensor.
            """
            try:
                # Convert to integers, handling different input types
                def safe_parse_class(cls_val):
                    # Handle string and numeric inputs
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
                
                result = torch.tensor([s1_label, s2_label])
                return result
            
            except Exception as e:
                return torch.tensor([-1, -1])

        # Add audio and classification pipelines
        sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_mix)
        sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_s1)
        sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_s2)
        sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_noise)
        sb.dataio.dataset.add_dynamic_item([train_data], class_pipeline)

        # Set output keys
        output_keys = ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig", "class_labels"]
    else:
        # If classifier is not used, set output keys without class labels
        output_keys = ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]

    # Set output keys for the dataset
    sb.dataio.dataset.set_output_keys([train_data], output_keys)

    return train_data


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        
    # Print device information
    device = run_opts.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 50}")
    print(f"DEVICE INFORMATION")
    print(f"{'=' * 50}")
    print(f"Using device: {device}")
    
    try:
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current GPU: {torch.cuda.current_device()}")
        else:
            print("CUDA available: No")
    except Exception as e:
        print(f"Error getting CUDA information: {e}")
    print(f"{'=' * 50}\n")

    # Create metrics file path from hparams
    if "metrics_file" in hparams:
        metrics_file = hparams["metrics_file"]
    else:
        # Fallback to hardcoded path with seed if not in yaml
        seed = hparams.get("seed", 0)
        metrics_file = f"/iitjhome/m23mac004/ZZZZZ/SepFormer_WavLM_Separation_Classification_{seed}.csv"
    
    log_metrics = create_csv_logger(metrics_file)
    hparams["log_metrics"] = log_metrics

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)  

    logger = get_logger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Data preparation for whamr
    if "whamr" in hparams["data_folder"] and hparams["use_speedperturb"]:
        logger.info("WHAMR dataset detected but using custom implementation")
        pass


    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from recipes.WHAMandWHAMR.separation.dynamic_mixing import dynamic_mix_data_prep

        # Check and process the base folder for dynamic mixing if needed
        if "processed" not in hparams["base_folder_dm"]:
            # If the processed folder doesn't exist, create it
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from recipes.WHAMandWHAMR.meta.preprocess_dynamic_mixing import (
                    resample_folder,
                )

                logger.info("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # Update the base folder path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                logger.info("Using existing processed folder")
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Prepare the dynamic mixing dataset
        train_data = dynamic_mix_data_prep(
            tr_csv=hparams["train_data"],
            data_root_folder=hparams["data_folder"],
            base_folder_dm=hparams["base_folder_dm"],
            sample_rate=hparams["sample_rate"],
            num_spks=hparams["num_spks"],
            max_training_signal_len=hparams["training_signal_len"],
            batch_size=hparams["dataloader_opts"]["batch_size"],
            num_workers=hparams["dataloader_opts"]["num_workers"],
        )
    else:
        # Standard dataset preparation
        train_data = dataio_prep(hparams)

    # Load pretrained separator if specified
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = SeparationAndClassification(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Re-initialize parameters if not using pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        train_loader_kwargs=hparams["dataloader_opts"],
    )

    # Evaluation
    if "test_data" in hparams:
        test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["test_data"],
            replacements={"data_root": hparams["data_folder"]},
        )
        # Add dynamic items to test data
        dataio_prep(hparams)
        
        # Run evaluation
        separator.evaluate(test_data, min_key="si-snr")
        separator.save_results(test_data)

