import torch
import torch.nn.functional as F
from transformers import Trainer
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from typing import Dict


class LowRankTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.custom_args = kwargs.pop('custom_args', None)
        super().__init__(*args, **kwargs)

        # Ensure it has attributes
        assert hasattr(self.model, 'train_ranks'), 'Model must have train ranks attribute'
        assert hasattr(self.model, 'log_vars_ce'), 'Must have log vars ce'
        assert hasattr(self.model, '_set_rank_all'), 'Must have set rank all'
        assert hasattr(self.model, 'max_rank'), 'Models must have max rank attribute'

    def create_optimizer(self):
        log_var_params = list(self.model.log_vars_ce.parameters())
        log_var_params_id = {id(p) for p in log_var_params}

        # Separate model parameters from uncertainty
        model_params = [p for p in self.model.parameters() if id(p) not in log_var_params_id]

        optimizer_grouped_parameters = [
            {
                'params': model_params,
                'lr': self.args.learning_rate,
            },
            {
                'params': log_var_params,
                'lr': 1e-2
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.args.weight_decay
        )
        self.optimizer = optimizer
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.model.training:
            # --- TRAINING LOGIC ---
            labels = inputs.pop("labels")
            r_anchor = model.max_rank
            all_possible_variants = sorted([r for r in model.train_ranks if r < r_anchor], reverse=True)
            progress = self.state.global_step / self.state.max_steps

            if progress < self.custom_args.warmup_proportion:
                model._set_rank_all(r_anchor)
                outputs = model(**inputs)

                ce_anchor = F.cross_entropy(outputs.logits, labels)
                log_var_a = model.log_vars_ce[str(r_anchor)].to(ce_anchor.device)
                loss = torch.exp(-log_var_a) * ce_anchor + log_var_a
                
                with torch.no_grad():
                    acc = (outputs.logits.argmax(1) == labels).float().mean()

                self.log({"loss": loss.item(), "acc": acc.item(), "phase": "warmup"})
                
                return (loss, outputs) if return_outputs else loss

            else:
                    
                if self.custom_args.sampling_strategy == 'curriculum':

                    # 1. Determine the training progress

                    # 2. Define curriculum stages
                    num_stages = len(all_possible_variants)
                    stage_idx = min(int(progress * num_stages), num_stages - 1)

                    # Get variables
                    available_variants = all_possible_variants[:stage_idx + 1]

                    if not available_variants:
                        available_variants = [all_possible_variants[0]]
                    
                    r_variant = random.choice(available_variants)

                elif self.custom_args.sampling_strategy == 'epoch_curriculum':
                    if self.state.epoch < 1.0:
                        # Epoch 1: Use the standard curriculum logic
                        num_stages = len(all_possible_variants)
                        stage_idx = min(int(progress * num_stages), num_stages - 1)
                        available_variants = all_possible_variants[:stage_idx + 1]
                        if not available_variants:
                            available_variants = [all_possible_variants[0]]
                        r_variant = random.choice(available_variants)
                    else:
                        # Epoch 2+: Use standard uniform random sampling
                        r_variant = random.choice(all_possible_variants)
                        
                elif self.custom_args.sampling_strategy == 'faster_curriculum':
                    num_variants = len(all_possible_variants) # Define num_variants

                    if progress < 0.5:
                        # PHASE 1: Sequential Introduction (0% -> 50% progress)
                        # Normalize progress within this phase to determine how many ranks to "unlock"
                        phase1_progress = progress / 0.5
                        
                        # Calculate the number of variants to include in the pool
                        num_unlocked = min(int(phase1_progress * num_variants) + 1, num_variants)
                        available_variants = all_possible_variants[:num_unlocked]
                        r_variant = random.choice(available_variants)

                    else:
                        # 3. PHASE 2: Joint Fine-Tuning (50% -> 100% progress)
                        # All variants are available for sampling
                        available_variants = all_possible_variants
                        r_variant = random.choice(available_variants)

                else:
                    r_variant = random.choice(all_possible_variants)

                #possible_variants = [r for r in model.train_ranks if r < r_anchor]
                #r_variant = random.choice(possible_variants)
                model._set_rank_all(r_anchor)
                output_anchor = model(**inputs)
                labels.to(output_anchor.logits.device)
                ce_anchor = F.cross_entropy(output_anchor.logits, labels)
                model._set_rank_all(r_variant)
                outputs_variant = model(**inputs)
                ce_variant = F.cross_entropy(outputs_variant.logits, labels)
                model._set_rank_all(model.max_rank)
                loss_device = ce_anchor.device
                log_var_a = model.log_vars_ce[str(r_anchor)].to(loss_device)
                log_var_v = model.log_vars_ce[str(r_variant)].to(loss_device)
                log_var_a = torch.clamp(log_var_a, min=-10, max=10)
                log_var_v = torch.clamp(log_var_v, min=-10, max=10)
                gradient_floor = 1e-2
                loss_a = (torch.exp(-log_var_a) + gradient_floor) * ce_anchor + log_var_a
                loss_b = (torch.exp(-log_var_v) + gradient_floor) * ce_variant + log_var_v
                loss = loss_a + loss_b
                outputs = {"logits": output_anchor.logits}

                # Log training metrics
                with torch.no_grad():
                    acc = (output_anchor.logits.argmax(1) == labels).float().mean()
                
                # Log training metrics
                
                self.log({
                    "loss": loss.item(), # Note: I changed the keys to be more standard
                    f"ce_rank_{r_anchor}": ce_anchor.item(),
                    f"ce_rank_{r_variant}": ce_variant.item(),
                    "acc": acc.item(),
                    "r_anchor": float(r_anchor),
                    "r_variant": float(r_variant),
                    f"s_rank_{r_anchor}": log_var_a.item(),
                    f"s_rank_{r_variant}": log_var_v.item(),
                })

                return (loss, outputs) if return_outputs else loss
        else:
            # --- EVALUATION LOGIC ---
            # Perform a standard, single forward pass. The `evaluate` method
            # has already set the correct rank for us.
            outputs = model(**inputs)
            # The Trainer's prediction_step will handle the loss calculation
            # from the model's output.
            return (outputs.loss, outputs)

    def evaluate(
        self, 
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        ignore_keys=None,
        metric_key_prefix='eval'
    ):
        """Overrides the default evaluation to test the model at all specified ranks"""
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        original_rank = self.model.max_rank

        all_metrics = {}

        print("\n -- Running multi-rank evaluation ---")
        ranks_to_eval = sorted(self.model.train_ranks, reverse=True)
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        for rank in ranks_to_eval:
            print(f"Evaluationg at rank {rank}")
            self.model._set_rank_all(rank)
            
            # Run standard evaluation loop
            output = self.evaluation_loop(
                eval_dataloader,
                description=f"Evaluating rank {rank}",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )            
            # Rename metrics
            rank_metrics = {f"{metric_key_prefix}_rank_{rank}/{k.replace(metric_key_prefix + '_', '')}": v for k, v in output.metrics.items()}
            all_metrics.update(rank_metrics)

        # Restore original rank
        self.model._set_rank_all(original_rank)

        # The Trainer needs a primary metric for `load_best_model_at_end`.
        # We'll use the loss from the highest rank for this purpose.
        max_rank_metric_key = f"{metric_key_prefix}_rank_{original_rank}/loss"
        if max_rank_metric_key in all_metrics:
            all_metrics[f"{metric_key_prefix}_loss"] = all_metrics[max_rank_metric_key]

        self.log(all_metrics)
        return all_metrics

class ComparableBaselineTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides compute_loss to also log training accuracy at each logging step.
        """
        # Get the standard outputs (loss, logits, etc.) from the model
        outputs = model(**inputs)
        
        # If we are in a training step, calculate and log accuracy for this batch
        if self.is_in_train:
            labels = inputs.get("labels")
            if labels is not None:
                with torch.no_grad():
                    preds = outputs.logits.argmax(-1)
                    acc = (preds == labels).float().mean()
                    # Log loss and accuracy. The trainer will average these over logging_steps.
                    self.log({"train_loss": outputs.loss.item(), "acc": acc.item()})
        
        # Return the loss for the backward pass, or all outputs if requested
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overrides the standard evaluate method to format the log keys to be
        comparable with the multi-rank evaluation from LowRankTrainer.
        """
        # This is the standard evaluation loop, which we call to get the metrics
        output = super().evaluation_loop(
            self.get_eval_dataloader(eval_dataset),
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        original_metrics = output.metrics
        
        # --- Custom Formatting Logic ---
        # Create a new dictionary with keys formatted for comparability
        # e.g., 'eval_accuracy' becomes 'eval/rank_baseline/accuracy'
        renamed_metrics = {}
        for key, value in original_metrics.items():
            # Removes the 'eval_' prefix to get the clean metric name
            clean_key = key[len(metric_key_prefix) + 1:] 
            renamed_metrics[f"{metric_key_prefix}/rank_baseline/{clean_key}"] = value
        
        # Add the original primary metric back for `load_best_model_at_end`
        if f"{metric_key_prefix}_loss" in original_metrics:
            renamed_metrics[f"{metric_key_prefix}_loss"] = original_metrics[f"{metric_key_prefix}_loss"]

        # Log the newly formatted metrics
        self.log(renamed_metrics)

        # The rest of this is standard boilerplate from the parent Trainer.evaluate
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, renamed_metrics)
        return renamed_metrics