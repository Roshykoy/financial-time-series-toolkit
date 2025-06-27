# src/cvae_engine_improved.py - Enhanced CVAE engine with comprehensive error handling
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

from src.utils.safe_math import (
    safe_divide, safe_log, safe_exp, safe_softmax, safe_normalize,
    safe_cosine_similarity, stable_kl_divergence, check_tensor_health,
    safe_tensor_operation, gradient_clipping_with_health_check
)
from src.utils.error_handling import (
    ModelError, GPUError, robust_operation, safe_model_operation
)


class ImprovedCVAELossComputer:
    """Enhanced CVAE loss computer with comprehensive error handling and numerical stability."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.eps = config.get('numerical_stability_eps', 1e-8)
        self.error_counts = defaultdict(int)
        
    @safe_tensor_operation("reconstruction_loss")
    def reconstruction_loss(self, reconstruction_logits, target_combinations):
        """
        Enhanced reconstruction loss with comprehensive error handling.
        
        Args:
            reconstruction_logits: [batch_size, 6, num_numbers]
            target_combinations: [batch_size, 6] - 1-based indices
        
        Returns:
            loss: Reconstruction loss
        """
        # Input validation
        if reconstruction_logits.size(0) == 0:
            warnings.warn("Empty batch in reconstruction loss", RuntimeWarning)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if not check_tensor_health(reconstruction_logits, "reconstruction_logits"):
            raise ModelError("Invalid reconstruction logits detected")
        
        if not check_tensor_health(target_combinations, "target_combinations"):
            raise ModelError("Invalid target combinations detected")
        
        batch_size = target_combinations.size(0)
        
        # Validate target combinations are in valid range
        if target_combinations.min() < 1 or target_combinations.max() > 49:
            raise ModelError(f"Target combinations out of range [1, 49]: {target_combinations.min()}-{target_combinations.max()}")
        
        # Convert to 0-based indices for cross-entropy
        targets_zero_based = target_combinations - 1
        
        # Ensure targets are within valid range for logits
        num_classes = reconstruction_logits.size(-1)
        if targets_zero_based.max() >= num_classes:
            raise ModelError(f"Target index {targets_zero_based.max()} exceeds logits classes {num_classes}")
        
        # Compute cross-entropy loss for each position with error handling
        total_loss = 0
        valid_positions = 0
        
        for pos in range(6):
            try:
                pos_logits = reconstruction_logits[:, pos, :]  # [batch_size, num_numbers]
                pos_targets = targets_zero_based[:, pos]      # [batch_size]
                
                # Check for valid logits
                if not check_tensor_health(pos_logits, f"pos_logits_{pos}"):
                    warnings.warn(f"Skipping position {pos} due to invalid logits", RuntimeWarning)
                    continue
                
                # Apply stable softmax for numerical stability
                pos_probs = safe_softmax(pos_logits, dim=-1, eps=self.eps)
                
                # Manual cross-entropy with numerical stability
                gathered_probs = pos_probs.gather(1, pos_targets.unsqueeze(1)).squeeze(1)
                pos_loss = -safe_log(gathered_probs, eps=self.eps).mean()
                
                if torch.isfinite(pos_loss):
                    total_loss += pos_loss
                    valid_positions += 1
                else:
                    warnings.warn(f"Non-finite loss at position {pos}", RuntimeWarning)
                    
            except Exception as e:
                warnings.warn(f"Error computing loss for position {pos}: {e}", RuntimeWarning)
                continue
        
        if valid_positions == 0:
            warnings.warn("No valid positions for reconstruction loss", RuntimeWarning)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss / valid_positions
    
    @safe_tensor_operation("kl_divergence")
    def kl_divergence_loss(self, mu, logvar, mu_prior, logvar_prior):
        """
        Enhanced KL divergence loss with numerical stability.
        
        KL(q(z|x,c) || p(z|c)) where c is temporal context
        """
        # Input validation
        tensors_to_check = [
            (mu, "mu"), (logvar, "logvar"), 
            (mu_prior, "mu_prior"), (logvar_prior, "logvar_prior")
        ]
        
        for tensor, name in tensors_to_check:
            if not check_tensor_health(tensor, name):
                warnings.warn(f"Invalid tensor {name} in KL divergence", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Use stable KL divergence computation
        try:
            kl_div = stable_kl_divergence(mu, logvar, mu_prior, logvar_prior, eps=self.eps)
            
            # Additional safety check
            if not torch.isfinite(kl_div):
                warnings.warn("Non-finite KL divergence, using zero", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Clamp to reasonable range to prevent explosion
            kl_div = torch.clamp(kl_div, max=100.0)
            
            return kl_div
            
        except Exception as e:
            warnings.warn(f"KL divergence computation failed: {e}", RuntimeWarning)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    @safe_model_operation("contrastive_loss")
    def hard_contrastive_loss(self, model, positive_combinations, negative_combinations, 
                            pair_counts, df, current_indices):
        """
        Enhanced contrastive loss with comprehensive error handling.
        """
        try:
            batch_size = len(positive_combinations)
            if batch_size == 0:
                warnings.warn("Empty positive combinations for contrastive loss", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            device = self.device
            
            # Get embeddings for positive samples with error handling
            try:
                mu_pos, logvar_pos = model.encode(positive_combinations, pair_counts)
                if not check_tensor_health(mu_pos, "mu_pos") or not check_tensor_health(logvar_pos, "logvar_pos"):
                    raise ModelError("Invalid positive embeddings")
                
                z_pos = model.reparameterize(mu_pos, logvar_pos)
                if not check_tensor_health(z_pos, "z_pos"):
                    raise ModelError("Invalid positive latent variables")
                    
            except Exception as e:
                warnings.warn(f"Failed to encode positive samples: {e}", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Sample hard negatives with error handling
            try:
                hard_negatives = self._mine_hard_negatives_safe(
                    model, positive_combinations, negative_combinations, 
                    pair_counts, df, current_indices
                )
                
                if len(hard_negatives) == 0:
                    warnings.warn("No hard negatives found, using random negatives", RuntimeWarning)
                    num_neg = min(self.config.get('negative_samples', 8), len(negative_combinations))
                    hard_negatives = random.sample(negative_combinations, num_neg)
                    
            except Exception as e:
                warnings.warn(f"Hard negative mining failed: {e}", RuntimeWarning)
                # Fallback to random negatives
                num_neg = min(self.config.get('negative_samples', 8), len(negative_combinations))
                hard_negatives = random.sample(negative_combinations, num_neg) if negative_combinations else []
                
                if not hard_negatives:
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Get embeddings for hard negatives with error handling
            try:
                mu_neg, logvar_neg = model.encode(hard_negatives, pair_counts)
                if not check_tensor_health(mu_neg, "mu_neg") or not check_tensor_health(logvar_neg, "logvar_neg"):
                    raise ModelError("Invalid negative embeddings")
                
                z_neg = model.reparameterize(mu_neg, logvar_neg)
                if not check_tensor_health(z_neg, "z_neg"):
                    raise ModelError("Invalid negative latent variables")
                    
            except Exception as e:
                warnings.warn(f"Failed to encode negative samples: {e}", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Compute contrastive loss in latent space with numerical stability
            try:
                # Reshape for batch processing
                num_neg_per_pos = z_neg.size(0) // batch_size
                z_neg_reshaped = z_neg.view(batch_size, num_neg_per_pos, -1)
                
                # Compute similarities with safe operations
                similarities = []
                for i in range(batch_size):
                    pos_z = z_pos[i:i+1]  # [1, latent_dim]
                    neg_z = z_neg_reshaped[i]  # [num_neg, latent_dim]
                    
                    sim = safe_cosine_similarity(pos_z, neg_z, dim=-1, eps=self.eps)
                    similarities.append(sim)
                
                similarities = torch.stack(similarities)  # [batch_size, num_neg]
                
                # InfoNCE-style loss with numerical stability
                temperature = max(self.config.get('contrastive_temperature', 0.2), self.eps)
                
                # Compute exponentials with overflow protection
                exp_sim = safe_exp(similarities / temperature, max_exp=50.0)
                
                # Positive similarity (self-similarity should be 1.0)
                pos_sim = torch.ones(batch_size, device=device)
                exp_pos = safe_exp(pos_sim / temperature, max_exp=50.0)
                
                # Contrastive loss with safe operations
                denominator = exp_pos + exp_sim.sum(dim=-1)
                safe_denominator = torch.clamp(denominator, min=self.eps)
                
                contrastive_loss = -safe_log(exp_pos / safe_denominator, eps=self.eps)
                
                # Check final loss
                if not check_tensor_health(contrastive_loss, "contrastive_loss"):
                    warnings.warn("Invalid contrastive loss, returning zero", RuntimeWarning)
                    return torch.tensor(0.0, device=self.device, requires_grad=True)
                
                return contrastive_loss.mean()
                
            except Exception as e:
                warnings.warn(f"Contrastive loss computation failed: {e}", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        except Exception as e:
            self.error_counts['contrastive_loss'] += 1
            warnings.warn(f"Contrastive loss failed (count: {self.error_counts['contrastive_loss']}): {e}", RuntimeWarning)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _mine_hard_negatives_safe(self, model, positive_combinations, negative_pool, 
                                pair_counts, df, current_indices):
        """
        Enhanced hard negative mining with comprehensive error handling.
        """
        if not negative_pool:
            return []
        
        batch_size = len(positive_combinations)
        num_negatives = self.config.get('negative_samples', 8)
        hard_ratio = self.config.get('hard_negative_ratio', 0.5)
        num_hard = max(1, int(num_negatives * hard_ratio))
        num_random = num_negatives - num_hard
        
        hard_negatives = []
        
        try:
            with torch.no_grad():
                # Get positive embeddings safely
                try:
                    mu_pos, logvar_pos = model.encode(positive_combinations, pair_counts)
                    z_pos = model.reparameterize(mu_pos, logvar_pos)
                    
                    if not check_tensor_health(z_pos, "z_pos_mining"):
                        raise ModelError("Invalid positive embeddings for mining")
                        
                except Exception as e:
                    warnings.warn(f"Failed to get positive embeddings for mining: {e}", RuntimeWarning)
                    # Fallback to random sampling
                    return random.sample(negative_pool, min(num_negatives, len(negative_pool)))
                
                for i in range(batch_size):
                    try:
                        batch_hard_negatives = []
                        
                        # Sample candidate negatives safely
                        max_candidates = min(200, len(negative_pool))
                        if max_candidates == 0:
                            continue
                        
                        candidate_negatives = random.sample(negative_pool, max_candidates)
                        
                        if not candidate_negatives:
                            # Fallback to random negatives
                            fallback_negatives = random.sample(negative_pool, min(num_negatives, len(negative_pool)))
                            hard_negatives.extend(fallback_negatives)
                            continue
                        
                        # Get embeddings for candidates safely
                        try:
                            mu_neg_cand, logvar_neg_cand = model.encode(candidate_negatives, pair_counts)
                            z_neg_cand = model.reparameterize(mu_neg_cand, logvar_neg_cand)
                            
                            if not check_tensor_health(z_neg_cand, "z_neg_candidates"):
                                raise ModelError("Invalid negative candidate embeddings")
                                
                        except Exception as e:
                            warnings.warn(f"Failed to encode negative candidates: {e}", RuntimeWarning)
                            # Use random negatives for this batch
                            fallback_negatives = random.sample(negative_pool, min(num_negatives, len(negative_pool)))
                            hard_negatives.extend(fallback_negatives)
                            continue
                        
                        # Compute similarities safely
                        try:
                            pos_z = z_pos[i:i+1]  # [1, latent_dim]
                            similarities = safe_cosine_similarity(pos_z, z_neg_cand, dim=-1, eps=self.eps)
                            
                            if not check_tensor_health(similarities, "similarities"):
                                raise ModelError("Invalid similarities")
                            
                            # Select hardest negatives (most similar to positive)
                            if len(candidate_negatives) >= num_hard:
                                try:
                                    _, hard_indices = torch.topk(similarities, num_hard, largest=True)
                                    hard_negs = [candidate_negatives[idx.item()] for idx in hard_indices]
                                    batch_hard_negatives.extend(hard_negs)
                                except Exception as topk_error:
                                    warnings.warn(f"TopK selection failed: {topk_error}", RuntimeWarning)
                                    # Use first num_hard candidates
                                    batch_hard_negatives.extend(candidate_negatives[:num_hard])
                            else:
                                # Use all candidates as hard negatives
                                batch_hard_negatives.extend(candidate_negatives)
                            
                            # Add random negatives to fill quota
                            remaining_negatives = num_negatives - len(batch_hard_negatives)
                            if remaining_negatives > 0:
                                available_random = [neg for neg in negative_pool if neg not in batch_hard_negatives]
                                if available_random:
                                    num_random_sample = min(remaining_negatives, len(available_random))
                                    random_negs = random.sample(available_random, num_random_sample)
                                    batch_hard_negatives.extend(random_negs)
                            
                            hard_negatives.extend(batch_hard_negatives[:num_negatives])
                            
                        except Exception as sim_error:
                            warnings.warn(f"Similarity computation failed: {sim_error}", RuntimeWarning)
                            # Fallback to random for this batch
                            fallback_negatives = random.sample(negative_pool, min(num_negatives, len(negative_pool)))
                            hard_negatives.extend(fallback_negatives)
                            
                    except Exception as batch_error:
                        warnings.warn(f"Hard negative mining failed for batch {i}: {batch_error}", RuntimeWarning)
                        # Continue with next batch
                        continue
                
        except Exception as e:
            warnings.warn(f"Hard negative mining completely failed: {e}", RuntimeWarning)
            # Final fallback: return random negatives
            return random.sample(negative_pool, min(num_negatives * batch_size, len(negative_pool)))
        
        # Ensure we have at least some negatives
        if not hard_negatives and negative_pool:
            warnings.warn("No hard negatives mined, using random fallback", RuntimeWarning)
            return random.sample(negative_pool, min(num_negatives * batch_size, len(negative_pool)))
        
        return hard_negatives


class SafeCVAETrainer:
    """Enhanced CVAE trainer with comprehensive error handling and recovery."""
    
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.loss_computer = ImprovedCVAELossComputer(config)
        self.error_counts = defaultdict(int)
        self.training_history = []
        
    @robust_operation(max_retries=3, exceptions=(RuntimeError, ModelError))
    def compute_total_loss(self, model, batch_data, epoch_info=None):
        """
        Compute total CVAE loss with comprehensive error handling and recovery.
        """
        try:
            # Unpack batch data safely
            positive_combinations = batch_data.get('positive_combinations', [])
            negative_combinations = batch_data.get('negative_combinations', [])
            pair_counts = batch_data.get('pair_counts', {})
            df = batch_data.get('df')
            current_indices = batch_data.get('current_indices', [])
            
            if not positive_combinations:
                warnings.warn("No positive combinations in batch", RuntimeWarning)
                return torch.tensor(0.0, device=self.device, requires_grad=True), {}
            
            # Forward pass with error handling
            try:
                model_output = model(positive_combinations, pair_counts)
                
                # Validate model output
                required_keys = ['reconstruction_logits', 'mu', 'logvar', 'mu_prior', 'logvar_prior']
                for key in required_keys:
                    if key not in model_output:
                        raise ModelError(f"Missing key '{key}' in model output")
                    if not check_tensor_health(model_output[key], key):
                        raise ModelError(f"Invalid tensor '{key}' in model output")
                
            except Exception as e:
                raise ModelError(f"Model forward pass failed: {e}")
            
            # Compute individual losses with error handling
            losses = {}
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Reconstruction loss
            try:
                recon_loss = self.loss_computer.reconstruction_loss(
                    model_output['reconstruction_logits'], 
                    positive_combinations
                )
                losses['reconstruction'] = recon_loss
                total_loss = total_loss + self.config.get('reconstruction_weight', 1.0) * recon_loss
            except Exception as e:
                warnings.warn(f"Reconstruction loss failed: {e}", RuntimeWarning)
                losses['reconstruction'] = torch.tensor(0.0, device=self.device)
            
            # KL divergence loss
            try:
                kl_loss = self.loss_computer.kl_divergence_loss(
                    model_output['mu'], model_output['logvar'],
                    model_output['mu_prior'], model_output['logvar_prior']
                )
                losses['kl_divergence'] = kl_loss
                total_loss = total_loss + self.config.get('kl_weight', 0.01) * kl_loss
            except Exception as e:
                warnings.warn(f"KL divergence loss failed: {e}", RuntimeWarning)
                losses['kl_divergence'] = torch.tensor(0.0, device=self.device)
            
            # Contrastive loss (optional, with error handling)
            if negative_combinations and self.config.get('contrastive_weight', 0.0) > 0:
                try:
                    contrastive_loss = self.loss_computer.hard_contrastive_loss(
                        model, positive_combinations, negative_combinations,
                        pair_counts, df, current_indices
                    )
                    losses['contrastive'] = contrastive_loss
                    total_loss = total_loss + self.config.get('contrastive_weight', 0.1) * contrastive_loss
                except Exception as e:
                    warnings.warn(f"Contrastive loss failed: {e}", RuntimeWarning)
                    losses['contrastive'] = torch.tensor(0.0, device=self.device)
            
            # Final safety check
            if not check_tensor_health(total_loss, "total_loss"):
                warnings.warn("Invalid total loss, using reconstruction loss only", RuntimeWarning)
                total_loss = losses.get('reconstruction', torch.tensor(0.0, device=self.device, requires_grad=True))
            
            # Record training history
            loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}
            loss_dict['total'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
            self.training_history.append(loss_dict)
            
            return total_loss, losses
            
        except Exception as e:
            self.error_counts['total_loss'] += 1
            raise ModelError(f"Total loss computation failed (count: {self.error_counts['total_loss']}): {e}")
    
    def safe_backward_pass(self, loss, model, optimizer, scaler=None):
        """
        Perform safe backward pass with gradient health checking.
        """
        try:
            # Clear gradients
            optimizer.zero_grad()
            
            # Backward pass with mixed precision handling
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Check for gradient health before unscaling
                parameters = list(model.parameters())
                if any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) 
                       for p in parameters if p.grad is not None):
                    warnings.warn("NaN or Inf gradients detected before unscaling", RuntimeWarning)
                    return False
                
                # Unscale gradients
                scaler.unscale_(optimizer)
                
                # Clip gradients with health check
                try:
                    grad_norm = gradient_clipping_with_health_check(
                        parameters, 
                        self.config.get('gradient_clip_norm', 1.0),
                        error_if_nonfinite=False
                    )
                except Exception as grad_error:
                    warnings.warn(f"Gradient clipping failed: {grad_error}", RuntimeWarning)
                    return False
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
            else:
                # Standard backward pass
                loss.backward()
                
                # Clip gradients with health check
                try:
                    grad_norm = gradient_clipping_with_health_check(
                        list(model.parameters()), 
                        self.config.get('gradient_clip_norm', 1.0),
                        error_if_nonfinite=False
                    )
                except Exception as grad_error:
                    warnings.warn(f"Gradient clipping failed: {grad_error}", RuntimeWarning)
                    return False
                
                # Optimizer step
                optimizer.step()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Backward pass failed: {e}", RuntimeWarning)
            return False
    
    def get_training_summary(self):
        """Get summary of training progress and error statistics."""
        if not self.training_history:
            return {"message": "No training history available"}
        
        recent_losses = self.training_history[-10:]  # Last 10 batches
        avg_losses = {}
        
        if recent_losses:
            for key in recent_losses[0].keys():
                values = [loss[key] for loss in recent_losses if key in loss]
                if values:
                    avg_losses[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "total_batches": len(self.training_history),
            "error_counts": dict(self.error_counts),
            "recent_average_losses": avg_losses,
            "last_loss": self.training_history[-1] if self.training_history else None
        }