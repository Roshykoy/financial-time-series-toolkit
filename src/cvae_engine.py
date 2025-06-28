# src/cvae_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict

class CVAELossComputer:
    """Handles all loss computations for the CVAE training."""
    
    def __init__(self, config):
        self.config = config
        # Handle both device strings and torch.device objects
        device = config['device']
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)
        
    def reconstruction_loss(self, reconstruction_logits, target_combinations):
        """
        Computes reconstruction loss for generated vs actual combinations.
        
        Args:
            reconstruction_logits: [batch_size, 6, num_numbers]
            target_combinations: [batch_size, 6] - 1-based indices
        
        Returns:
            loss: Reconstruction loss
        """
        batch_size = target_combinations.size(0)
        
        # Convert to 0-based indices for cross-entropy
        targets_zero_based = target_combinations - 1
        
        # Compute cross-entropy loss for each position
        total_loss = 0
        for pos in range(6):
            pos_logits = reconstruction_logits[:, pos, :]  # [batch_size, num_numbers]
            pos_targets = targets_zero_based[:, pos]      # [batch_size]
            pos_loss = F.cross_entropy(pos_logits, pos_targets, reduction='mean')
            total_loss += pos_loss
        
        return total_loss / 6  # Average over positions
    
    def kl_divergence_loss(self, mu, logvar, mu_prior, logvar_prior):
        """
        Computes KL divergence between posterior and context-dependent prior.
        
        KL(q(z|x,c) || p(z|c)) where c is temporal context
        """
        # Standard KL divergence formula between two Gaussians
        var_ratio = torch.exp(logvar - logvar_prior)
        t1 = (mu - mu_prior).pow(2) / torch.exp(logvar_prior)
        t2 = var_ratio
        t3 = logvar_prior - logvar
        kl_div = 0.5 * torch.sum(t1 + t2 + t3 - 1, dim=-1)
        
        return kl_div.mean()
    
    def hard_contrastive_loss(self, model, positive_combinations, negative_combinations, 
                            pair_counts, temporal_sequences, current_indices):
        """
        Computes contrastive loss with hard negative mining.
        
        Args:
            model: CVAE model
            positive_combinations: Real winning combinations
            negative_combinations: Pool of negative samples
            pair_counts: Historical pair frequencies
            df: Historical data for temporal context
            current_indices: Current draw indices
        """
        batch_size = len(positive_combinations)
        device = self.device
        
        # Get embeddings for positive samples
        mu_pos, logvar_pos = model.encode(positive_combinations, pair_counts)
        z_pos = model.reparameterize(mu_pos, logvar_pos)
        
        # Sample hard negatives for each positive
        hard_negatives = self._mine_hard_negatives(
            model, positive_combinations, negative_combinations, 
            pair_counts, temporal_sequences, current_indices
        )
        
        # Get embeddings for hard negatives
        mu_neg, logvar_neg = model.encode(hard_negatives, pair_counts)
        z_neg = model.reparameterize(mu_neg, logvar_neg)
        
        # Compute contrastive loss in latent space
        pos_expanded = z_pos.unsqueeze(1)  # [batch, 1, latent_dim]
        neg_expanded = z_neg.view(batch_size, -1, z_neg.size(-1))  # [batch, num_neg, latent_dim]
        
        # Cosine similarity
        pos_norm = F.normalize(pos_expanded, dim=-1)
        neg_norm = F.normalize(neg_expanded, dim=-1)
        
        similarities = torch.bmm(pos_norm, neg_norm.transpose(1, 2)).squeeze(1)  # [batch, num_neg]
        
        # InfoNCE-style loss
        temperature = self.config['contrastive_temperature']
        exp_sim = torch.exp(similarities / temperature)
        
        # Positive similarity (self-similarity, should be high)
        pos_sim = torch.ones(batch_size, device=device)
        exp_pos = torch.exp(pos_sim / temperature)
        
        # Contrastive loss
        contrastive_loss = -torch.log(exp_pos / (exp_pos + exp_sim.sum(dim=-1)))
        
        return contrastive_loss.mean()
    
    def _mine_hard_negatives(self, model, positive_combinations, negative_pool, 
                           pair_counts, temporal_sequences, current_indices):
        """
        Mines hard negatives that are most confusing for the current model.
        """
        batch_size = len(positive_combinations)
        num_negatives = self.config['negative_samples']
        hard_ratio = self.config['hard_negative_ratio']
        num_hard = int(num_negatives * hard_ratio)
        num_random = num_negatives - num_hard
        
        hard_negatives = []
        
        with torch.no_grad():
            # Get positive embeddings
            mu_pos, logvar_pos = model.encode(positive_combinations, pair_counts)
            z_pos = model.reparameterize(mu_pos, logvar_pos)
            
            for i in range(batch_size):
                batch_hard_negatives = []
                
                # Sample candidate negatives
                candidate_negatives = random.sample(negative_pool, min(200, len(negative_pool)))
                
                if len(candidate_negatives) == 0:
                    # Fallback to random negatives
                    batch_negatives = random.sample(negative_pool, num_negatives)
                    hard_negatives.extend(batch_negatives)
                    continue
                
                # Get embeddings for candidates
                mu_neg_cand, logvar_neg_cand = model.encode(candidate_negatives, pair_counts)
                z_neg_cand = model.reparameterize(mu_neg_cand, logvar_neg_cand)
                
                # Compute similarities to positive
                pos_z = z_pos[i:i+1]  # [1, latent_dim]
                similarities = F.cosine_similarity(pos_z, z_neg_cand, dim=-1)
                
                # Select hardest negatives (most similar to positive)
                if len(candidate_negatives) >= num_hard:
                    _, hard_indices = torch.topk(similarities, num_hard, largest=True)
                    hard_samples = [candidate_negatives[idx] for idx in hard_indices.cpu().numpy()]
                else:
                    hard_samples = candidate_negatives
                
                # Add random negatives
                remaining_candidates = [neg for neg in candidate_negatives if neg not in hard_samples]
                if len(remaining_candidates) >= num_random:
                    random_samples = random.sample(remaining_candidates, num_random)
                else:
                    random_samples = remaining_candidates
                    # Fill remaining with random from pool
                    additional_needed = num_random - len(random_samples)
                    if additional_needed > 0:
                        additional_samples = random.sample(
                            [neg for neg in negative_pool if neg not in hard_samples + random_samples],
                            min(additional_needed, len(negative_pool) - len(hard_samples) - len(random_samples))
                        )
                        random_samples.extend(additional_samples)
                
                batch_negatives = hard_samples + random_samples
                hard_negatives.extend(batch_negatives)
        
        return hard_negatives
    
    def meta_learning_loss(self, meta_learner, number_sets, temporal_context, 
                          true_scores, predicted_ensemble_scores):
        """
        Loss for training the meta-learner to predict optimal ensemble weights.
        
        Args:
            meta_learner: Meta-learning model
            number_sets: Batch of number combinations
            temporal_context: Temporal context features
            true_scores: Ground truth ranking scores
            predicted_ensemble_scores: Scores from current ensemble
        """
        # This is a placeholder for meta-learning loss
        # In practice, this would involve learning from historical performance
        # of different ensemble configurations
        
        # For now, use MSE between predicted and true ranking
        mse_loss = F.mse_loss(predicted_ensemble_scores, true_scores)
        
        return mse_loss

def train_one_epoch_cvae(model, meta_learner, train_loader, optimizers, device, config, epoch):
    """
    Trains the CVAE model for one epoch with dual objectives.
    
    Args:
        model: CVAE model
        meta_learner: Meta-learning model for ensemble weights
        train_loader: Training data loader
        optimizers: Dict of optimizers for different components
        device: Training device
        config: Configuration dictionary
        epoch: Current epoch number
    
    Returns:
        loss_dict: Dictionary of various loss components
    """
    model.train()
    meta_learner.train()
    
    # Ensure all components are in training mode
    model.temporal_encoder.train()
    if hasattr(model, 'graph_encoder'):
        model.graph_encoder.train()
    
    # Create config copy with proper device handling for CVAELossComputer
    train_config = config.copy()
    if isinstance(device, torch.device):
        train_config['device'] = device
    else:
        train_config['device'] = torch.device(device)
    
    loss_computer = CVAELossComputer(train_config)
    epoch_losses = defaultdict(list)
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack batch
        positive_combinations = batch['positive_combinations']
        negative_pool = batch['negative_pool']
        pair_counts = batch['pair_counts']
        current_indices = batch['current_indices']
        temporal_sequences = batch['temporal_sequences']
        
        # Zero gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass through CVAE
        reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context = model(
            positive_combinations, pair_counts, temporal_sequences, current_indices
        )
        
        # Convert combinations to tensor for loss computation
        target_tensor = torch.tensor(positive_combinations, device=device)
        
        # Compute losses
        recon_loss = loss_computer.reconstruction_loss(reconstruction_logits, target_tensor)
        kl_loss = loss_computer.kl_divergence_loss(mu, logvar, mu_prior, logvar_prior)
        contrastive_loss = loss_computer.hard_contrastive_loss(
            model, positive_combinations, negative_pool, 
            pair_counts, temporal_sequences, current_indices
        )
        
        # Total CVAE loss
        cvae_loss = (config['reconstruction_weight'] * recon_loss + 
                    config['kl_weight'] * kl_loss + 
                    config['contrastive_weight'] * contrastive_loss)
        
        # Backward pass for CVAE
        cvae_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
        
        # Update CVAE
        optimizers['cvae'].step()
        
        # Meta-learner training (every few batches to reduce overhead)
        if batch_idx % 3 == 0:
            # Zero gradients for meta-learner
            optimizers['meta'].zero_grad()
            
            # Generate some scores for meta-learning
            with torch.no_grad():
                # Use detached temporal context to avoid gradient issues
                detached_context = temporal_context.detach()
                generated_combinations, _ = model.generate(detached_context, num_samples=1)
                generated_combinations = generated_combinations.cpu().numpy().tolist()
            
            # Mock scorer scores for meta-learning training
            mock_scores = {
                'generative': torch.randn(len(positive_combinations), device=device),
                'temporal': torch.randn(len(positive_combinations), device=device),
                'i_ching': torch.randn(len(positive_combinations), device=device)
            }
            
            # Meta-learner forward pass with detached context
            ensemble_weights, ensemble_scores, confidence = meta_learner(
                positive_combinations, detached_context, mock_scores
            )
            
            # Simple meta-learning loss (could be more sophisticated)
            meta_loss = F.mse_loss(ensemble_scores, torch.ones_like(ensemble_scores))
            meta_loss = config['meta_learning_weight'] * meta_loss
            
            # Backward pass for meta-learner
            meta_loss.backward()
            optimizers['meta'].step()
            
            epoch_losses['meta_loss'].append(meta_loss.item())
        
        # Record losses
        epoch_losses['reconstruction_loss'].append(recon_loss.item())
        epoch_losses['kl_loss'].append(kl_loss.item())
        epoch_losses['contrastive_loss'].append(contrastive_loss.item())
        epoch_losses['total_cvae_loss'].append(cvae_loss.item())
        
        # Logging
        if batch_idx % config['log_interval'] == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}: '
                  f'Recon: {recon_loss.item():.4f}, '
                  f'KL: {kl_loss.item():.4f}, '
                  f'Contrastive: {contrastive_loss.item():.4f}')
    
    # Return average losses
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    return avg_losses

def evaluate_cvae(model, meta_learner, val_loader, device, config):
    """
    Evaluates the CVAE model on validation data.
    """
    model.eval()
    meta_learner.eval()
    
    # Create config copy with proper device handling for CVAELossComputer
    eval_config = config.copy()
    if isinstance(device, torch.device):
        eval_config['device'] = device
    else:
        eval_config['device'] = torch.device(device)
    
    loss_computer = CVAELossComputer(eval_config)
    val_losses = defaultdict(list)
    
    with torch.no_grad():
        for batch in val_loader:
            positive_combinations = batch['positive_combinations']
            negative_pool = batch['negative_pool']
            pair_counts = batch['pair_counts']
            current_indices = batch['current_indices']
            temporal_sequences = batch['temporal_sequences']
            
            # Forward pass
            reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context = model(
                positive_combinations, pair_counts, temporal_sequences, current_indices
            )
            
            target_tensor = torch.tensor(positive_combinations, device=device)
            
            # Compute losses
            recon_loss = loss_computer.reconstruction_loss(reconstruction_logits, target_tensor)
            kl_loss = loss_computer.kl_divergence_loss(mu, logvar, mu_prior, logvar_prior)
            
            val_losses['reconstruction_loss'].append(recon_loss.item())
            val_losses['kl_loss'].append(kl_loss.item())
    
    avg_val_losses = {key: np.mean(values) for key, values in val_losses.items()}
    
    # Return main validation loss and full metrics to match expected signature
    total_val_loss = avg_val_losses.get('reconstruction_loss', 0.0) + avg_val_losses.get('kl_loss', 0.0)
    
    return total_val_loss, avg_val_losses