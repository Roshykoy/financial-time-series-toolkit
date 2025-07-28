# src/cvae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.graph_encoder import NumberGraphEncoder
from src.temporal_context import TemporalContextEncoder

class GenerativeDecoder(nn.Module):
    """
    Decoder that generates lottery number combinations from latent representations
    and temporal context.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.context_dim = config['temporal_context_dim']
        self.num_numbers = config['num_lotto_numbers']
        self.decoder_hidden_dim = config['decoder_hidden_dim']
        
        # Input projection: combine latent code and temporal context
        self.input_projection = nn.Sequential(
            nn.Linear(self.latent_dim + self.context_dim, config['decoder_projection_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # Multi-layer decoder
        decoder_layers = []
        input_dim = config['decoder_projection_dim']
        
        for i in range(config['decoder_layers']):
            decoder_layers.extend([
                nn.Linear(input_dim, self.decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout'])
            ])
            input_dim = self.decoder_hidden_dim
        
        self.decoder_network = nn.Sequential(*decoder_layers)
        
        # Output heads for generating 6 numbers
        self.number_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.decoder_hidden_dim, config['number_generation_hidden']),
                nn.ReLU(),
                nn.Linear(config['number_generation_hidden'], self.num_numbers)
            )
            for _ in range(6)
        ])
        
        # Constraint enforcement network
        self.constraint_network = nn.Sequential(
            nn.Linear(6 * self.num_numbers, config['constraint_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['constraint_hidden_dim'], 6 * self.num_numbers)
        )
        
    def forward(self, z, context):
        """
        Generates lottery number combinations.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            context: Temporal context [batch_size, context_dim]
        
        Returns:
            logits: Raw logits for each position [batch_size, 6, num_numbers]
            probs: Probabilities after constraint enforcement [batch_size, 6, num_numbers]
        """
        batch_size = z.size(0)
        
        # Combine latent and context
        combined_input = torch.cat([z, context], dim=-1)
        
        # Project and decode
        projected = self.input_projection(combined_input)
        hidden = self.decoder_network(projected)
        
        # Generate logits for each number position
        position_logits = []
        for generator in self.number_generators:
            logits = generator(hidden)  # [batch_size, num_numbers]
            position_logits.append(logits)
        
        raw_logits = torch.stack(position_logits, dim=1)  # [batch_size, 6, num_numbers]
        
        # Apply constraint enforcement
        flattened_logits = raw_logits.view(batch_size, -1)
        constraint_adjustment = self.constraint_network(flattened_logits)
        adjusted_logits = (flattened_logits + constraint_adjustment).view(batch_size, 6, self.num_numbers)
        
        # Convert to probabilities
        probs = F.softmax(adjusted_logits, dim=-1)
        
        return adjusted_logits, probs
    
    def sample_combinations(self, z, context, temperature=1.0, top_k=None):
        """
        Samples valid lottery combinations ensuring no duplicates.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            context: Temporal context [batch_size, context_dim]
            temperature: Sampling temperature
            top_k: Optional top-k sampling
        
        Returns:
            combinations: Valid number combinations [batch_size, 6]
            log_probs: Log probabilities of sampled combinations
        """
        batch_size = z.size(0)
        device = z.device
        
        _, probs = self.forward(z, context)
        
        combinations = []
        log_probs = []
        
        for b in range(batch_size):
            selected_numbers = []
            combination_log_prob = 0.0
            available_mask = torch.ones(self.num_numbers, device=device)
            
            for pos in range(6):
                # Apply availability mask
                masked_probs = probs[b, pos] * available_mask
                masked_probs = masked_probs / masked_probs.sum()
                
                # Apply temperature
                if temperature != 1.0:
                    masked_probs = torch.pow(masked_probs, 1.0 / temperature)
                    masked_probs = masked_probs / masked_probs.sum()
                
                # Top-k sampling if specified
                if top_k is not None:
                    top_probs, top_indices = torch.topk(masked_probs, min(top_k, masked_probs.size(0)))
                    sample_idx = torch.multinomial(top_probs, 1).item()
                    selected_num = top_indices[sample_idx].item()
                else:
                    selected_num = torch.multinomial(masked_probs, 1).item()
                
                # Convert to 1-based indexing
                selected_numbers.append(selected_num + 1)
                combination_log_prob += torch.log(masked_probs[selected_num] + 1e-8)
                
                # Update availability mask
                available_mask[selected_num] = 0.0
            
            combinations.append(sorted(selected_numbers))
            log_probs.append(combination_log_prob)
        
        return torch.tensor(combinations, device=device), torch.stack(log_probs)

class ConditionalVAE(nn.Module):
    """
    Main Conditional Variational Autoencoder for lottery number generation.
    Enhanced with KL collapse prevention and improved numerical stability.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        
        # KL annealing parameters
        self.kl_annealing_epochs = config.get('kl_annealing_epochs', 10)
        self.kl_min_beta = config.get('kl_min_beta', 0.0)
        self.kl_max_beta = config.get('kl_max_beta', 1.0)
        self.current_beta = self.kl_min_beta
        
        # Encoder components
        self.graph_encoder = NumberGraphEncoder(config)
        self.temporal_encoder = TemporalContextEncoder(config)
        
        # Decoder
        self.decoder = GenerativeDecoder(config)
        
        # Prior network (learns data-dependent prior)
        self.prior_network = nn.Sequential(
            nn.Linear(config['temporal_context_dim'], config['prior_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['prior_hidden_dim'], self.latent_dim * 2)  # mean and logvar
        )
        
    def encode(self, number_sets, pair_counts):
        """Encode number combinations to latent space."""
        mu, logvar = self.graph_encoder(number_sets, pair_counts)
        return mu, logvar
    
    def get_temporal_context(self, df, current_indices):
            """Get temporal context for given indices - FIXED for device consistency."""
            contexts = []
            trend_features = []
            device = next(self.parameters()).device  # Get model device
            
            for idx in current_indices:
                sequence = self.temporal_encoder.prepare_sequence_data(df, idx)
                # Ensure sequence is on correct device (this is now handled in prepare_sequence_data)
                sequence = sequence.to(device)
                context, trends = self.temporal_encoder(sequence)
                contexts.append(context.squeeze(0))
                trend_features.append(trends.squeeze(0))
            
            return torch.stack(contexts), torch.stack(trend_features)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE with improved numerical stability."""
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        
        # Add small epsilon for numerical stability
        eps = torch.randn_like(std) 
        return mu + eps * (std + self.config.get('numerical_stability_eps', 1e-8))
    
    def get_prior_params(self, context):
        """Get context-dependent prior parameters with regularization."""
        prior_params = self.prior_network(context)
        mu_prior, logvar_prior = torch.chunk(prior_params, 2, dim=-1)
        
        # Regularize prior to prevent collapse
        # Encourage non-zero variance in prior
        logvar_prior = torch.clamp(logvar_prior, min=-5, max=5)
        
        # Add small regularization to prevent perfect posterior-prior matching
        prior_noise = torch.randn_like(mu_prior) * 0.01
        mu_prior = mu_prior + prior_noise
        
        return mu_prior, logvar_prior
    
    def forward(self, number_sets, pair_counts, temporal_sequences, current_indices):
        """
        Full forward pass for training.
        
        Args:
            number_sets: Batch of number combinations
            pair_counts: Pair frequency counts
            temporal_sequences: Pre-computed temporal sequences [batch_size, seq_len, 6]
            current_indices: Current draw indices
        
        Returns:
            reconstruction_logits: [batch_size, 6, num_numbers]
            mu, logvar: Posterior parameters
            mu_prior, logvar_prior: Prior parameters
            context: Temporal context
        """
        # Encode inputs
        mu, logvar = self.encode(number_sets, pair_counts)
        
        # Process pre-computed temporal sequences
        context, _ = self.temporal_encoder(temporal_sequences)
        
        # Sample from posterior with KL regularization
        z = self.reparameterize(mu, logvar)
        
        # Store KL components for analysis
        self._last_mu = mu.detach()
        self._last_logvar = logvar.detach()
        self._last_mu_prior = mu_prior.detach()
        self._last_logvar_prior = logvar_prior.detach()
        
        # Get context-dependent prior
        mu_prior, logvar_prior = self.get_prior_params(context)
        
        # Decode
        reconstruction_logits, _ = self.decoder(z, context)
        
        return reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context
    
    def generate(self, context, num_samples=1, temperature=1.0):
        """
        Generate new combinations given temporal context.
        
        Args:
            context: Temporal context [batch_size, context_dim]
            num_samples: Number of samples per context
            temperature: Sampling temperature
        
        Returns:
            generated_combinations: [batch_size * num_samples, 6]
            log_probs: Log probabilities of generated combinations
        """
        self.eval()
        with torch.no_grad():
            batch_size = context.size(0)
            device = context.device
            
            # Expand context for multiple samples
            if num_samples > 1:
                context_expanded = context.unsqueeze(1).expand(-1, num_samples, -1)
                context_expanded = context_expanded.contiguous().view(-1, context.size(-1))
            else:
                context_expanded = context
            
            # Sample from prior
            mu_prior, logvar_prior = self.get_prior_params(context_expanded)
            z = self.reparameterize(mu_prior, logvar_prior)
            
            # Generate combinations
            combinations, log_probs = self.decoder.sample_combinations(
                z, context_expanded, temperature=temperature
            )
            
            return combinations, log_probs