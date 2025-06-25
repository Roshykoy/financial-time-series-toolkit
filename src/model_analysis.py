# src/model_analysis.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

def analyze_model_gradients(model, data_loader, device, max_batches=5):
    """
    Analyzes gradient flow through the model to detect training issues.
    """
    print("Analyzing gradient flow...")
    
    gradient_stats = defaultdict(list)
    
    model.train()
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
            
        # Forward pass
        positive_combinations = batch['positive_combinations']
        pair_counts = batch['pair_counts']
        current_indices = batch['current_indices']
        
        # Dummy df for this analysis
        df = pd.DataFrame()  # This would need to be passed properly
        
        try:
            reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context = model(
                positive_combinations, pair_counts, df, current_indices
            )
            
            # Compute loss
            target_tensor = torch.tensor(positive_combinations, device=device)
            loss = torch.nn.functional.cross_entropy(
                reconstruction_logits.view(-1, reconstruction_logits.size(-1)),
                (target_tensor - 1).view(-1)
            )
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Collect gradient statistics
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    param_norm = param.data.norm().item()
                    gradient_stats[name].append({
                        'grad_norm': grad_norm,
                        'param_norm': param_norm,
                        'grad_to_param_ratio': grad_norm / (param_norm + 1e-8)
                    })
                else:
                    gradient_stats[name].append({
                        'grad_norm': 0.0,
                        'param_norm': param.data.norm().item(),
                        'grad_to_param_ratio': 0.0
                    })
        except Exception as e:
            print(f"Error in gradient analysis: {e}")
            break
    
    # Summarize gradient statistics
    print("\nGradient Flow Summary:")
    print("-" * 50)
    
    for name, stats_list in gradient_stats.items():
        if len(stats_list) > 0:
            avg_grad_norm = np.mean([s['grad_norm'] for s in stats_list])
            avg_param_norm = np.mean([s['param_norm'] for s in stats_list])
            avg_ratio = np.mean([s['grad_to_param_ratio'] for s in stats_list])
            
            # Detect potential issues
            status = "✓"
            if avg_grad_norm < 1e-7:
                status = "⚠️ Very small gradients"
            elif avg_grad_norm > 100:
                status = "⚠️ Very large gradients"
            elif avg_ratio > 1.0:
                status = "⚠️ Large gradient-to-parameter ratio"
            
            print(f"{name:40} | Grad: {avg_grad_norm:.2e} | Param: {avg_param_norm:.2e} | Ratio: {avg_ratio:.2e} | {status}")
    
    return gradient_stats

def analyze_latent_space_interpolation(model, combination1, combination2, pair_counts, steps=10):
    """
    Analyzes interpolation in latent space between two combinations.
    """
    print(f"Analyzing latent space interpolation between {combination1} and {combination2}")
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Encode both combinations
        mu1, logvar1 = model.encode([combination1], pair_counts)
        mu2, logvar2 = model.encode([combination2], pair_counts)
        
        z1 = model.reparameterize(mu1, logvar1)
        z2 = model.reparameterize(mu2, logvar2)
        
        # Create interpolation
        interpolation_results = []
        alphas = np.linspace(0, 1, steps)
        
        # Need temporal context for decoder
        dummy_context = torch.zeros(1, model.config['temporal_context_dim'], device=device)
        
        for alpha in alphas:
            # Linear interpolation in latent space
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode interpolated latent code
            reconstruction_logits, probs = model.decoder(z_interp, dummy_context)
            
            # Get most likely combination
            reconstructed_combination = []
            for pos in range(6):
                most_likely = torch.argmax(reconstruction_logits[0, pos, :]).item() + 1
                reconstructed_combination.append(most_likely)
            
            interpolation_results.append({
                'alpha': alpha,
                'combination': reconstructed_combination,
                'latent_distance_to_start': torch.norm(z_interp - z1).item(),
                'latent_distance_to_end': torch.norm(z_interp - z2).item()
            })
        
        # Print results
        print("\nInterpolation Results:")
        print("-" * 60)
        for result in interpolation_results:
            print(f"α={result['alpha']:.2f}: {result['combination']} "
                  f"(dist_start: {result['latent_distance_to_start']:.3f}, "
                  f"dist_end: {result['latent_distance_to_end']:.3f})")
    
    return interpolation_results

def analyze_attention_weights(model, combination, pair_counts, save_path="outputs/attention_analysis.png"):
    """
    Analyzes attention weights in the graph encoder and temporal encoder.
    """
    print("Analyzing attention weights...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Hook to capture attention weights
    attention_weights = {}
    
    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                attention_weights[name] = output.detach().cpu()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower():
            hook = module.register_forward_hook(attention_hook(name))
            hooks.append(hook)
    
    try:
        with torch.no_grad():
            # Forward pass to capture attention
            mu, logvar = model.encode([combination], pair_counts)
            
        # Create visualization if attention weights were captured
        if attention_weights:
            fig, axes = plt.subplots(len(attention_weights), 1, figsize=(12, 4*len(attention_weights)))
            if len(attention_weights) == 1:
                axes = [axes]
            
            for idx, (name, weights) in enumerate(attention_weights.items()):
                if len(weights.shape) >= 2:
                    # Take first head if multi-head attention
                    if len(weights.shape) == 4:  # [batch, heads, seq, seq]
                        weights_viz = weights[0, 0]
                    elif len(weights.shape) == 3:  # [batch, seq, seq]
                        weights_viz = weights[0]
                    else:
                        weights_viz = weights
                    
                    im = axes[idx].imshow(weights_viz.numpy(), cmap='Blues', aspect='auto')
                    axes[idx].set_title(f'Attention Weights: {name}')
                    axes[idx].set_xlabel('Key Position')
                    axes[idx].set_ylabel('Query Position')
                    plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Attention analysis saved to: {save_path}")
        else:
            print("No attention weights captured - model may not have attention mechanisms")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return attention_weights

def analyze_reconstruction_quality(model, test_combinations, pair_counts, df, indices):
    """
    Analyzes reconstruction quality for specific test combinations.
    """
    print("Analyzing reconstruction quality...")
    
    model.eval()
    device = next(model.parameters()).device
    
    reconstruction_analysis = []
    
    with torch.no_grad():
        for i, (combination, index) in enumerate(zip(test_combinations, indices)):
            print(f"Analyzing combination {i+1}: {combination}")
            
            # Forward pass
            reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context = model(
                [combination], pair_counts, df, [index]
            )
            
            # Get reconstruction probabilities
            reconstruction_probs = torch.softmax(reconstruction_logits, dim=-1)
            
            # Analyze each position
            position_analysis = []
            for pos in range(6):
                target_num = combination[pos] - 1  # Convert to 0-based
                prob_at_target = reconstruction_probs[0, pos, target_num].item()
                
                # Get top 3 predictions
                top_probs, top_indices = torch.topk(reconstruction_probs[0, pos], 3)
                top_predictions = [(idx.item() + 1, prob.item()) for idx, prob in zip(top_indices, top_probs)]
                
                position_analysis.append({
                    'position': pos,
                    'target': combination[pos],
                    'prob_at_target': prob_at_target,
                    'top_predictions': top_predictions,
                    'rank_of_target': (reconstruction_probs[0, pos] >= prob_at_target).sum().item()
                })
            
            # Overall combination analysis
            total_likelihood = 1.0
            for pos in range(6):
                total_likelihood *= reconstruction_probs[0, pos, combination[pos] - 1].item()
            
            reconstruction_analysis.append({
                'combination': combination,
                'total_likelihood': total_likelihood,
                'log_likelihood': np.log(total_likelihood + 1e-8),
                'position_analysis': position_analysis,
                'latent_stats': {
                    'mu_mean': mu.mean().item(),
                    'mu_std': mu.std().item(),
                    'logvar_mean': logvar.mean().item(),
                    'logvar_std': logvar.std().item()
                }
            })
    
    # Print summary
    print("\nReconstruction Quality Summary:")
    print("-" * 50)
    
    for analysis in reconstruction_analysis:
        combo = analysis['combination']
        likelihood = analysis['total_likelihood']
        log_likelihood = analysis['log_likelihood']
        
        print(f"Combination: {combo}")
        print(f"  Total likelihood: {likelihood:.2e}")
        print(f"  Log likelihood: {log_likelihood:.3f}")
        
        avg_prob_at_target = np.mean([pos['prob_at_target'] for pos in analysis['position_analysis']])
        avg_rank = np.mean([pos['rank_of_target'] for pos in analysis['position_analysis']])
        
        print(f"  Avg prob at target: {avg_prob_at_target:.3f}")
        print(f"  Avg rank of target: {avg_rank:.1f}")
        print()
    
    return reconstruction_analysis

def generate_model_summary(model, input_shapes=None):
    """
    Generates a comprehensive summary of the model architecture.
    """
    print("Model Architecture Summary")
    print("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer Name':<30} {'Output Shape':<20} {'Param Count':<15}")
    print("-" * 65)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += param_count
            trainable_params += trainable_count
            
            # Try to infer output shape (simplified)
            output_shape = "Unknown"
            if hasattr(module, 'weight') and module.weight is not None:
                if len(module.weight.shape) == 2:  # Linear layer
                    output_shape = f"(..., {module.weight.shape[0]})"
                elif len(module.weight.shape) == 4:  # Conv layer
                    output_shape = f"(..., {module.weight.shape[0]}, H, W)"
            
            print(f"{name:<30} {output_shape:<20} {param_count:<15,}")
    
    print("-" * 65)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimation
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"Estimated parameter memory: {param_size_mb:.2f} MB")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size_mb
    }

def debug_training_step(model, batch, device, config):
    """
    Debugs a single training step to identify potential issues.
    """
    print("Debugging training step...")
    
    model.train()
    
    try:
        # Unpack batch
        positive_combinations = batch['positive_combinations']
        pair_counts = batch['pair_counts']
        current_indices = batch['current_indices']
        
        print(f"Batch size: {len(positive_combinations)}")
        print(f"Sample combination: {positive_combinations[0] if positive_combinations else 'None'}")
        
        # Check for NaN or invalid values
        for i, combo in enumerate(positive_combinations):
            if not all(1 <= num <= config['num_lotto_numbers'] for num in combo):
                print(f"Warning: Invalid combination at index {i}: {combo}")
            if len(set(combo)) != 6:
                print(f"Warning: Duplicate numbers at index {i}: {combo}")
        
        # Dummy df for this debug
        df = pd.DataFrame()
        
        # Forward pass
        print("Running forward pass...")
        reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context = model(
            positive_combinations, pair_counts, df, current_indices
        )
        
        print(f"Reconstruction logits shape: {reconstruction_logits.shape}")
        print(f"Mu shape: {mu.shape}, range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
        print(f"Logvar shape: {logvar.shape}, range: [{logvar.min().item():.3f}, {logvar.max().item():.3f}]")
        print(f"Context shape: {context.shape}")
        
        # Check for NaN or inf
        if torch.isnan(reconstruction_logits).any():
            print("Warning: NaN detected in reconstruction logits")
        if torch.isinf(reconstruction_logits).any():
            print("Warning: Inf detected in reconstruction logits")
        
        # Compute loss
        target_tensor = torch.tensor(positive_combinations, device=device) - 1  # 0-based
        
        recon_loss = 0
        for pos in range(6):
            pos_loss = torch.nn.functional.cross_entropy(
                reconstruction_logits[:, pos, :], target_tensor[:, pos]
            )
            recon_loss += pos_loss
        recon_loss /= 6
        
        # KL loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
        
        total_loss = recon_loss + 0.1 * kl_loss
        
        print(f"Reconstruction loss: {recon_loss.item():.6f}")
        print(f"KL loss: {kl_loss.item():.6f}")
        print(f"Total loss: {total_loss.item():.6f}")
        
        # Backward pass
        print("Running backward pass...")
        model.zero_grad()
        total_loss.backward()
        
        # Check gradients
        grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm()
                grad_norm += param_grad_norm ** 2
                param_count += 1
        
        grad_norm = grad_norm ** 0.5
        print(f"Total gradient norm: {grad_norm.item():.6f}")
        print(f"Parameters with gradients: {param_count}")
        
        print("Training step debug completed successfully!")
        
    except Exception as e:
        print(f"Error in training step: {e}")
        import traceback
        traceback.print_exc()

def save_model_checkpoint_with_metadata(model, meta_learner, optimizer, epoch, loss, save_path):
    """
    Saves a model checkpoint with comprehensive metadata.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'meta_learner_state_dict': meta_learner.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model.config,
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_summary': generate_model_summary(model)
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")
    
    return checkpoint