# src/training_pipeline.py
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Import CVAE components
from src.config import CONFIG
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.cvae_engine import train_one_epoch_cvae, evaluate_cvae
from src.cvae_data_loader import create_cvae_data_loaders, CVAEBatch
from src.feature_engineering import FeatureEngineer

# Legacy imports for compatibility
from src.temporal_scorer import TemporalScorer
from src.i_ching_scorer import IChingScorer

# Add these imports at the top of src/training_pipeline.py
from src.debug_utils import ModelDebugger, debug_training_step, quick_model_test

# Add this function after the existing imports
def debug_training_setup(cvae_model, meta_learner, config, device):
    """Run debugging checks before training starts."""
    print("🔍 Running pre-training debugging checks...")
    
    # Test CVAE model
    print("Testing CVAE model...")
    cvae_debugger = ModelDebugger(cvae_model, device, verbose=False)
    
    def create_dummy_inputs():
        batch_size = 2
        positive_combinations = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        pair_counts = {(i, j): 1 for i in range(1, 50) for j in range(i+1, 50)}
        
        # Create minimal dummy dataframe
        import pandas as pd
        import numpy as np
        dummy_data = []
        for i in range(50):
            row = [i, f"2024-01-{i+1:02d}"] + list(np.random.choice(range(1, 50), 7, replace=False)) + [0]*25
            dummy_data.append(row)
        
        col_names = ['Draw', 'Date'] + [f'Winning_Num_{i}' for i in range(1, 8)] + [f'Col_{i}' for i in range(25)]
        df = pd.DataFrame(dummy_data, columns=col_names)
        current_indices = [10, 20]
        
        return positive_combinations, pair_counts, df, current_indices
    
    def dummy_loss(outputs):
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
            return outputs[0].mean() if isinstance(outputs[0], torch.Tensor) else torch.tensor(0.0, device=device)
        return torch.tensor(0.0, device=device)
    
    cvae_success = cvae_debugger.comprehensive_check(create_dummy_inputs, dummy_loss)
    
    # Test Meta-learner
    print("Testing Meta-learner...")
    meta_debugger = ModelDebugger(meta_learner, device, verbose=False)
    
    def create_meta_inputs():
        number_sets = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        temporal_context = torch.randn(2, config['temporal_context_dim'], device=device)
        scorer_scores = {
            'generative': torch.randn(2, device=device),
            'temporal': torch.randn(2, device=device),
            'i_ching': torch.randn(2, device=device)
        }
        return number_sets, temporal_context, scorer_scores
    
    def meta_loss(outputs):
        return outputs[1].mean() if len(outputs) > 1 else torch.tensor(0.0, device=device)
    
    meta_success = meta_debugger.comprehensive_check(create_meta_inputs, meta_loss)
    
    if cvae_success and meta_success:
        print("✅ All pre-training checks passed!")
        return True
    else:
        print("❌ Some pre-training checks failed. Consider running test_model_debug.py")
        return False
    
class EMAModel:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (self.decay * self.shadow[name] + 
                                   (1 - self.decay) * param.data)
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def setup_optimizers_and_schedulers(cvae_model, meta_learner, config):
    """Sets up optimizers and learning rate schedulers."""
    
    # CVAE optimizer
    cvae_optimizer = optim.AdamW(
        cvae_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Meta-learner optimizer
    meta_optimizer = optim.AdamW(
        meta_learner.parameters(),
        lr=config['learning_rate'] * 0.5,  # Slightly lower LR for meta-learner
        weight_decay=config['weight_decay']
    )
    
    optimizers = {
        'cvae': cvae_optimizer,
        'meta': meta_optimizer
    }
    
    # Learning rate schedulers
    schedulers = {}
    if config['use_scheduler']:
        if config['scheduler_type'] == 'cosine':
            schedulers['cvae'] = optim.lr_scheduler.CosineAnnealingLR(
                cvae_optimizer, T_max=config['epochs']
            )
            schedulers['meta'] = optim.lr_scheduler.CosineAnnealingLR(
                meta_optimizer, T_max=config['epochs']
            )
        elif config['scheduler_type'] == 'step':
            schedulers['cvae'] = optim.lr_scheduler.StepLR(
                cvae_optimizer, step_size=config['epochs']//3, gamma=0.5
            )
            schedulers['meta'] = optim.lr_scheduler.StepLR(
                meta_optimizer, step_size=config['epochs']//3, gamma=0.5
            )
    
    return optimizers, schedulers

def save_training_plots(train_losses, val_losses, epoch, save_dir):
    """Saves training progress plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reconstruction loss
    axes[0, 0].plot(train_losses['reconstruction_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(val_losses['reconstruction_loss'], label='Validation', alpha=0.7)
    axes[0, 0].set_title('Reconstruction Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # KL Divergence loss
    axes[0, 1].plot(train_losses['kl_loss'], label='Train', alpha=0.7)
    axes[0, 1].plot(val_losses['kl_loss'], label='Validation', alpha=0.7)
    axes[0, 1].set_title('KL Divergence Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Contrastive loss
    if 'contrastive_loss' in train_losses:
        axes[1, 0].plot(train_losses['contrastive_loss'], label='Train', alpha=0.7)
        axes[1, 0].set_title('Contrastive Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Total loss
    axes[1, 1].plot(train_losses['total_cvae_loss'], label='Train Total', alpha=0.7)
    axes[1, 1].set_title('Total CVAE Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_progress_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()

def run_training():
    """
    Main training pipeline for the CVAE-based lottery prediction system.
    """
    print("--- Starting CVAE Training Pipeline ---")
    print("Architecture: Conditional VAE + Graph Encoder + Temporal Context + Meta-Learner")
    
    device = torch.device(CONFIG['device'])
    print(f"Training on device: {device}")
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    col_names = [
        'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
        'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
        'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
        '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
        'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
        'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
        'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
    ]
    
    df = pd.read_csv(CONFIG["data_path"], header=None, skiprows=33, names=col_names)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} historical draws")
    
    # Fit feature engineer (still needed for some components)
    print("Fitting feature engineer...")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)
    
    # Create data loaders
    print("Creating CVAE data loaders...")
    train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, CONFIG)
    
    # Initialize models
    print("Initializing CVAE and Meta-Learner models...")
    cvae_model = ConditionalVAE(CONFIG).to(device)
    meta_learner = AttentionMetaLearner(CONFIG).to(device)
    
    # Print model sizes
    cvae_params = sum(p.numel() for p in cvae_model.parameters() if p.requires_grad)
    meta_params = sum(p.numel() for p in meta_learner.parameters() if p.requires_grad)
    print(f"CVAE Model parameters: {cvae_params:,}")
    print(f"Meta-Learner parameters: {meta_params:,}")
    print(f"Total parameters: {cvae_params + meta_params:,}")
    
    # Setup optimizers and schedulers
    optimizers, schedulers = setup_optimizers_and_schedulers(cvae_model, meta_learner, CONFIG)
    
    # Initialize EMA for stable inference
    if CONFIG['ema_decay'] > 0:
        ema_cvae = EMAModel(cvae_model, decay=CONFIG['ema_decay'])
        ema_meta = EMAModel(meta_learner, decay=CONFIG['ema_decay'])
    else:
        ema_cvae = ema_meta = None
    
    # Training history
    train_history = {
        'reconstruction_loss': [],
        'kl_loss': [],
        'contrastive_loss': [],
        'total_cvae_loss': [],
        'meta_loss': []
    }
    
    val_history = {
        'reconstruction_loss': [],
        'kl_loss': []
    }
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if CONFIG['use_mixed_precision'] and device.type == 'cuda' else None    

    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 40)
        
        # Training phase
        # Note: We need to modify the training function to work with our data loader
        epoch_train_losses = train_one_epoch_cvae_modified(
            cvae_model, meta_learner, train_loader, optimizers, device, CONFIG, epoch, df, scaler
        )
        
        # Validation phase
        epoch_val_losses = evaluate_cvae_modified(
            cvae_model, meta_learner, val_loader, device, CONFIG, df
        )
        
        # Update EMA
        if ema_cvae:
            ema_cvae.update()
            ema_meta.update()
        
        # Update learning rate schedulers
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Record losses
        for key, value in epoch_train_losses.items():
            if key in train_history:
                train_history[key].append(value)
        
        for key, value in epoch_val_losses.items():
            if key in val_history:
                val_history[key].append(value)
        
        # Print epoch summary
        print(f"Train - Recon: {epoch_train_losses['reconstruction_loss']:.4f}, "
              f"KL: {epoch_train_losses['kl_loss']:.4f}, "
              f"Contrastive: {epoch_train_losses['contrastive_loss']:.4f}")
        print(f"Val   - Recon: {epoch_val_losses['reconstruction_loss']:.4f}, "
              f"KL: {epoch_val_losses['kl_loss']:.4f}")
        
        # Save best model
        current_val_loss = epoch_val_losses['reconstruction_loss'] + epoch_val_losses['kl_loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            print(f"New best validation loss: {best_val_loss:.4f} - Saving models...")
            
            # Apply EMA for saving
            if ema_cvae:
                ema_cvae.apply_shadow()
                ema_meta.apply_shadow()
            
            # Save models with complete data (config + state_dict)
            os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
            
            # Save CVAE with config for proper loading
            cvae_save_data = {
                'cvae_state_dict': cvae_model.state_dict(),
                'meta_learner_state_dict': meta_learner.state_dict(),
                'config': CONFIG.copy(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss if 'best_val_loss' in locals() else 0.0
            }
            torch.save(cvae_save_data, CONFIG["model_save_path"])
            
            # Save meta-learner separately for compatibility
            torch.save(meta_learner.state_dict(), CONFIG["meta_learner_save_path"])
            
            # Restore original parameters
            if ema_cvae:
                ema_cvae.restore()
                ema_meta.restore()
        
        # Save training plots
        if CONFIG['plot_latent_space'] and (epoch + 1) % CONFIG['save_interval'] == 0:
            save_training_plots(train_history, val_history, epoch + 1, "outputs/training_plots")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'cvae_state_dict': cvae_model.state_dict(),
                'meta_state_dict': meta_learner.state_dict(),
                'cvae_optimizer': optimizers['cvae'].state_dict(),
                'meta_optimizer': optimizers['meta'].state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'config': CONFIG
            }
            torch.save(checkpoint, f"models/checkpoint_epoch_{epoch+1}.pth")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    
    # Save final artifacts
    print("Saving final artifacts...")
    joblib.dump(feature_engineer, CONFIG["feature_engineer_path"])
    
    # Generate and save sample combinations
    if CONFIG['save_generation_samples']:
        print("Generating sample combinations...")
        generate_sample_combinations(cvae_model, meta_learner, df, feature_engineer, CONFIG)
    
    print(f"Models saved to: {CONFIG['model_save_path']}")
    print(f"Meta-learner saved to: {CONFIG['meta_learner_save_path']}")
    print(f"Feature engineer saved to: {CONFIG['feature_engineer_path']}")

# Modified training functions to work with our data structure
def train_one_epoch_cvae_debug(model, meta_learner, train_loader, optimizers, device, config, epoch, df, scaler=None):
    """Enhanced training function with debugging."""
    from src.cvae_engine import CVAELossComputer
    from collections import defaultdict
    
    model.train()
    meta_learner.train()
    
    loss_computer = CVAELossComputer(config)
    epoch_losses = defaultdict(list)
    
    # Debug first batch extensively
    for batch_idx, batch_dict in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        
        if batch_idx == 0:  # Debug first batch of each epoch
            print(f"🔍 Debugging first batch of epoch {epoch+1}...")
            
            # Check batch health
            print(f"Batch keys: {batch_dict.keys()}")
            print(f"Positive combinations count: {len(batch_dict['positive_combinations'])}")
            print(f"Sample combination: {batch_dict['positive_combinations'][0] if batch_dict['positive_combinations'] else 'None'}")
        
        try:
            with debug_training_step(model, batch_dict, device, f"epoch_{epoch+1}_batch_{batch_idx}") as debugger:
                # Unpack batch
                positive_combinations = batch_dict['positive_combinations']
                negative_pool = batch_dict['negative_pool']
                pair_counts = batch_dict['pair_counts']
                current_indices = batch_dict['current_indices']
                temporal_sequences = batch_dict['temporal_sequences']
                
                # Validate combinations
                for i, combo in enumerate(positive_combinations):
                    if not all(1 <= num <= config['num_lotto_numbers'] for num in combo):
                        raise ValueError(f"Invalid combination at index {i}: {combo}")
                    if len(set(combo)) != 6:
                        raise ValueError(f"Duplicate numbers in combination {i}: {combo}")
                
                # Zero gradients
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                
                # Forward pass
                reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context = model(
                    positive_combinations, pair_counts, temporal_sequences, current_indices
                )
                
                # Check tensor health after forward pass
                if batch_idx == 0:
                    debugger.check_tensor_health(reconstruction_logits, "reconstruction_logits")
                    debugger.check_tensor_health(mu, "mu")
                    debugger.check_tensor_health(logvar, "logvar")
                    debugger.check_tensor_health(temporal_context, "temporal_context")
                
                # Convert combinations to tensor for loss computation
                target_tensor = torch.tensor(positive_combinations, device=device)
                
                # Compute losses
                recon_loss = loss_computer.reconstruction_loss(reconstruction_logits, target_tensor)
                kl_loss = loss_computer.kl_divergence_loss(mu, logvar, mu_prior, logvar_prior)
                contrastive_loss = loss_computer.hard_contrastive_loss(
                    model, positive_combinations, negative_pool, 
                    pair_counts, df, current_indices
                )
                
                # Check loss health
                if batch_idx == 0:
                    debugger.check_tensor_health(recon_loss, "recon_loss")
                    debugger.check_tensor_health(kl_loss, "kl_loss")
                    debugger.check_tensor_health(contrastive_loss, "contrastive_loss")
                
                # Total CVAE loss
                cvae_loss = (config['reconstruction_weight'] * recon_loss + 
                            config['kl_weight'] * kl_loss + 
                            config['contrastive_weight'] * contrastive_loss)
                
                # Backward pass with debugging
                if scaler:
                    scaler.scale(cvae_loss).backward(retain_graph=True)
                    scaler.unscale_(optimizers['cvae'])
                    
                    # Check gradients after unscaling
                    if batch_idx == 0:
                        total_grad_norm = 0
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param_grad_norm = param.grad.data.norm()
                                total_grad_norm += param_grad_norm ** 2
                        print(f"Total gradient norm before clipping: {(total_grad_norm ** 0.5):.6f}")
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                    scaler.step(optimizers['cvae'])
                    scaler.update()
                else:
                    cvae_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                    optimizers['cvae'].step()
                
                # Record losses
                epoch_losses['reconstruction_loss'].append(recon_loss.item())
                epoch_losses['kl_loss'].append(kl_loss.item())
                epoch_losses['contrastive_loss'].append(contrastive_loss.item())
                epoch_losses['total_cvae_loss'].append(cvae_loss.item())
                
                # Check for loss explosion
                if cvae_loss.item() > 1000:
                    print(f"⚠️  High loss detected: {cvae_loss.item():.2f}")
                
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            print(f"Batch info: positive_combinations={len(batch_dict.get('positive_combinations', []))}")
            
            # Emergency cleanup
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Continue with next batch or re-raise for critical errors
            if "CUDA out of memory" in str(e) or "device" in str(e).lower():
                raise  # Critical errors should stop training
            else:
                print("Continuing with next batch...")
                continue
    
    # Return average losses
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    return avg_losses

# Replace the train_one_epoch_cvae_modified function in src/training_pipeline.py
# with this fixed version that handles mixed precision overflow

def train_one_epoch_cvae_modified(model, meta_learner, train_loader, optimizers, device, config, epoch, df, scaler=None):
    """
    Enhanced training function for CVAE with meta-learner - FIXED for mixed precision overflow.
    
    Args:
        model: CVAE model
        meta_learner: Meta-learner model  
        train_loader: Training data loader
        optimizers: Dictionary of optimizers
        device: Training device
        config: Configuration dictionary
        epoch: Current epoch number
        df: Historical data DataFrame
        scaler: Mixed precision scaler (optional)
    
    Returns:
        epoch_losses: Dictionary of average losses
    """
    from src.cvae_engine import CVAELossComputer
    
    model.train()
    meta_learner.train()
    
    # Initialize loss computer
    loss_computer = CVAELossComputer(config)
    epoch_losses = defaultdict(list)
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    
    # Track mixed precision overflow occurrences
    overflow_count = 0
    successful_batches = 0
    
    for batch_idx, batch_dict in enumerate(progress_bar):
        try:
            # Extract batch data
            positive_combinations = batch_dict['positive_combinations']
            negative_pool = batch_dict['negative_pool']
            pair_counts = batch_dict['pair_counts']
            current_indices = batch_dict['current_indices']
            temporal_sequences = batch_dict['temporal_sequences']
            
            # Validate combinations
            for i, combo in enumerate(positive_combinations):
                if not all(1 <= num <= config['num_lotto_numbers'] for num in combo):
                    raise ValueError(f"Invalid combination at index {i}: {combo}")
                if len(set(combo)) != 6:
                    raise ValueError(f"Duplicate numbers in combination {i}: {combo}")
            
            # Zero gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            # Forward pass with mixed precision handling
            if scaler:
                # Mixed precision training with overflow detection
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    try:
                        reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context = model(
                            positive_combinations, pair_counts, temporal_sequences, current_indices
                        )
                        
                        # Check for overflow in model outputs BEFORE loss computation
                        outputs_to_check = [reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context]
                        has_overflow = False
                        
                        for i, output in enumerate(outputs_to_check):
                            if torch.isinf(output).any() or torch.isnan(output).any():
                                has_overflow = True
                                break
                            # Check for values too large for fp16
                            if output.dtype == torch.float16 and output.abs().max() > 65504:  # fp16 max value
                                has_overflow = True
                                break
                        
                        if has_overflow:
                            raise OverflowError("Model output overflow detected")
                        
                        # Convert combinations to tensor for loss computation
                        target_tensor = torch.tensor(positive_combinations, device=device)
                        
                        # Compute losses with overflow checking
                        recon_loss = loss_computer.reconstruction_loss(reconstruction_logits, target_tensor)
                        kl_loss = loss_computer.kl_divergence_loss(mu, logvar, mu_prior, logvar_prior)
                        contrastive_loss = loss_computer.hard_contrastive_loss(
                            model, positive_combinations, negative_pool, 
                            pair_counts, df, current_indices
                        )
                        
                        # Check losses for overflow
                        losses_to_check = [recon_loss, kl_loss, contrastive_loss]
                        for loss_tensor in losses_to_check:
                            if torch.isinf(loss_tensor).any() or torch.isnan(loss_tensor).any():
                                raise OverflowError("Loss overflow detected")
                        
                        # Total CVAE loss with clamping to prevent overflow
                        cvae_loss = (config['reconstruction_weight'] * torch.clamp(recon_loss, max=100) + 
                                    config['kl_weight'] * torch.clamp(kl_loss, max=50) + 
                                    config['contrastive_weight'] * torch.clamp(contrastive_loss, max=100))
                        
                        # Final overflow check
                        if torch.isinf(cvae_loss).any() or torch.isnan(cvae_loss).any():
                            raise OverflowError("Final loss overflow")
                        
                    except (OverflowError, RuntimeError) as overflow_error:
                        if "overflow" in str(overflow_error).lower() or "half" in str(overflow_error).lower():
                            overflow_count += 1
                            if overflow_count <= 5:  # Only warn for first few overflows
                                print(f"⚠️  Mixed precision overflow in batch {batch_idx} (#{overflow_count})")
                            
                            # Skip this batch gracefully
                            continue
                        else:
                            raise  # Re-raise non-overflow errors
                
                # Backward pass with mixed precision
                try:
                    scaler.scale(cvae_loss).backward(retain_graph=True)
                    
                    # Check if scaler detected overflow
                    scaler.unscale_(optimizers['cvae'])
                    
                    # Check gradients for overflow after unscaling
                    grad_overflow = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                                grad_overflow = True
                                break
                    
                    if grad_overflow:
                        print(f"⚠️  Gradient overflow detected in batch {batch_idx}")
                        continue
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                    
                    # Optimizer step
                    scaler.step(optimizers['cvae'])
                    scaler.update()
                    
                except RuntimeError as grad_error:
                    if "overflow" in str(grad_error).lower():
                        overflow_count += 1
                        continue
                    else:
                        raise
                
            else:
                # Standard precision training (safer)
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
                    pair_counts, df, current_indices
                )
                
                # Total CVAE loss
                cvae_loss = (config['reconstruction_weight'] * recon_loss + 
                            config['kl_weight'] * kl_loss + 
                            config['contrastive_weight'] * contrastive_loss)
                
                # Backward pass
                cvae_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                optimizers['cvae'].step()
            
            # Meta-learner training (every few batches to reduce overhead)
            if batch_idx % config.get('meta_learner_frequency', 5) == 0:
                try:
                    # Generate dummy scorer scores for meta-learner training
                    scorer_scores = {
                        'generative': torch.randn(len(positive_combinations), device=device) * 0.1,  # Smaller values
                        'temporal': torch.randn(len(positive_combinations), device=device) * 0.1, 
                        'i_ching': torch.randn(len(positive_combinations), device=device) * 0.1
                    }
                    
                    # Meta-learner forward pass
                    ensemble_weights, final_scores, confidence = meta_learner(
                        positive_combinations, temporal_context.detach(), scorer_scores
                    )
                    
                    # Simple meta-learner loss (encourage confident predictions)
                    meta_loss = -confidence.mean() + 0.1 * final_scores.var()
                    
                    # Clamp meta loss to prevent overflow
                    meta_loss = torch.clamp(meta_loss, min=-10, max=10)
                    
                    # Meta-learner backward pass
                    optimizers['meta'].zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), config['gradient_clip_norm'])
                    optimizers['meta'].step()
                    
                    epoch_losses['meta_loss'].append(meta_loss.item())
                    
                except Exception as meta_e:
                    # Meta-learner training is optional, continue without it
                    if batch_idx <= 5:  # Only warn for first few batches
                        print(f"Warning: Meta-learner training failed for batch {batch_idx}: {meta_e}")
            
            # Record losses
            epoch_losses['reconstruction_loss'].append(recon_loss.item())
            epoch_losses['kl_loss'].append(kl_loss.item()) 
            epoch_losses['contrastive_loss'].append(contrastive_loss.item())
            epoch_losses['total_cvae_loss'].append(cvae_loss.item())
            
            successful_batches += 1
            
            # Update progress bar
            current_loss = cvae_loss.item()
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'KL': f'{kl_loss.item():.4f}',
                'Contrastive': f'{contrastive_loss.item():.4f}',
                'Success': f'{successful_batches}/{batch_idx+1}'
            })
            
            # Check for loss explosion
            if current_loss > 1000:
                print(f"⚠️  High loss detected: {current_loss:.2f}")
                
        except Exception as e:
            error_msg = str(e)
            if "overflow" in error_msg.lower() or "half" in error_msg.lower():
                overflow_count += 1
                if overflow_count <= 3:
                    print(f"⚠️  Overflow error in batch {batch_idx}: {error_msg}")
                continue
            else:
                print(f"❌ Unexpected error in batch {batch_idx}: {e}")
                print(f"Batch info: positive_combinations={len(batch_dict.get('positive_combinations', []))}")
                
                # Emergency cleanup
                for optimizer in optimizers.values():
                    optimizer.zero_grad()
                
                # For critical errors, stop training
                if "CUDA out of memory" in str(e):
                    raise
                else:
                    continue
    
    # Print overflow summary
    if overflow_count > 0:
        print(f"\n⚠️  Mixed precision overflow occurred in {overflow_count} batches")
        print(f"✅ Successfully processed {successful_batches}/{len(train_loader)} batches")
        if overflow_count > len(train_loader) * 0.5:
            print("💡 Consider disabling mixed precision training (set use_mixed_precision=False)")
    
    # CRITICAL FIX: Ensure all required keys exist even if training failed
    required_keys = ['reconstruction_loss', 'kl_loss', 'contrastive_loss', 'total_cvae_loss', 'meta_loss']
    for key in required_keys:
        if key not in epoch_losses or not epoch_losses[key]:
            epoch_losses[key] = [0.0]  # Provide default value
            print(f"⚠️  No {key} recorded - using default value 0.0")
    
    # Return average losses
    avg_losses = {key: np.mean(values) if values else 0.0 for key, values in epoch_losses.items()}
    
    # Additional safety check
    for key in required_keys:
        if key not in avg_losses:
            avg_losses[key] = 0.0
    
    return avg_losses

def evaluate_cvae_modified(model, meta_learner, val_loader, device, config, df):
    """Modified evaluation function."""
    from src.cvae_engine import CVAELossComputer
    from collections import defaultdict
    
    model.eval()
    meta_learner.eval()
    
    loss_computer = CVAELossComputer(config)
    val_losses = defaultdict(list)
    
    with torch.no_grad():
        for batch_dict in val_loader:
            positive_combinations = batch_dict['positive_combinations']
            pair_counts = batch_dict['pair_counts']
            current_indices = batch_dict['current_indices']
            temporal_sequences = batch_dict['temporal_sequences']
            
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
    return avg_val_losses

def generate_sample_combinations(cvae_model, meta_learner, df, feature_engineer, config):
    """Generates sample combinations to verify the model is working."""
    cvae_model.eval()
    device = next(cvae_model.parameters()).device
    
    with torch.no_grad():
        # Get recent temporal context
        latest_index = len(df) - 1
        sequence = cvae_model.temporal_encoder.prepare_sequence_data(df, latest_index).to(device)
        context, _ = cvae_model.temporal_encoder(sequence)
        
        # Generate combinations
        combinations, log_probs = cvae_model.generate(
            context, 
            num_samples=10, 
            temperature=config['generation_temperature']
        )
        
        print("\nSample generated combinations:")
        for i, (combo, log_prob) in enumerate(zip(combinations, log_probs)):
            combo_list = combo.cpu().numpy().tolist()
            print(f"  {i+1}: {combo_list} (log_prob: {log_prob.item():.3f})")
        
        # Save to file
        sample_file = "outputs/sample_generations.txt"
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        with open(sample_file, 'w') as f:
            f.write("Sample Generated Combinations\n")
            f.write("=" * 30 + "\n")
            for i, (combo, log_prob) in enumerate(zip(combinations, log_probs)):
                combo_list = combo.cpu().numpy().tolist()
                f.write(f"{i+1}: {combo_list} (log_prob: {log_prob.item():.3f})\n")

def validate_one_epoch_cvae_modified(model, meta_learner, val_loader, device, config, df):
    """
    Validation function for CVAE with meta-learner.
    
    Args:
        model: CVAE model
        meta_learner: Meta-learner model
        val_loader: Validation data loader
        device: Device to run on
        config: Configuration dictionary
        df: Historical data DataFrame
    
    Returns:
        val_losses: Dictionary of average validation losses
    """
    from src.cvae_engine import CVAELossComputer
    
    model.eval()
    meta_learner.eval()
    
    loss_computer = CVAELossComputer(config)
    val_losses = defaultdict(list)
    
    with torch.no_grad():
        for batch_dict in tqdm(val_loader, desc="Validation"):
            try:
                # Extract batch data
                positive_combinations = batch_dict['positive_combinations']
                negative_pool = batch_dict['negative_pool']
                pair_counts = batch_dict['pair_counts']
                current_indices = batch_dict['current_indices']
                temporal_sequences = batch_dict['temporal_sequences']
                
                # Forward pass
                reconstruction_logits, mu, logvar, mu_prior, logvar_prior, temporal_context = model(
                    positive_combinations, pair_counts, temporal_sequences, current_indices
                )
                
                # Convert combinations to tensor
                target_tensor = torch.tensor(positive_combinations, device=device)
                
                # Compute losses
                recon_loss = loss_computer.reconstruction_loss(reconstruction_logits, target_tensor)
                kl_loss = loss_computer.kl_divergence_loss(mu, logvar, mu_prior, logvar_prior)
                contrastive_loss = loss_computer.hard_contrastive_loss(
                    model, positive_combinations, negative_pool, 
                    pair_counts, df, current_indices
                )
                
                # Total loss
                total_loss = (config['reconstruction_weight'] * recon_loss + 
                             config['kl_weight'] * kl_loss + 
                             config['contrastive_weight'] * contrastive_loss)
                
                # Record losses
                val_losses['reconstruction_loss'].append(recon_loss.item())
                val_losses['kl_loss'].append(kl_loss.item())
                val_losses['contrastive_loss'].append(contrastive_loss.item())
                val_losses['total_loss'].append(total_loss.item())
                
            except Exception as e:
                print(f"Warning: Validation batch failed: {e}")
                continue
    
    # Return average losses
    avg_losses = {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
    return avg_losses

def train_meta_learner_epoch(meta_learner, cvae_model, train_loader, optimizer, device, config, df):
    """
    Dedicated meta-learner training epoch.
    
    Args:
        meta_learner: Meta-learner model
        cvae_model: Trained CVAE model (frozen)
        train_loader: Training data loader
        optimizer: Meta-learner optimizer
        device: Training device
        config: Configuration dictionary
        df: Historical data DataFrame
    
    Returns:
        avg_loss: Average meta-learner loss
    """
    meta_learner.train()
    cvae_model.eval()  # Keep CVAE frozen
    
    epoch_losses = []
    
    with torch.no_grad():  # Don't compute gradients for CVAE
        for batch_dict in tqdm(train_loader, desc="Meta-Learner Training"):
            try:
                positive_combinations = batch_dict['positive_combinations']
                pair_counts = batch_dict['pair_counts']
                current_indices = batch_dict['current_indices']
                temporal_sequences = batch_dict['temporal_sequences']
                
                # Get temporal context from frozen CVAE
                _, _, _, _, _, temporal_context = cvae_model(
                    positive_combinations, pair_counts, temporal_sequences, current_indices
                )
                
                # Generate dummy scorer scores (in real scenario, these would come from actual scorers)
                scorer_scores = {
                    'generative': torch.randn(len(positive_combinations), device=device),
                    'temporal': torch.randn(len(positive_combinations), device=device),
                    'i_ching': torch.randn(len(positive_combinations), device=device)
                }
                
                # Meta-learner forward pass (enable gradients for meta-learner only)
                with torch.enable_grad():
                    ensemble_weights, final_scores, confidence = meta_learner(
                        positive_combinations, temporal_context, scorer_scores
                    )
                    
                    # Meta-learner loss: encourage confident, diverse predictions
                    confidence_loss = -confidence.mean()  # Maximize confidence
                    diversity_loss = -final_scores.var()   # Encourage diverse scores
                    regularization = 0.01 * sum(p.pow(2).sum() for p in meta_learner.parameters())
                    
                    meta_loss = confidence_loss + 0.1 * diversity_loss + regularization
                    
                    # Backward pass
                    optimizer.zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(meta_learner.parameters(), config['gradient_clip_norm'])
                    optimizer.step()
                    
                    epoch_losses.append(meta_loss.item())
                    
            except Exception as e:
                print(f"Warning: Meta-learner batch failed: {e}")
                continue
    

def get_pareto_parameters():
    """Get the latest Pareto Front optimization parameters for training integration."""
    import json
    from pathlib import Path
    
    best_params_dir = Path("models/best_parameters")
    if not best_params_dir.exists():
        return None
    
    # Find latest Pareto parameter file
    pareto_files = list(best_params_dir.glob("pareto_selected_*.json"))
    if not pareto_files:
        return None
    
    latest_file = max(pareto_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"Warning: Could not load Pareto parameters: {e}")
        return None


def apply_pareto_parameters(config, pareto_params):
    """Apply Pareto Front parameters to training configuration."""
    if not pareto_params or 'best_parameters' not in pareto_params:
        return config
    
    best_params = pareto_params['best_parameters']
    updated_config = config.copy()
    
    # Map Pareto parameters to training config
    param_mapping = {
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'hidden_dim': 'hidden_size',
        'dropout_rate': 'dropout_rate',
        'weight_decay': 'weight_decay'
    }
    
    for pareto_key, config_key in param_mapping.items():
        if pareto_key in best_params:
            updated_config[config_key] = best_params[pareto_key]
            print(f"🎯 Applied Pareto parameter: {config_key} = {best_params[pareto_key]}")
    
    return updated_config