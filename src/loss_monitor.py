# src/loss_monitor.py
"""
Comprehensive loss monitoring and debugging system for CVAE training.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os
import json
from datetime import datetime

class LossMonitor:
    """Advanced loss monitoring and debugging system."""
    
    def __init__(self, config, save_dir="outputs/loss_monitoring"):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss history tracking
        self.loss_history = {
            'reconstruction': deque(maxlen=1000),
            'kl_divergence': deque(maxlen=1000),
            'contrastive': deque(maxlen=1000),
            'total': deque(maxlen=1000),
            'meta_learner': deque(maxlen=1000)
        }
        
        # Batch-level statistics
        self.batch_stats = defaultdict(list)
        
        # Epoch-level summaries
        self.epoch_summaries = []
        
        # Problematic patterns detection
        self.pattern_detectors = {
            'zero_reconstruction': [],
            'kl_collapse': [],
            'gradient_explosion': [],
            'loss_spikes': []
        }
        
        # Real-time monitoring thresholds
        self.thresholds = {
            'reconstruction_zero_threshold': 1e-8,
            'kl_collapse_threshold': 1e-6,
            'gradient_explosion_threshold': 100.0,
            'loss_spike_multiplier': 10.0
        }
        
    def log_batch_losses(self, epoch, batch_idx, losses_dict, gradients_dict=None):
        """Log detailed batch-level loss information."""
        timestamp = datetime.now().isoformat()
        
        # Store losses
        for loss_name, loss_value in losses_dict.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            self.loss_history[loss_name].append(loss_value)
        
        # Detect patterns
        self._detect_patterns(epoch, batch_idx, losses_dict)
        
        # Store batch statistics
        batch_info = {
            'epoch': epoch,
            'batch': batch_idx,
            'timestamp': timestamp,
            'losses': {k: v.item() if isinstance(v, torch.Tensor) else v 
                      for k, v in losses_dict.items()},
        }
        
        if gradients_dict:
            batch_info['gradients'] = {k: v.item() if isinstance(v, torch.Tensor) else v 
                                     for k, v in gradients_dict.items()}
        
        self.batch_stats[epoch].append(batch_info)
        return batch_info
    
    def _detect_patterns(self, epoch, batch_idx, losses_dict):
        """Detect problematic training patterns."""
        recon_loss = losses_dict.get('reconstruction', 0)
        kl_loss = losses_dict.get('kl_divergence', 0)
        total_loss = losses_dict.get('total', 0)
        
        # Convert to float if tensor
        if isinstance(recon_loss, torch.Tensor):
            recon_loss = recon_loss.item()
        if isinstance(kl_loss, torch.Tensor):
            kl_loss = kl_loss.item()
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss.item()
        
        # Zero reconstruction loss (perfect memorization)
        if recon_loss < self.thresholds['reconstruction_zero_threshold']:
            self.pattern_detectors['zero_reconstruction'].append({
                'epoch': epoch, 'batch': batch_idx, 'value': recon_loss
            })
        
        # KL collapse
        if kl_loss < self.thresholds['kl_collapse_threshold']:
            self.pattern_detectors['kl_collapse'].append({
                'epoch': epoch, 'batch': batch_idx, 'value': kl_loss
            })
        
        # Loss spikes
        if len(self.loss_history['total']) > 1:
            prev_loss = list(self.loss_history['total'])[-2]
            if total_loss > prev_loss * self.thresholds['loss_spike_multiplier']:
                self.pattern_detectors['loss_spikes'].append({
                    'epoch': epoch, 'batch': batch_idx, 
                    'current': total_loss, 'previous': prev_loss
                })
    
    def analyze_epoch_patterns(self, epoch):
        """Analyze patterns at the end of an epoch."""
        if epoch not in self.batch_stats:
            return {}
        
        epoch_data = self.batch_stats[epoch]
        
        # Calculate statistics
        reconstruction_losses = [b['losses'].get('reconstruction', 0) for b in epoch_data]
        kl_losses = [b['losses'].get('kl_divergence', 0) for b in epoch_data]
        total_losses = [b['losses'].get('total', 0) for b in epoch_data]
        
        analysis = {
            'epoch': epoch,
            'total_batches': len(epoch_data),
            'reconstruction_stats': {
                'mean': np.mean(reconstruction_losses),
                'std': np.std(reconstruction_losses),
                'min': np.min(reconstruction_losses),
                'max': np.max(reconstruction_losses),
                'zero_count': sum(1 for x in reconstruction_losses if x < 1e-8)
            },
            'kl_stats': {
                'mean': np.mean(kl_losses),
                'std': np.std(kl_losses),
                'min': np.min(kl_losses),
                'max': np.max(kl_losses),
                'collapse_count': sum(1 for x in kl_losses if x < 1e-6)
            },
            'total_loss_stats': {
                'mean': np.mean(total_losses),
                'std': np.std(total_losses),
                'min': np.min(total_losses),
                'max': np.max(total_losses)
            }
        }
        
        self.epoch_summaries.append(analysis)
        return analysis
    
    def generate_diagnostic_report(self, epoch):
        """Generate comprehensive diagnostic report."""
        report = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'pattern_detections': {}
        }
        
        # Add pattern detection results
        for pattern_name, detections in self.pattern_detectors.items():
            recent_detections = [d for d in detections if d['epoch'] == epoch]
            report['pattern_detections'][pattern_name] = {
                'count': len(recent_detections),
                'instances': recent_detections[-5:]  # Last 5 instances
            }
        
        # Add epoch analysis
        if self.epoch_summaries:
            report['epoch_analysis'] = self.epoch_summaries[-1]
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Save report
        report_path = os.path.join(self.save_dir, f'diagnostic_report_epoch_{epoch}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, report):
        """Generate specific recommendations based on detected patterns."""
        recommendations = []
        
        patterns = report['pattern_detections']
        
        # Zero reconstruction loss
        if patterns['zero_reconstruction']['count'] > 5:
            recommendations.append({
                'issue': 'Frequent zero reconstruction loss',
                'severity': 'high',
                'suggestion': 'Possible overfitting - consider: 1) Increase dropout, 2) Reduce model capacity, 3) Add data augmentation'
            })
        
        # KL collapse
        if patterns['kl_collapse']['count'] > 10:
            recommendations.append({
                'issue': 'KL divergence collapse',
                'severity': 'high', 
                'suggestion': 'Add β-VAE annealing schedule or increase KL weight'
            })
        
        # Loss spikes
        if patterns['loss_spikes']['count'] > 3:
            recommendations.append({
                'issue': 'Frequent loss spikes',
                'severity': 'medium',
                'suggestion': 'Consider: 1) Lower learning rate, 2) Stronger gradient clipping, 3) Check data quality'
            })
        
        # Epoch analysis recommendations
        if 'epoch_analysis' in report:
            analysis = report['epoch_analysis']
            
            # High reconstruction loss
            if analysis['reconstruction_stats']['mean'] > 2.0:
                recommendations.append({
                    'issue': 'High reconstruction loss',
                    'severity': 'medium',
                    'suggestion': 'Model may be underfitting - consider increasing capacity or reducing regularization'
                })
            
            # High KL divergence
            if analysis['kl_stats']['mean'] > 10.0:
                recommendations.append({
                    'issue': 'High KL divergence',
                    'severity': 'medium',
                    'suggestion': 'Consider reducing KL weight or checking prior/posterior balance'
                })
        
        return recommendations
    
    def create_loss_plots(self, epoch, save_plots=True):
        """Create comprehensive loss visualization plots."""
        if not self.batch_stats.get(epoch):
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epoch_data = self.batch_stats[epoch]
        batch_indices = range(len(epoch_data))
        
        # Reconstruction loss
        recon_losses = [b['losses'].get('reconstruction', 0) for b in epoch_data]
        axes[0, 0].plot(batch_indices, recon_losses, 'b-', alpha=0.7)
        axes[0, 0].set_title('Reconstruction Loss')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # KL Divergence (log scale for better visibility)
        kl_losses = [b['losses'].get('kl_divergence', 0) for b in epoch_data]
        axes[0, 1].semilogy(batch_indices, kl_losses, 'r-', alpha=0.7)
        axes[0, 1].set_title('KL Divergence (Log Scale)')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].grid(True)
        
        # Contrastive loss
        cont_losses = [b['losses'].get('contrastive', 0) for b in epoch_data]
        axes[0, 2].plot(batch_indices, cont_losses, 'g-', alpha=0.7)
        axes[0, 2].set_title('Contrastive Loss')
        axes[0, 2].set_xlabel('Batch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        # Total loss
        total_losses = [b['losses'].get('total', 0) for b in epoch_data]
        axes[1, 0].plot(batch_indices, total_losses, 'k-', alpha=0.7)
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Loss component ratios
        if recon_losses and kl_losses and cont_losses:
            ratios = []
            for r, k, c in zip(recon_losses, kl_losses, cont_losses):
                total = r + k + c
                if total > 0:
                    ratios.append([r/total, k/total, c/total])
                else:
                    ratios.append([0, 0, 0])
            
            ratios = np.array(ratios)
            axes[1, 1].plot(batch_indices, ratios[:, 0], 'b-', label='Reconstruction', alpha=0.7)
            axes[1, 1].plot(batch_indices, ratios[:, 1], 'r-', label='KL Divergence', alpha=0.7)
            axes[1, 1].plot(batch_indices, ratios[:, 2], 'g-', label='Contrastive', alpha=0.7)
            axes[1, 1].set_title('Loss Component Ratios')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Pattern detection summary
        axes[1, 2].axis('off')
        pattern_text = f"Pattern Detection (Epoch {epoch}):\n"
        for pattern_name, detections in self.pattern_detectors.items():
            count = len([d for d in detections if d['epoch'] == epoch])
            pattern_text += f"{pattern_name}: {count}\n"
        axes[1, 2].text(0.1, 0.5, pattern_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.save_dir, f'loss_analysis_epoch_{epoch}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return plot_path
        else:
            return fig
    
    def print_epoch_summary(self, epoch):
        """Print human-readable epoch summary."""
        if epoch not in self.batch_stats:
            print(f"No data available for epoch {epoch}")
            return
        
        analysis = self.analyze_epoch_patterns(epoch)
        
        print(f"\n📊 Loss Analysis Summary - Epoch {epoch}")
        print("=" * 50)
        
        # Reconstruction loss
        r_stats = analysis['reconstruction_stats']
        print(f"🔄 Reconstruction Loss:")
        print(f"   Mean: {r_stats['mean']:.6f} ± {r_stats['std']:.6f}")
        print(f"   Range: [{r_stats['min']:.6f}, {r_stats['max']:.6f}]")
        if r_stats['zero_count'] > 0:
            print(f"   ⚠️  Zero loss batches: {r_stats['zero_count']}/{analysis['total_batches']}")
        
        # KL divergence
        k_stats = analysis['kl_stats']
        print(f"📐 KL Divergence:")
        print(f"   Mean: {k_stats['mean']:.8f} ± {k_stats['std']:.8f}")
        print(f"   Range: [{k_stats['min']:.8f}, {k_stats['max']:.8f}]")
        if k_stats['collapse_count'] > 0:
            print(f"   ⚠️  Collapsed batches: {k_stats['collapse_count']}/{analysis['total_batches']}")
        
        # Total loss
        t_stats = analysis['total_loss_stats']
        print(f"📈 Total Loss:")
        print(f"   Mean: {t_stats['mean']:.6f} ± {t_stats['std']:.6f}")
        print(f"   Range: [{t_stats['min']:.6f}, {t_stats['max']:.6f}]")
        
        # Pattern warnings
        warnings = []
        if r_stats['zero_count'] > analysis['total_batches'] * 0.1:
            warnings.append("Excessive zero reconstruction loss (possible overfitting)")
        if k_stats['collapse_count'] > analysis['total_batches'] * 0.2:
            warnings.append("Frequent KL collapse (posterior ≈ prior)")
        if t_stats['std'] > t_stats['mean']:
            warnings.append("High loss variance (unstable training)")
        
        if warnings:
            print(f"\n⚠️  Training Warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        else:
            print(f"\n✅ Training appears stable")
            
    def save_monitoring_state(self, filepath=None):
        """Save the complete monitoring state."""
        if filepath is None:
            filepath = os.path.join(self.save_dir, 'monitoring_state.json')
        
        state = {
            'loss_history': {k: list(v) for k, v in self.loss_history.items()},
            'batch_stats': dict(self.batch_stats),
            'epoch_summaries': self.epoch_summaries,
            'pattern_detectors': dict(self.pattern_detectors),
            'thresholds': self.thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Monitoring state saved to {filepath}")