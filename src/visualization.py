# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from collections import Counter

def plot_training_progress(train_history, val_history, save_path="outputs/training_progress.png"):
    """
    Creates comprehensive training progress plots.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CVAE Training Progress', fontsize=16, fontweight='bold')
    
    # Reconstruction Loss
    axes[0, 0].plot(train_history['reconstruction_loss'], label='Train', alpha=0.8, linewidth=2)
    axes[0, 0].plot(val_history['reconstruction_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Reconstruction Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Divergence Loss
    axes[0, 1].plot(train_history['kl_loss'], label='Train', alpha=0.8, linewidth=2)
    axes[0, 1].plot(val_history['kl_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[0, 1].set_title('KL Divergence Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contrastive Loss
    if 'contrastive_loss' in train_history and len(train_history['contrastive_loss']) > 0:
        axes[0, 2].plot(train_history['contrastive_loss'], label='Train', alpha=0.8, linewidth=2, color='red')
        axes[0, 2].set_title('Contrastive Loss', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Contrastive Loss\nNot Available', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Contrastive Loss', fontweight='bold')
    
    # Total Loss
    axes[1, 0].plot(train_history['total_cvae_loss'], label='Total CVAE Loss', alpha=0.8, linewidth=2, color='purple')
    axes[1, 0].set_title('Total CVAE Loss', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Meta-Learner Loss
    if 'meta_loss' in train_history and len(train_history['meta_loss']) > 0:
        axes[1, 1].plot(train_history['meta_loss'], label='Meta Loss', alpha=0.8, linewidth=2, color='orange')
        axes[1, 1].set_title('Meta-Learner Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Meta-Learner Loss\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Meta-Learner Loss', fontweight='bold')
    
    # Learning Rate (if available)
    axes[1, 2].text(0.5, 0.5, 'Learning Rate\nTracking\n(Future Feature)', 
                   ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Learning Rate Schedule', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space_2d(latent_codes, labels, save_path="outputs/latent_space_2d.png"):
    """
    Visualizes the latent space in 2D using t-SNE.
    
    Args:
        latent_codes: numpy array of latent codes [n_samples, latent_dim]
        labels: list of labels ('historical' or 'random')
        save_path: path to save the plot
    """
    print("Computing t-SNE projection of latent space...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_codes)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE plot
    colors = ['blue' if label else 'red' for label in labels]
    scatter = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, alpha=0.6, s=20)
    ax1.set_title('t-SNE Latent Space Visualization', fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    
    # Custom legend
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Historical Combinations')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Random Combinations')
    ax1.legend(handles=[blue_patch, red_patch])
    ax1.grid(True, alpha=0.3)
    
    # PCA plot for comparison
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_codes)
    
    ax2.scatter(latent_pca[:, 0], latent_pca[:, 1], c=colors, alpha=0.6, s=20)
    ax2.set_title('PCA Latent Space Visualization', fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.legend(handles=[blue_patch, red_patch])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latent space visualization saved to: {save_path}")

def plot_number_frequency_comparison(generated_combinations, historical_data, save_path="outputs/frequency_comparison.png"):
    """
    Compares the frequency distribution of generated vs historical numbers.
    """
    # Extract numbers from generated combinations
    generated_numbers = [num for combo in generated_combinations for num in combo]
    generated_freq = Counter(generated_numbers)
    
    # Extract historical numbers
    winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
    historical_numbers = []
    for _, row in historical_data.iterrows():
        historical_numbers.extend(row[winning_cols].astype(int).tolist())
    historical_freq = Counter(historical_numbers)
    
    # Create frequency arrays
    numbers = range(1, 50)  # Mark Six numbers 1-49
    gen_counts = [generated_freq.get(num, 0) for num in numbers]
    hist_counts = [historical_freq.get(num, 0) for num in numbers]
    
    # Normalize to frequencies
    gen_freq_norm = np.array(gen_counts) / sum(gen_counts) if sum(gen_counts) > 0 else np.zeros(len(numbers))
    hist_freq_norm = np.array(hist_counts) / sum(hist_counts) if sum(hist_counts) > 0 else np.zeros(len(numbers))
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Raw frequency comparison
    x = np.arange(len(numbers))
    width = 0.35
    
    ax1.bar(x - width/2, gen_counts, width, label='Generated', alpha=0.7, color='skyblue')
    ax1.bar(x + width/2, hist_counts, width, label='Historical', alpha=0.7, color='lightcoral')
    ax1.set_title('Raw Frequency Comparison', fontweight='bold')
    ax1.set_xlabel('Number')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(x[::5])  # Show every 5th number
    ax1.set_xticklabels([str(i) for i in numbers[::5]])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Normalized frequency comparison
    ax2.bar(x - width/2, gen_freq_norm, width, label='Generated', alpha=0.7, color='skyblue')
    ax2.bar(x + width/2, hist_freq_norm, width, label='Historical', alpha=0.7, color='lightcoral')
    ax2.set_title('Normalized Frequency Comparison', fontweight='bold')
    ax2.set_xlabel('Number')
    ax2.set_ylabel('Relative Frequency')
    ax2.set_xticks(x[::5])
    ax2.set_xticklabels([str(i) for i in numbers[::5]])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Difference plot
    freq_diff = gen_freq_norm - hist_freq_norm
    colors = ['red' if diff < 0 else 'green' for diff in freq_diff]
    ax3.bar(x, freq_diff, color=colors, alpha=0.7)
    ax3.set_title('Frequency Difference (Generated - Historical)', fontweight='bold')
    ax3.set_xlabel('Number')
    ax3.set_ylabel('Frequency Difference')
    ax3.set_xticks(x[::5])
    ax3.set_xticklabels([str(i) for i in numbers[::5]])
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation
    correlation = np.corrcoef(gen_freq_norm, hist_freq_norm)[0, 1]
    print(f"Frequency correlation: {correlation:.4f}")
    
    return correlation

def plot_ensemble_weights_analysis(explanations, save_path="outputs/ensemble_weights_analysis.png"):
    """
    Analyzes and visualizes ensemble weight distributions.
    """
    # Extract weights
    generative_weights = [exp['weights']['generative'] for exp in explanations]
    temporal_weights = [exp['weights']['temporal'] for exp in explanations]
    iching_weights = [exp['weights']['i_ching'] for exp in explanations if 'i_ching' in exp['weights']]
    
    # Extract confidence scores
    confidences = [exp['confidence'] for exp in explanations]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Weight distribution histograms
    axes[0, 0].hist(generative_weights, bins=20, alpha=0.7, color='blue', label='Generative')
    axes[0, 0].hist(temporal_weights, bins=20, alpha=0.7, color='red', label='Temporal')
    if iching_weights:
        axes[0, 0].hist(iching_weights, bins=20, alpha=0.7, color='green', label='I-Ching')
    axes[0, 0].set_title('Ensemble Weight Distributions', fontweight='bold')
    axes[0, 0].set_xlabel('Weight')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence distribution
    axes[0, 1].hist(confidences, bins=20, alpha=0.7, color='purple')
    axes[0, 1].set_title('Confidence Score Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight correlation analysis
    axes[1, 0].scatter(generative_weights, temporal_weights, alpha=0.6, color='orange')
    axes[1, 0].set_title('Generative vs Temporal Weights', fontweight='bold')
    axes[1, 0].set_xlabel('Generative Weight')
    axes[1, 0].set_ylabel('Temporal Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence vs dominant weight
    dominant_weights = [max(exp['weights'].values()) for exp in explanations]
    axes[1, 1].scatter(dominant_weights, confidences, alpha=0.6, color='brown')
    axes[1, 1].set_title('Dominant Weight vs Confidence', fontweight='bold')
    axes[1, 1].set_xlabel('Dominant Weight')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_generation_statistics(combinations, save_path="outputs/generation_statistics.png"):
    """
    Analyzes statistical properties of generated combinations.
    """
    # Calculate statistics
    sums = [sum(combo) for combo in combinations]
    ranges = [max(combo) - min(combo) for combo in combinations]
    odd_counts = [sum(1 for num in combo if num % 2 == 1) for combo in combinations]
    consecutive_counts = []
    
    for combo in combinations:
        sorted_combo = sorted(combo)
        consecutive = sum(1 for i in range(len(sorted_combo)-1) if sorted_combo[i+1] - sorted_combo[i] == 1)
        consecutive_counts.append(consecutive)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sum distribution
    axes[0, 0].hist(sums, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Sum Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Sum of Numbers')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(sums), color='red', linestyle='--', label=f'Mean: {np.mean(sums):.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Range distribution
    axes[0, 1].hist(ranges, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Range Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Range (Max - Min)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(ranges), color='red', linestyle='--', label=f'Mean: {np.mean(ranges):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Odd number count distribution
    axes[1, 0].hist(odd_counts, bins=7, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Odd Numbers per Combination', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Odd Numbers')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Consecutive numbers distribution
    axes[1, 1].hist(consecutive_counts, bins=6, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Consecutive Numbers per Combination', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Consecutive Pairs')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xticks(range(6))
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nGeneration Statistics Summary:")
    print(f"Average sum: {np.mean(sums):.2f} ± {np.std(sums):.2f}")
    print(f"Average range: {np.mean(ranges):.2f} ± {np.std(ranges):.2f}")
    print(f"Average odd numbers: {np.mean(odd_counts):.2f} ± {np.std(odd_counts):.2f}")
    print(f"Average consecutive pairs: {np.mean(consecutive_counts):.2f} ± {np.std(consecutive_counts):.2f}")

def create_evaluation_dashboard(evaluation_results, save_path="outputs/evaluation_dashboard.png"):
    """
    Creates a comprehensive evaluation dashboard.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Generation Quality (top row)
    ax1 = fig.add_subplot(gs[0, 0])
    generation = evaluation_results['generation']
    metrics = ['Valid', 'Unique', 'Constraints OK']
    values = [
        generation['valid_combinations'],
        generation['unique_combinations'], 
        100 - generation['constraint_violations']
    ]
    colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_title('Generation Quality (%)', fontweight='bold')
    ax1.set_ylabel('Percentage')
    ax1.set_ylim(0, 100)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    # Ensemble Performance
    ax2 = fig.add_subplot(gs[0, 1])
    ensemble = evaluation_results['ensemble']
    if 'error' not in ensemble:
        win_rate = ensemble['win_rate'] * 100
        color = 'green' if win_rate >= 60 else 'orange' if win_rate >= 50 else 'red'
        ax2.pie([win_rate, 100-win_rate], labels=['Wins', 'Losses'], 
               colors=[color, 'lightgray'], autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Ensemble Win Rate\n{win_rate:.1f}%', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Ensemble\nEvaluation\nFailed', ha='center', va='center')
        ax2.set_title('Ensemble Performance', fontweight='bold')
    
    # Reconstruction Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    reconstruction = evaluation_results['reconstruction']
    exact_match = reconstruction['exact_match_rate'] * 100
    partial_match = reconstruction['average_partial_matches'] / 6 * 100
    
    categories = ['Exact\nMatch', 'Partial\nMatch\n(Avg)']
    values = [exact_match, partial_match]
    colors = ['darkgreen', 'lightgreen']
    bars = ax3.bar(categories, values, color=colors, alpha=0.8)
    ax3.set_title('Reconstruction Accuracy', fontweight='bold')
    ax3.set_ylabel('Percentage')
    ax3.set_ylim(0, 100)
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Latent Space Quality
    ax4 = fig.add_subplot(gs[0, 3])
    latent = evaluation_results['latent_space']
    if latent['cluster_quality']:
        separation = latent['cluster_quality']['separation_ratio']
        color = 'green' if separation >= 1.0 else 'orange' if separation >= 0.5 else 'red'
        ax4.bar(['Separation\nRatio'], [separation], color=color, alpha=0.7)
        ax4.set_title('Latent Space Quality', fontweight='bold')
        ax4.set_ylabel('Separation Ratio')
        ax4.text(0, separation + 0.05, f'{separation:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Latent Space\nAnalysis\nUnavailable', ha='center', va='center')
        ax4.set_title('Latent Space Quality', fontweight='bold')
    
    # Detailed metrics (bottom section)
    ax5 = fig.add_subplot(gs[1:, :])
    ax5.axis('off')
    
    # Create detailed metrics table
    metrics_text = "DETAILED EVALUATION METRICS\n" + "="*50 + "\n\n"
    
    # Generation metrics
    metrics_text += "GENERATION QUALITY:\n"
    metrics_text += f"• Valid combinations: {generation['valid_combinations']}/100\n"
    metrics_text += f"• Unique combinations: {generation['unique_combinations']}\n"
    metrics_text += f"• Constraint violations: {generation['constraint_violations']}\n"
    if 'distribution_stats' in generation and generation['distribution_stats']:
        stats = generation['distribution_stats']
        metrics_text += f"• Frequency correlation: {stats['freq_correlation']:.4f}\n"
        metrics_text += f"• KS statistic: {stats['ks_statistic']:.4f}\n"
    metrics_text += "\n"
    
    # Ensemble metrics
    metrics_text += "ENSEMBLE PERFORMANCE:\n"
    if 'error' not in ensemble:
        metrics_text += f"• Win rate: {ensemble['win_rate']:.2%}\n"
        metrics_text += f"• Successful rankings: {ensemble['wins']}/{ensemble['total_comparisons']}\n"
    else:
        metrics_text += f"• Error: {ensemble['error']}\n"
    metrics_text += "\n"
    
    # Reconstruction metrics
    metrics_text += "RECONSTRUCTION ACCURACY:\n"
    metrics_text += f"• Exact match rate: {reconstruction['exact_match_rate']:.2%}\n"
    metrics_text += f"• Average partial matches: {reconstruction['average_partial_matches']:.2f}/6\n"
    metrics_text += f"• Average reconstruction loss: {reconstruction['average_reconstruction_loss']:.4f}\n"
    metrics_text += f"• Position accuracies: {', '.join([f'{acc:.1%}' for acc in reconstruction['position_accuracies']])}\n"
    metrics_text += "\n"
    
    # Latent space metrics
    metrics_text += "LATENT SPACE QUALITY:\n"
    if latent['cluster_quality']:
        cluster = latent['cluster_quality']
        metrics_text += f"• Separation ratio: {cluster['separation_ratio']:.4f}\n"
        metrics_text += f"• Centroid distance: {cluster['centroid_distance']:.4f}\n"
        metrics_text += f"• Historical intra-cluster distance: {cluster['historical_intra_dist']:.4f}\n"
        metrics_text += f"• Random intra-cluster distance: {cluster['random_intra_dist']:.4f}\n"
    else:
        metrics_text += "• Cluster analysis not available\n"
    
    ax5.text(0.02, 0.98, metrics_text, transform=ax5.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('CVAE Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation dashboard saved to: {save_path}")