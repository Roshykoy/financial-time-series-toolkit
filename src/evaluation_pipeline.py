# src/evaluation_pipeline.py
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CONFIG
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.feature_engineering import FeatureEngineer
from src.temporal_scorer import TemporalScorer
from src.i_ching_scorer import IChingScorer
from src.inference_pipeline import GenerativeEnsemble

class CVAEEvaluator:
    """
    Comprehensive evaluation suite for the CVAE-based lottery prediction system.
    Tests multiple aspects: generation quality, ensemble performance, and predictive power.
    """
    
    def __init__(self, cvae_model, meta_learner, feature_engineer, df, config):
        self.cvae_model = cvae_model
        self.meta_learner = meta_learner
        self.feature_engineer = feature_engineer
        self.df = df
        self.config = config
        self.device = next(cvae_model.parameters()).device
        
        # Initialize scorers for evaluation
        self.temporal_scorer = TemporalScorer(config)
        self.temporal_scorer.fit(df)
        self.i_ching_scorer = IChingScorer(config)
        
    def evaluate_generation_quality(self, num_samples=100):
        """
        Evaluates the quality of generated combinations.
        
        Metrics:
        - Validity rate (all combinations are valid)
        - Diversity (uniqueness of generated combinations)
        - Distribution similarity to historical data
        - Constraint satisfaction (no duplicates within combination)
        """
        print("Evaluating generation quality...")
        
        results = {
            'valid_combinations': 0,
            'unique_combinations': 0,
            'constraint_violations': 0,
            'generated_combinations': [],
            'distribution_stats': {}
        }
        
        # Generate samples
        generated_combinations = []
        with torch.no_grad():
            # Use recent temporal context
            current_index = len(self.df)
            sequence = self.cvae_model.temporal_encoder.prepare_sequence_data(
                self.df, current_index
            ).to(self.device)
            context, _ = self.cvae_model.temporal_encoder(sequence)
            
            # Generate in batches
            batch_size = 10
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - len(generated_combinations))
                batch_context = context.expand(current_batch_size, -1)
                
                combinations, _ = self.cvae_model.generate(
                    batch_context, num_samples=1, temperature=0.8
                )
                
                for i in range(current_batch_size):
                    combo = combinations[i].cpu().numpy().tolist()
                    generated_combinations.append(combo)
        
        # Analyze generated combinations
        unique_combinations = set()
        
        for combo in generated_combinations:
            # Check validity
            if (len(combo) == 6 and 
                all(isinstance(num, (int, np.integer)) for num in combo) and
                all(1 <= num <= self.config['num_lotto_numbers'] for num in combo)):
                results['valid_combinations'] += 1
                
                # Check for duplicates within combination
                if len(set(combo)) == 6:
                    unique_combinations.add(tuple(sorted(combo)))
                else:
                    results['constraint_violations'] += 1
            
            results['generated_combinations'].append(combo)
        
        results['unique_combinations'] = len(unique_combinations)
        
        # Analyze distribution statistics
        if results['valid_combinations'] > 0:
            valid_combos = [combo for combo in generated_combinations 
                          if len(set(combo)) == 6 and 
                          all(1 <= num <= self.config['num_lotto_numbers'] for num in combo)]
            
            # Number frequency distribution
            all_numbers = [num for combo in valid_combos for num in combo]
            number_freq = np.bincount(all_numbers, minlength=self.config['num_lotto_numbers']+1)[1:]
            
            # Historical frequency for comparison
            historical_numbers = []
            winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
            for _, row in self.df.iterrows():
                historical_numbers.extend(row[winning_cols].astype(int).tolist())
            historical_freq = np.bincount(historical_numbers, minlength=self.config['num_lotto_numbers']+1)[1:]
            
            # Statistical comparison
            results['distribution_stats'] = {
                'generated_freq': number_freq,
                'historical_freq': historical_freq,
                'freq_correlation': np.corrcoef(number_freq, historical_freq)[0, 1],
                'ks_statistic': self._compute_ks_statistic(number_freq, historical_freq)
            }
        
        return results
    
    def evaluate_ensemble_performance(self, num_test_samples=50):
        """
        Evaluates the ensemble's ability to rank historical winning combinations
        higher than random combinations.
        """
        print("Evaluating ensemble ranking performance...")
        
        # Split data for evaluation
        train_size = int(len(self.df) * 0.85)
        test_df = self.df.iloc[train_size:].reset_index(drop=True)
        
        if len(test_df) == 0:
            print("Warning: No test data available")
            return {'error': 'No test data'}
        
        # Create ensemble
        ensemble = GenerativeEnsemble(
            cvae_model=self.cvae_model,
            meta_learner=self.meta_learner,
            temporal_scorer=self.temporal_scorer,
            i_ching_scorer=self.i_ching_scorer,
            feature_engineer=self.feature_engineer,
            df=self.df,
            config=self.config,
            use_i_ching=True
        )
        
        results = {
            'wins': 0,
            'total_comparisons': 0,
            'win_rates_by_method': defaultdict(int),
            'score_distributions': defaultdict(list)
        }
        
        winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        num_test_samples = min(num_test_samples, len(test_df))
        
        for idx in tqdm(range(num_test_samples), desc="Testing ensemble"):
            row = test_df.iloc[idx]
            positive_combination = row[winning_cols].astype(int).tolist()
            
            # Score the actual winning combination
            positive_scores = ensemble.score_candidates([positive_combination])
            
            # Generate random negative combinations
            negative_combinations = []
            for _ in range(self.config['evaluation_neg_samples']):
                while True:
                    negative_combo = sorted(random.sample(
                        range(1, self.config['num_lotto_numbers'] + 1), 6
                    ))
                    if tuple(negative_combo) not in self.feature_engineer.historical_sets:
                        negative_combinations.append(negative_combo)
                        break
            
            # Score negative combinations
            negative_scores = ensemble.score_candidates(negative_combinations)
            
            # Compare scores using ensemble
            all_combinations = [positive_combination] + negative_combinations
            all_scores = ensemble.score_candidates(all_combinations)
            
            # Use meta-learner for final ranking
            ranked_combinations, final_scores, _ = ensemble.ensemble_rerank(
                all_combinations, all_scores
            )
            
            # Check if positive combination is ranked first
            if ranked_combinations[0] == positive_combination:
                results['wins'] += 1
            
            results['total_comparisons'] += 1
            
            # Record score distributions
            results['score_distributions']['positive_generative'].append(
                positive_scores['generative'][0]
            )
            results['score_distributions']['positive_temporal'].append(
                positive_scores['temporal'][0]
            )
            results['score_distributions']['negative_generative'].extend(
                negative_scores['generative']
            )
            results['score_distributions']['negative_temporal'].extend(
                negative_scores['temporal']
            )
        
        results['win_rate'] = results['wins'] / results['total_comparisons'] if results['total_comparisons'] > 0 else 0
        
        return results
    
    def evaluate_reconstruction_accuracy(self, num_test_samples=100):
        """
        Evaluates how well the CVAE can reconstruct historical winning combinations.
        """
        print("Evaluating reconstruction accuracy...")
        
        # Use validation set
        train_size = int(len(self.df) * 0.85)
        test_df = self.df.iloc[train_size:].reset_index(drop=True)
        
        results = {
            'reconstruction_losses': [],
            'exact_matches': 0,
            'partial_matches': [],
            'position_accuracies': [0] * 6
        }
        
        winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        num_test_samples = min(num_test_samples, len(test_df))
        
        with torch.no_grad():
            for idx in tqdm(range(num_test_samples), desc="Testing reconstruction"):
                row = test_df.iloc[idx]
                target_combination = row[winning_cols].astype(int).tolist()
                
                # Forward pass through CVAE
                current_index = train_size + idx
                reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context = self.cvae_model(
                    [target_combination], 
                    self.feature_engineer.pair_counts, 
                    self.df, 
                    [current_index]
                )
                
                # Compute reconstruction loss
                target_tensor = torch.tensor([target_combination], device=self.device) - 1  # 0-based
                recon_loss = 0
                for pos in range(6):
                    pos_loss = torch.nn.functional.cross_entropy(
                        reconstruction_logits[:, pos, :], target_tensor[:, pos]
                    )
                    recon_loss += pos_loss.item()
                
                results['reconstruction_losses'].append(recon_loss / 6)
                
                # Get most likely reconstruction
                reconstructed_numbers = []
                for pos in range(6):
                    most_likely = torch.argmax(reconstruction_logits[0, pos, :]).item() + 1  # 1-based
                    reconstructed_numbers.append(most_likely)
                
                # Check accuracy
                if sorted(reconstructed_numbers) == sorted(target_combination):
                    results['exact_matches'] += 1
                
                # Position-wise accuracy
                for pos in range(6):
                    if reconstructed_numbers[pos] in target_combination:
                        results['position_accuracies'][pos] += 1
                
                # Partial match count
                matches = len(set(reconstructed_numbers) & set(target_combination))
                results['partial_matches'].append(matches)
        
        # Normalize position accuracies
        results['position_accuracies'] = [acc / num_test_samples for acc in results['position_accuracies']]
        results['exact_match_rate'] = results['exact_matches'] / num_test_samples
        results['average_partial_matches'] = np.mean(results['partial_matches'])
        results['average_reconstruction_loss'] = np.mean(results['reconstruction_losses'])
        
        return results
    
    def evaluate_latent_space_quality(self, num_samples=200):
        """
        Evaluates the quality of the learned latent space.
        """
        print("Evaluating latent space quality...")
        
        results = {
            'latent_codes': [],
            'combinations': [],
            'is_historical': [],
            'cluster_quality': {}
        }
        
        # Encode historical combinations
        winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        historical_combinations = []
        for _, row in self.df.iterrows():
            combo = row[winning_cols].astype(int).tolist()
            historical_combinations.append(combo)
        
        # Sample subset for evaluation
        sampled_historical = random.sample(historical_combinations, 
                                         min(num_samples // 2, len(historical_combinations)))
        
        # Generate random combinations
        random_combinations = []
        while len(random_combinations) < num_samples // 2:
            combo = sorted(random.sample(range(1, self.config['num_lotto_numbers'] + 1), 6))
            if tuple(combo) not in self.feature_engineer.historical_sets:
                random_combinations.append(combo)
        
        all_combinations = sampled_historical + random_combinations
        is_historical = [True] * len(sampled_historical) + [False] * len(random_combinations)
        
        # Encode all combinations
        with torch.no_grad():
            for i, combo in enumerate(tqdm(all_combinations, desc="Encoding combinations")):
                mu, logvar = self.cvae_model.encode([combo], self.feature_engineer.pair_counts)
                z = self.cvae_model.reparameterize(mu, logvar)
                
                results['latent_codes'].append(z.cpu().numpy().flatten())
                results['combinations'].append(combo)
                results['is_historical'].append(is_historical[i])
        
        # Analyze latent space clustering
        latent_codes = np.array(results['latent_codes'])
        
        # Compute separation between historical and random combinations
        historical_codes = latent_codes[np.array(results['is_historical'])]
        random_codes = latent_codes[~np.array(results['is_historical'])]
        
        if len(historical_codes) > 0 and len(random_codes) > 0:
            # Compute mean distances
            historical_centroid = np.mean(historical_codes, axis=0)
            random_centroid = np.mean(random_codes, axis=0)
            centroid_distance = np.linalg.norm(historical_centroid - random_centroid)
            
            # Compute intra-cluster distances
            historical_intra_dist = np.mean([
                np.linalg.norm(code - historical_centroid) for code in historical_codes
            ])
            random_intra_dist = np.mean([
                np.linalg.norm(code - random_centroid) for code in random_codes
            ])
            
            results['cluster_quality'] = {
                'centroid_distance': centroid_distance,
                'historical_intra_dist': historical_intra_dist,
                'random_intra_dist': random_intra_dist,
                'separation_ratio': centroid_distance / (historical_intra_dist + random_intra_dist)
            }
        
        return results
    
    def _compute_ks_statistic(self, dist1, dist2):
        """Compute Kolmogorov-Smirnov statistic between two distributions."""
        from scipy import stats
        return stats.ks_2samp(dist1, dist2).statistic
    
    def generate_evaluation_report(self, save_path="outputs/evaluation_report.txt"):
        """
        Runs comprehensive evaluation and generates a detailed report.
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE CVAE EVALUATION")
        print("=" * 60)
        
        # Run all evaluations
        generation_results = self.evaluate_generation_quality()
        ensemble_results = self.evaluate_ensemble_performance()
        reconstruction_results = self.evaluate_reconstruction_accuracy()
        latent_results = self.evaluate_latent_space_quality()
        
        # Generate report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("MARK SIX CVAE EVALUATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Generation Quality
            f.write("1. GENERATION QUALITY\n")
            f.write("-" * 25 + "\n")
            f.write(f"Valid combinations: {generation_results['valid_combinations']}/100 ({generation_results['valid_combinations']}%)\n")
            f.write(f"Unique combinations: {generation_results['unique_combinations']}\n")
            f.write(f"Constraint violations: {generation_results['constraint_violations']}\n")
            
            if 'distribution_stats' in generation_results and generation_results['distribution_stats']:
                stats = generation_results['distribution_stats']
                f.write(f"Frequency correlation with historical: {stats['freq_correlation']:.4f}\n")
                f.write(f"KS statistic: {stats['ks_statistic']:.4f}\n")
            f.write("\n")
            
            # Ensemble Performance
            f.write("2. ENSEMBLE PERFORMANCE\n")
            f.write("-" * 26 + "\n")
            if 'error' not in ensemble_results:
                f.write(f"Win rate: {ensemble_results['win_rate']:.2%}\n")
                f.write(f"Wins: {ensemble_results['wins']}/{ensemble_results['total_comparisons']}\n")
            else:
                f.write(f"Error: {ensemble_results['error']}\n")
            f.write("\n")
            
            # Reconstruction Accuracy
            f.write("3. RECONSTRUCTION ACCURACY\n")
            f.write("-" * 29 + "\n")
            f.write(f"Exact match rate: {reconstruction_results['exact_match_rate']:.2%}\n")
            f.write(f"Average partial matches: {reconstruction_results['average_partial_matches']:.2f}/6\n")
            f.write(f"Average reconstruction loss: {reconstruction_results['average_reconstruction_loss']:.4f}\n")
            f.write("Position accuracies: " + ", ".join([f"{acc:.2%}" for acc in reconstruction_results['position_accuracies']]) + "\n")
            f.write("\n")
            
            # Latent Space Quality
            f.write("4. LATENT SPACE QUALITY\n")
            f.write("-" * 25 + "\n")
            if latent_results['cluster_quality']:
                cluster = latent_results['cluster_quality']
                f.write(f"Centroid distance: {cluster['centroid_distance']:.4f}\n")
                f.write(f"Separation ratio: {cluster['separation_ratio']:.4f}\n")
                f.write(f"Historical intra-cluster distance: {cluster['historical_intra_dist']:.4f}\n")
                f.write(f"Random intra-cluster distance: {cluster['random_intra_dist']:.4f}\n")
            f.write("\n")
        
        # Print summary to console
        print("\nEVALUATION SUMMARY:")
        print(f"• Generation validity: {generation_results['valid_combinations']}%")
        print(f"• Ensemble win rate: {ensemble_results.get('win_rate', 0):.2%}")
        print(f"• Reconstruction accuracy: {reconstruction_results['exact_match_rate']:.2%}")
        print(f"• Report saved to: {save_path}")
        
        return {
            'generation': generation_results,
            'ensemble': ensemble_results,
            'reconstruction': reconstruction_results,
            'latent_space': latent_results
        }

def run_evaluation(use_i_ching=False):
    """
    Main evaluation pipeline for the CVAE system.
    """
    print("\n--- Starting CVAE Evaluation Pipeline ---")
    
    # Check if models exist
    if not os.path.exists(CONFIG["model_save_path"]) or not os.path.exists(CONFIG["meta_learner_save_path"]):
        print(f"\n[ERROR] Model files not found!")
        print("Please train the models first (Main Menu -> Option 1).")
        return
    
    device = torch.device(CONFIG['device'])
    print(f"Running evaluation on: {device}")
    
    # Load data
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
    
    # Load models
    print("Loading trained models...")
    feature_engineer = joblib.load(CONFIG["feature_engineer_path"])
    
    cvae_model = ConditionalVAE(CONFIG).to(device)
    cvae_model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device, weights_only=False))
    cvae_model.eval()
    
    meta_learner = AttentionMetaLearner(CONFIG).to(device)
    meta_learner.load_state_dict(torch.load(CONFIG["meta_learner_save_path"], map_location=device, weights_only=False))
    meta_learner.eval()
    
    # Create evaluator
    evaluator = CVAEEvaluator(cvae_model, meta_learner, feature_engineer, df, CONFIG)
    
    # Run comprehensive evaluation
    results = evaluator.generate_evaluation_report()
    
    return results