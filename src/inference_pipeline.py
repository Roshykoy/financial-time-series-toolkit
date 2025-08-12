# src/inference_pipeline.py
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import os
from collections import defaultdict
import random

from src.config import CONFIG
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.feature_engineering import FeatureEngineer
from src.temporal_scorer import TemporalScorer
from src.i_ching_scorer import IChingScorer

class GenerativeEnsemble:
    """
    New ensemble system that combines:
    1. CVAE-generated combinations
    2. Legacy heuristic scorers 
    3. Meta-learner for dynamic weighting
    """
    
    def __init__(self, cvae_model, meta_learner, temporal_scorer, i_ching_scorer, 
                 feature_engineer, df, config, use_i_ching=False):
        self.cvae_model = cvae_model
        self.meta_learner = meta_learner
        self.temporal_scorer = temporal_scorer
        self.i_ching_scorer = i_ching_scorer
        self.feature_engineer = feature_engineer
        self.df = df
        self.config = config
        self.use_i_ching = use_i_ching
        self.device = next(cvae_model.parameters()).device
        
        # Set models to evaluation mode
        self.cvae_model.eval()
        self.meta_learner.eval()
        
    def generate_candidates(self, num_candidates=50, temperature=0.8, diversity_factor=1.0):
        """
        Generate candidate number combinations using the CVAE.
        
        Args:
            num_candidates: Number of combinations to generate
            temperature: Sampling temperature (higher = more diverse)
            diversity_factor: Factor to encourage diversity in latent sampling
        
        Returns:
            candidates: List of unique 6-number combinations
            generation_scores: CVAE probability scores for each combination
        """
        try:
            with torch.no_grad():
                # Get current temporal context
                current_index = len(self.df)  # Hypothetical next draw
                sequence = self.cvae_model.temporal_encoder.prepare_sequence_data(
                    self.df, current_index
                ).to(self.device)
                context, trend_features = self.cvae_model.temporal_encoder(sequence)
                
                # Generate multiple batches for diversity
                candidates = []
                generation_scores = []
                
                batch_size = min(10, num_candidates)
                num_batches = (num_candidates + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    current_batch_size = min(batch_size, num_candidates - len(candidates))
                    
                    # Add diversity to context for different batches
                    if diversity_factor > 0 and batch_idx > 0:
                        noise = torch.randn_like(context) * diversity_factor * 0.1
                        context_with_noise = context + noise
                    else:
                        context_with_noise = context
                    
                    # Expand context for batch
                    batch_context = context_with_noise.expand(current_batch_size, -1)
                    
                    # Generate combinations
                    batch_combinations, batch_log_probs = self.cvae_model.generate(
                        batch_context,
                        num_samples=1,
                        temperature=temperature
                    )
                    
                    # Convert to list format and filter duplicates
                    for i in range(current_batch_size):
                        combo = batch_combinations[i].cpu().numpy().tolist()
                        combo_tuple = tuple(sorted(combo))
                        
                        # Check for duplicates and validity
                        if (combo_tuple not in [tuple(sorted(c)) for c in candidates] and
                            len(set(combo)) == 6 and
                            all(1 <= num <= self.config['num_lotto_numbers'] for num in combo)):
                            
                            candidates.append(sorted(combo))
                            generation_scores.append(batch_log_probs[i].item())
                
                # If we don't have enough unique candidates, generate more
                max_attempts = num_candidates * 3  # Prevent infinite loop
                attempts = 0
                
                while len(candidates) < num_candidates and attempts < max_attempts:
                    batch_context = context.expand(1, -1)
                    combination, log_prob = self.cvae_model.generate(
                        batch_context, num_samples=1, temperature=temperature * 1.2
                    )
                    
                    combo = combination[0].cpu().numpy().tolist()
                    combo_tuple = tuple(sorted(combo))
                    
                    if (combo_tuple not in [tuple(sorted(c)) for c in candidates] and
                        len(set(combo)) == 6 and
                        all(1 <= num <= self.config['num_lotto_numbers'] for num in combo)):
                        
                        candidates.append(sorted(combo))
                        generation_scores.append(log_prob[0].item())
                    
                    attempts += 1
                
                # If still not enough candidates, fill with random valid combinations
                if len(candidates) < num_candidates:
                    print(f"Warning: Only generated {len(candidates)}/{num_candidates} unique candidates. Filling with random combinations.")
                    while len(candidates) < num_candidates:
                        random_combo = sorted(random.sample(range(1, self.config['num_lotto_numbers'] + 1), 6))
                        if tuple(random_combo) not in [tuple(sorted(c)) for c in candidates]:
                            candidates.append(random_combo)
                            generation_scores.append(0.0)  # Assign neutral score to random combinations
                
                return candidates[:num_candidates], generation_scores[:num_candidates]
                
        except Exception as e:
            print(f"Error in generate_candidates: {e}")
            # Fallback to random combinations
            print("Falling back to random combination generation...")
            candidates = []
            generation_scores = []
            
            while len(candidates) < num_candidates:
                random_combo = sorted(random.sample(range(1, self.config['num_lotto_numbers'] + 1), 6))
                if tuple(random_combo) not in [tuple(sorted(c)) for c in candidates]:
                    candidates.append(random_combo)
                    generation_scores.append(0.0)
            
            return candidates, generation_scores
    
    def score_candidates(self, candidates):
        """
        Score candidates using all available scoring methods.
        
        Args:
            candidates: List of 6-number combinations
        
        Returns:
            scores_dict: Dictionary with scores from each method
        """
        scores_dict = {
            'generative': [],
            'temporal': [],
            'i_ching': []
        }
        
        current_index = len(self.df)
        
        for combination in candidates:
            try:
                # Generative score (from CVAE)
                with torch.no_grad():
                    # Get temporal context
                    sequence = self.cvae_model.temporal_encoder.prepare_sequence_data(
                        self.df, current_index
                    ).to(self.device)
                    context, _ = self.cvae_model.temporal_encoder(sequence)
                    
                    # Encode combination
                    mu, logvar = self.cvae_model.encode([combination], self.feature_engineer.pair_counts)
                    z = self.cvae_model.reparameterize(mu, logvar)
                    
                    # Get generative probability
                    reconstruction_logits, probs = self.cvae_model.decoder(z, context)
                    
                    # Compute likelihood of this specific combination
                    target_tensor = torch.tensor([combination], device=self.device) - 1  # 0-based
                    combo_likelihood = 1.0
                    for pos in range(6):
                        prob_val = probs[0, pos, target_tensor[0, pos]].item()
                        combo_likelihood *= max(prob_val, 1e-8)  # Prevent zero probability
                    
                    scores_dict['generative'].append(float(np.log(combo_likelihood + 1e-8)))
            except Exception as e:
                print(f"Warning: Error computing generative score for {combination}: {e}")
                scores_dict['generative'].append(0.0)
            
            try:
                # Temporal score
                temporal_score = self.temporal_scorer.score(combination, current_index)
                scores_dict['temporal'].append(temporal_score)
            except Exception as e:
                print(f"Warning: Error computing temporal score for {combination}: {e}")
                scores_dict['temporal'].append(0.0)
            
            try:
                # I-Ching score
                if self.use_i_ching and self.i_ching_scorer:
                    i_ching_score = self.i_ching_scorer.score(combination)
                    scores_dict['i_ching'].append(i_ching_score)
                else:
                    scores_dict['i_ching'].append(0.0)
            except Exception as e:
                print(f"Warning: Error computing I-Ching score for {combination}: {e}")
                scores_dict['i_ching'].append(0.0)
        
        return scores_dict
    
    def ensemble_rerank(self, candidates, scores_dict):
        """
        Re-rank candidates using the meta-learner for dynamic ensemble weights.
        
        Args:
            candidates: List of number combinations
            scores_dict: Dictionary of scores from different methods
        
        Returns:
            ranked_candidates: Candidates sorted by final ensemble score
            final_scores: Final ensemble scores
            explanations: Weight explanations from meta-learner
        """
        try:
            # Get temporal context for meta-learner
            current_index = len(self.df)
            sequence = self.cvae_model.temporal_encoder.prepare_sequence_data(
                self.df, current_index
            ).to(self.device)
            temporal_context, _ = self.cvae_model.temporal_encoder(sequence)
            
            # Expand context for all candidates
            batch_context = temporal_context.expand(len(candidates), -1)
            
            # Convert scores to tensors
            scorer_scores = {}
            for method in ['generative', 'temporal', 'i_ching']:
                scorer_scores[method] = torch.tensor(
                    scores_dict[method], device=self.device, dtype=torch.float32
                )
            
            # Get ensemble weights and final scores
            with torch.no_grad():
                ensemble_weights, final_scores, confidence = self.meta_learner(
                    candidates, batch_context, scorer_scores
                )
                
                # Get explanations
                explanations = self.meta_learner.get_weight_explanations(
                    candidates, batch_context, scorer_scores
                )
            
            # Sort candidates by final scores
            scored_candidates = list(zip(candidates, final_scores.cpu().numpy(), explanations))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            ranked_candidates = [item[0] for item in scored_candidates]
            final_scores_sorted = [item[1] for item in scored_candidates]
            explanations_sorted = [item[2] for item in scored_candidates]
            
            return ranked_candidates, final_scores_sorted, explanations_sorted
            
        except Exception as e:
            print(f"Warning: Error in ensemble re-ranking: {e}")
            print("Using simple scoring fallback...")
            
            # Fallback to simple weighted average
            weights = self.config['ensemble_weights']
            final_scores = []
            
            for i in range(len(candidates)):
                score = (weights['generative'] * scores_dict['generative'][i] +
                        weights['temporal'] * scores_dict['temporal'][i] +
                        (weights['i_ching'] * scores_dict['i_ching'][i] if self.use_i_ching else 0))
                final_scores.append(score)
            
            # Sort by scores
            scored_candidates = list(zip(candidates, final_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            ranked_candidates = [item[0] for item in scored_candidates]
            final_scores_sorted = [item[1] for item in scored_candidates]
            
            # Create dummy explanations
            explanations_sorted = [{
                'weights': weights,
                'confidence': 0.5,
                'reasoning': "Fallback scoring due to meta-learner error"
            } for _ in candidates]
            
            return ranked_candidates, final_scores_sorted, explanations_sorted
    
    def generate_recommendations(self, num_sets, temperature=0.8, verbose=True):
        """
        Main method to generate final recommendations.
        
        Args:
            num_sets: Number of final recommendations to return
            temperature: Generation temperature
            verbose: Whether to print progress
        
        Returns:
            recommendations: Final recommended combinations
            detailed_results: Detailed scoring information
        """
        if verbose:
            print(f"Generating {num_sets} recommendations using CVAE + Ensemble...")
        
        try:
            # Step 1: Generate candidate combinations
            if verbose:
                print("Step 1: Generating candidate combinations...")
            
            num_candidates = max(num_sets * 5, 50)  # Generate more candidates than needed
            candidates, generation_scores = self.generate_candidates(
                num_candidates=num_candidates,
                temperature=temperature
            )
            
            if verbose:
                print(f"Generated {len(candidates)} unique candidates")
            
            # Step 2: Score all candidates
            if verbose:
                print("Step 2: Scoring candidates with all methods...")
            
            scores_dict = self.score_candidates(candidates)
            
            # Step 3: Ensemble re-ranking
            if verbose:
                print("Step 3: Re-ranking with meta-learner...")
            
            ranked_candidates, final_scores, explanations = self.ensemble_rerank(
                candidates, scores_dict
            )
            
            # Step 4: Select top recommendations
            recommendations = ranked_candidates[:num_sets]
            
            # Prepare detailed results
            detailed_results = []
            for i in range(min(num_sets, len(ranked_candidates))):
                result = {
                    'combination': recommendations[i],
                    'final_score': final_scores[i],
                    'individual_scores': {
                        'generative': scores_dict['generative'][candidates.index(recommendations[i])],
                        'temporal': scores_dict['temporal'][candidates.index(recommendations[i])],
                        'i_ching': scores_dict['i_ching'][candidates.index(recommendations[i])] if self.use_i_ching else 0.0
                    },
                    'explanation': explanations[i],
                    'rank': i + 1
                }
                detailed_results.append(result)
            
            if verbose:
                print(f"Generated {len(recommendations)} final recommendations")
            
            return recommendations, detailed_results
            
        except Exception as e:
            print(f"Error in generate_recommendations: {e}")
            print("Generating fallback random recommendations...")
            
            # Fallback to random recommendations
            recommendations = []
            for i in range(num_sets):
                random_combo = sorted(random.sample(range(1, self.config['num_lotto_numbers'] + 1), 6))
                recommendations.append(random_combo)
            
            detailed_results = [{
                'combination': combo,
                'final_score': 0.0,
                'individual_scores': {'generative': 0.0, 'temporal': 0.0, 'i_ching': 0.0},
                'explanation': {'weights': {}, 'confidence': 0.0, 'reasoning': 'Fallback random generation'},
                'rank': i + 1
            } for i, combo in enumerate(recommendations)]
            
            return recommendations, detailed_results

def find_latest_model():
    """
    Find the most recently trained model by checking file timestamps.
    
    Returns:
        tuple: (cvae_model_path, meta_learner_path, feature_engineer_path, model_type)
    """
    # Define all possible model paths with their corresponding artifacts
    # PRIORITY ORDER: Standard CONFIG paths first (from latest training), then backups
    model_candidates = [
        # (cvae_path, meta_learner_path, feature_engineer_path, model_type)
        # 1. Standard CONFIG paths (used by optimized training and should be preferred)
        ("models/conservative_cvae_model.pth", "models/conservative_meta_learner.pth", "models/conservative_feature_engineer.pkl", "standard_optimized"),
        # 2. Thorough search results (high-quality optimized models)
        ("thorough_search_results/best_cvae_model.pth", "thorough_search_results/best_meta_learner.pth", "thorough_search_results/best_feature_engineer.pkl", "thorough_search_optimized"),
        # 3. Backup models from recent training
        ("models/best_cvae_model.pth", "models/best_meta_learner.pth", "models/best_feature_engineer.pkl", "best_backup"),
        # 4. Quick training models
        ("models/quick_cvae_model.pth", "models/quick_meta_learner.pth", "models/quick_feature_engineer.pkl", "quick"),
        # 5. Other potential model locations
        ("models/optimized_cvae_model.pth", "models/optimized_meta_learner.pth", "models/optimized_feature_engineer.pkl", "optimized"),
        ("models/ultra_quick_model.pth", "models/ultra_quick_meta_learner.pth", "models/ultra_quick_feature_engineer.pkl", "ultra_quick"),
        ("models/cvae_model.pth", "models/meta_learner.pth", "models/feature_engineer.pkl", "legacy_standard"),
    ]
    
    # First, check if the standard CONFIG paths exist (highest priority)
    standard_paths = [
        (CONFIG["model_save_path"], CONFIG["meta_learner_save_path"], CONFIG["feature_engineer_path"], "current_optimized")
    ]
    
    # Check if standard CONFIG models exist and are complete
    for cvae_path, meta_path, fe_path, model_type in standard_paths:
        if os.path.exists(cvae_path) and os.path.exists(meta_path) and os.path.exists(fe_path):
            print(f"🎯 Using current optimized models from CONFIG paths")
            return cvae_path, meta_path, fe_path, model_type
    
    # If standard CONFIG paths don't have complete models, fall back to discovery
    existing_models = []
    for cvae_path, meta_path, fe_path, model_type in model_candidates:
        if os.path.exists(cvae_path):
            # Get the timestamp of the CVAE model file
            cvae_mtime = os.path.getmtime(cvae_path)
            
            # Check if meta-learner exists (if not, use a fallback)
            if not os.path.exists(meta_path):
                # Look for alternative meta-learner paths
                alt_meta_paths = [
                    "models/best_meta_learner.pth",
                    "models/meta_learner.pth",
                    "thorough_search_results/best_meta_learner.pth"
                ]
                for alt_meta in alt_meta_paths:
                    if os.path.exists(alt_meta):
                        meta_path = alt_meta
                        break
            
            # Check if feature engineer exists (if not, use a fallback)
            if not os.path.exists(fe_path):
                # Look for alternative feature engineer paths
                alt_fe_paths = [
                    "models/best_feature_engineer.pkl",
                    "models/feature_engineer.pkl",
                    "thorough_search_results/best_feature_engineer.pkl"
                ]
                for alt_fe in alt_fe_paths:
                    if os.path.exists(alt_fe):
                        fe_path = alt_fe
                        break
            
            existing_models.append((cvae_path, meta_path, fe_path, model_type, cvae_mtime))
    
    if not existing_models:
        return None, None, None, None
    
    # Sort by modification time (newest first)
    existing_models.sort(key=lambda x: x[4], reverse=True)
    
    # Return the most recent model
    return existing_models[0][:4]  # Exclude timestamp from return

def run_inference(num_sets_to_generate, use_i_ching=False, temperature=0.8, verbose=True):
    """
    Main inference pipeline using the new generative approach.
    
    Args:
        num_sets_to_generate: Number of combinations to recommend
        use_i_ching: Whether to include I-Ching scorer
        temperature: Generation temperature (higher = more diverse)
        verbose: Whether to print detailed progress
    """
    print("\n--- Starting Generative Inference Pipeline ---")
    print("Architecture: CVAE + Meta-Learner + Ensemble Scoring")
    
    # Use dynamic model discovery
    print("\n🔍 Searching for latest trained model...")
    cvae_path, meta_path, fe_path, model_type = find_latest_model()
    
    if cvae_path is None:
        print(f"\n[ERROR] No trained models found!")
        print("Please train a model first (Main Menu -> Option 1).")
        return None, None
    
    print(f"✅ Found latest model: {model_type}")
    print(f"📁 CVAE model: {cvae_path}")
    print(f"📁 Meta-learner: {meta_path}")
    print(f"📁 Feature engineer: {fe_path}")
    
    device = torch.device(CONFIG['device'])
    print(f"Running inference on: {device}")
    
    try:
        # Load data
        print("Loading historical data...")
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
        
        # Load models using dynamic discovery
        print("Loading trained models...")
        
        # Load feature engineer
        feature_engineer = None
        if os.path.exists(fe_path):
            try:
                feature_engineer = joblib.load(fe_path)
                print(f"✅ Loaded feature engineer from: {fe_path}")
            except Exception as e:
                print(f"⚠️ Failed to load feature engineer from {fe_path}: {e}")
        
        if feature_engineer is None:
            print("⚠️ No feature engineer found, creating new one...")
            from src.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer()
            feature_engineer.fit(df)
            print("✅ Created and fitted new feature engineer")
        
        # Load CVAE model
        print(f"Loading CVAE model: {cvae_path}")
        model_data = torch.load(cvae_path, map_location=device, weights_only=False)
        
        if 'cvae_state_dict' in model_data:
            # New format with config
            if 'config' in model_data:
                model_config = model_data['config']
                print(f"✅ Using saved training configuration")
                print(f"   Hidden size: {model_config.get('hidden_size', 'unknown')}")
                print(f"   Latent dim: {model_config.get('latent_dim', 'unknown')}")
            else:
                model_config = CONFIG
                print("⚠️ No saved config found, using default")
            
            # Create model with correct configuration
            cvae_model = ConditionalVAE(model_config).to(device)
            cvae_model.load_state_dict(model_data['cvae_state_dict'])
            print("✅ Loaded CVAE from model (new format)")
            
            # Load meta-learner from same file if available
            meta_learner = AttentionMetaLearner(model_config).to(device)
            if 'meta_learner_state_dict' in model_data:
                meta_learner.load_state_dict(model_data['meta_learner_state_dict'])
                print("✅ Loaded meta-learner from same model file")
            else:
                # Load meta-learner from separate file
                if os.path.exists(meta_path):
                    try:
                        meta_learner.load_state_dict(torch.load(meta_path, map_location=device, weights_only=False))
                        print(f"✅ Loaded meta-learner from: {meta_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to load meta-learner from {meta_path}: {e}")
                        print("⚠️ Using default meta-learner")
                else:
                    print("⚠️ Meta-learner file not found, using default")
        else:
            # Old format - single state dict, use default config
            cvae_model = ConditionalVAE(CONFIG).to(device)
            cvae_model.load_state_dict(model_data)
            print("✅ Loaded CVAE from model (old format)")
            
            # Load meta-learner from separate file
            meta_learner = AttentionMetaLearner(CONFIG).to(device)
            if os.path.exists(meta_path):
                try:
                    meta_learner.load_state_dict(torch.load(meta_path, map_location=device, weights_only=False))
                    print(f"✅ Loaded meta-learner from: {meta_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load meta-learner from {meta_path}: {e}")
                    print("⚠️ Using default meta-learner")
            else:
                print("⚠️ Meta-learner file not found, using default")
        
        cvae_model.eval()
        meta_learner.eval()
        
        # Initialize heuristic scorers
        print("Initializing heuristic scorers...")
        temporal_scorer = TemporalScorer(CONFIG)
        temporal_scorer.fit(df)
        
        i_ching_scorer = IChingScorer(CONFIG) if use_i_ching else None
        
        # Create ensemble
        ensemble = GenerativeEnsemble(
            cvae_model=cvae_model,
            meta_learner=meta_learner,
            temporal_scorer=temporal_scorer,
            i_ching_scorer=i_ching_scorer,
            feature_engineer=feature_engineer,
            df=df,
            config=CONFIG,
            use_i_ching=use_i_ching
        )
        
        # Generate recommendations
        print(f"\nGenerating {num_sets_to_generate} number combinations...")
        print(f"Using I-Ching scorer: {'Yes' if use_i_ching else 'No'}")
        print(f"Generation temperature: {temperature}")
        
        recommendations, detailed_results = ensemble.generate_recommendations(
            num_sets=num_sets_to_generate,
            temperature=temperature,
            verbose=verbose
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("RECOMMENDED NUMBER COMBINATIONS")
        print("=" * 80)
        
        for i, result in enumerate(detailed_results):
            combination = result['combination']
            final_score = result['final_score']
            individual_scores = result['individual_scores']
            explanation = result['explanation']
            
            print(f"\nRank {i+1}: {combination}")
            print(f"  Final Score: {final_score:.4f}")
            print(f"  Individual Scores:")
            print(f"    • Generative (CVAE): {individual_scores['generative']:.4f}")
            print(f"    • Temporal: {individual_scores['temporal']:.4f}")
            if use_i_ching:
                print(f"    • I-Ching: {individual_scores['i_ching']:.4f}")
            print(f"  Ensemble Weights: {explanation['weights']}")
            print(f"  Confidence: {explanation['confidence']:.3f}")
            print(f"  Analysis: {explanation['reasoning']}")
        
        print("\n" + "=" * 80)
        
        # Save results to file
        save_results_to_file(detailed_results, use_i_ching, temperature)
        
        return recommendations, detailed_results
        
    except Exception as e:
        print(f"\n[ERROR] Critical error in inference pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_results_to_file(detailed_results, use_i_ching, temperature):
    """Save inference results to a file for future reference."""
    try:
        os.makedirs("outputs", exist_ok=True)
        
        filename = f"outputs/recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("MARK SIX LOTTERY RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"Using I-Ching: {'Yes' if use_i_ching else 'No'}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Architecture: CVAE + Meta-Learner + Ensemble\n")
            f.write("\n")
            
            for result in detailed_results:
                f.write(f"Rank {result['rank']}: {result['combination']}\n")
                f.write(f"  Final Score: {result['final_score']:.4f}\n")
                f.write(f"  Generative: {result['individual_scores']['generative']:.4f}\n")
                f.write(f"  Temporal: {result['individual_scores']['temporal']:.4f}\n")
                if use_i_ching:
                    f.write(f"  I-Ching: {result['individual_scores']['i_ching']:.4f}\n")
                f.write(f"  Reasoning: {result['explanation']['reasoning']}\n")
                f.write("\n")
        
        print(f"Results saved to: {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {e}")

# Legacy compatibility functions (for old scoring system)
class ScorerEnsemble:
    """Legacy ensemble class for backward compatibility."""
    
    def __init__(self, model, fe, temporal_scorer, i_ching_scorer, df_all_draws, config, use_i_ching=False):
        print("[WARNING] Using legacy ScorerEnsemble. Consider upgrading to GenerativeEnsemble.")
        self.config = config
        
    def score(self, number_set):
        print("[WARNING] Legacy scoring called. This functionality is deprecated.")
        return 0.5

def local_search(initial_set, scorer_ensemble, max_iterations, num_neighbors):
    """Legacy local search function for backward compatibility."""
    print("[WARNING] Local search is deprecated. Use generative inference instead.")
    return initial_set, 0.5