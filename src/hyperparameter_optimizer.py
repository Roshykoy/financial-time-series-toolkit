# src/hyperparameter_optimizer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import os
import joblib
import random
import itertools
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Tuple
import copy

from src.config import CONFIG
from src.model import ScoringModel
from src.engine import train_one_epoch, evaluate
from src.feature_engineering import FeatureEngineer
from src.sam import SAM

class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization system for the Mark Six AI project.
    Supports grid search, random search, and Bayesian optimization.
    """
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config.copy()
        self.optimization_history = []
        self.best_config = None
        self.best_score = float('-inf')
        
        # Define the parameter search space
        self.search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'hidden_size': [128, 256, 512, 768],
            'num_layers': [3, 4, 6, 8],
            'dropout': [0.1, 0.15, 0.2, 0.3],
            'batch_size': [32, 64, 128],
            'margin': [0.3, 0.5, 0.7, 1.0],
            'negative_samples': [16, 32, 64],
            'rho': [0.01, 0.05, 0.1],  # SAM parameter
            'use_sam_optimizer': [True, False]
        }
        
        # Create results directory
        self.results_dir = "hyperparameter_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def grid_search(self, max_combinations: int = 50, epochs_per_trial: int = 5):
        """
        Performs grid search over the hyperparameter space.
        
        Args:
            max_combinations: Maximum number of parameter combinations to try
            epochs_per_trial: Number of epochs to train each configuration
        """
        print(f"\n--- Starting Grid Search Hyperparameter Optimization ---")
        print(f"Max combinations: {max_combinations}")
        print(f"Epochs per trial: {epochs_per_trial}")
        
        # Generate all possible combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
            print(f"Randomly selected {max_combinations} combinations from {len(list(itertools.product(*param_values)))} total")
        
        print(f"Testing {len(all_combinations)} parameter combinations...")
        
        for i, combination in enumerate(tqdm(all_combinations, desc="Grid Search Progress")):
            # Create config for this combination
            trial_config = self.base_config.copy()
            for param_name, param_value in zip(param_names, combination):
                trial_config[param_name] = param_value
            
            print(f"\n--- Trial {i+1}/{len(all_combinations)} ---")
            print(f"Parameters: {dict(zip(param_names, combination))}")
            
            # Train and evaluate this configuration
            score = self._train_and_evaluate_config(trial_config, epochs_per_trial, trial_num=i+1)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                print(f"🎉 New best score: {score:.4f}")
            
            # Save progress
            self._save_trial_result(trial_config, score, i+1, "grid_search")
        
        self._save_optimization_summary("grid_search")
        return self.best_config, self.best_score
    
    def random_search(self, num_trials: int = 30, epochs_per_trial: int = 5):
        """
        Performs random search over the hyperparameter space.
        
        Args:
            num_trials: Number of random configurations to try
            epochs_per_trial: Number of epochs to train each configuration
        """
        print(f"\n--- Starting Random Search Hyperparameter Optimization ---")
        print(f"Number of trials: {num_trials}")
        print(f"Epochs per trial: {epochs_per_trial}")
        
        for i in range(num_trials):
            # Generate random configuration
            trial_config = self._generate_random_config()
            
            print(f"\n--- Trial {i+1}/{num_trials} ---")
            print(f"Parameters: {self._format_config_for_display(trial_config)}")
            
            # Train and evaluate this configuration
            score = self._train_and_evaluate_config(trial_config, epochs_per_trial, trial_num=i+1)
            
            # Update best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                print(f"🎉 New best score: {score:.4f}")
            
            # Save progress
            self._save_trial_result(trial_config, score, i+1, "random_search")
        
        self._save_optimization_summary("random_search")
        return self.best_config, self.best_score
    
    def bayesian_optimization(self, num_trials: int = 25, epochs_per_trial: int = 5):
        """
        Simple Bayesian optimization using random sampling with exploitation.
        (A more sophisticated implementation would use libraries like Optuna or scikit-optimize)
        """
        print(f"\n--- Starting Bayesian Optimization ---")
        print(f"Number of trials: {num_trials}")
        print(f"Epochs per trial: {epochs_per_trial}")
        
        # Initial random exploration phase
        exploration_trials = min(5, num_trials // 2)
        print(f"Initial exploration phase: {exploration_trials} trials")
        
        for i in range(exploration_trials):
            trial_config = self._generate_random_config()
            
            print(f"\n--- Exploration Trial {i+1}/{exploration_trials} ---")
            print(f"Parameters: {self._format_config_for_display(trial_config)}")
            
            score = self._train_and_evaluate_config(trial_config, epochs_per_trial, trial_num=i+1)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                print(f"🎉 New best score: {score:.4f}")
            
            self._save_trial_result(trial_config, score, i+1, "bayesian_exploration")
        
        # Exploitation phase - focus around best configurations
        exploitation_trials = num_trials - exploration_trials
        print(f"Exploitation phase: {exploitation_trials} trials")
        
        for i in range(exploitation_trials):
            # Generate config similar to best performing ones
            trial_config = self._generate_config_around_best()
            
            trial_num = exploration_trials + i + 1
            print(f"\n--- Exploitation Trial {trial_num}/{num_trials} ---")
            print(f"Parameters: {self._format_config_for_display(trial_config)}")
            
            score = self._train_and_evaluate_config(trial_config, epochs_per_trial, trial_num=trial_num)
            
            if score > self.best_score:
                self.best_score = score
                self.best_config = trial_config.copy()
                print(f"🎉 New best score: {score:.4f}")
            
            self._save_trial_result(trial_config, score, trial_num, "bayesian_exploitation")
        
        self._save_optimization_summary("bayesian_optimization")
        return self.best_config, self.best_score
    
    def _train_and_evaluate_config(self, config: Dict[str, Any], epochs: int, trial_num: int) -> float:
        """
        Trains and evaluates a model with the given configuration.
        Returns the validation score.
        """
        try:
            # Load and prepare data
            df = self._load_data()
            feature_engineer = FeatureEngineer()
            feature_engineer.fit(df)
            
            # Build negative pool (smaller for hyperparameter optimization)
            negative_pool = self._build_negative_pool(df, pool_size=5000, config=config)
            
            # Split data
            train_size = int(len(df) * 0.8)  # 80% for training, 20% for validation
            val_size = int(len(df) * 0.1)   # 10% for hyperparameter validation
            
            train_df = df.iloc[:train_size]
            val_df = df.iloc[train_size:train_size + val_size]
            
            # Create datasets and loaders
            train_dataset = ContrastiveDataset(train_df, feature_engineer, config, negative_pool)
            val_dataset = ContrastiveDataset(val_df, feature_engineer, config, negative_pool)
            
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
            
            # Initialize model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sample_features = feature_engineer.transform([1,2,3,4,5,6], 0)
            config['d_features'] = len(sample_features)
            
            model = ScoringModel(config).to(device)
            
            # Initialize optimizer
            if config['use_sam_optimizer']:
                base_optimizer = optim.AdamW
                optimizer = SAM(model.parameters(), base_optimizer, lr=config["learning_rate"], rho=config['rho'])
            else:
                optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 3  # Early stopping patience
            
            for epoch in range(epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
                val_loss = evaluate(model, val_loader, device, config)
                
                print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            # Return negative validation loss as score (higher is better)
            final_score = -best_val_loss
            print(f"  Final Score: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            print(f"  Error in trial {trial_num}: {str(e)}")
            return float('-inf')
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """Generates a random configuration from the search space."""
        config = self.base_config.copy()
        
        for param_name, param_values in self.search_space.items():
            config[param_name] = random.choice(param_values)
        
        return config
    
    def _generate_config_around_best(self) -> Dict[str, Any]:
        """Generates a configuration similar to the best performing ones."""
        if not self.optimization_history:
            return self._generate_random_config()
        
        # Get top 20% of configurations
        sorted_history = sorted(self.optimization_history, key=lambda x: x['score'], reverse=True)
        top_configs = sorted_history[:max(1, len(sorted_history) // 5)]
        
        # Start with a random top config
        base_config = random.choice(top_configs)['config'].copy()
        
        # Randomly modify 1-2 parameters
        params_to_modify = random.sample(list(self.search_space.keys()), random.randint(1, 2))
        
        for param in params_to_modify:
            if param in base_config:
                # Choose a value close to the current one
                current_value = base_config[param]
                available_values = self.search_space[param]
                
                if current_value in available_values:
                    current_idx = available_values.index(current_value)
                    # Choose from neighboring values
                    neighbor_indices = [
                        max(0, current_idx - 1),
                        current_idx,
                        min(len(available_values) - 1, current_idx + 1)
                    ]
                    base_config[param] = available_values[random.choice(neighbor_indices)]
                else:
                    base_config[param] = random.choice(available_values)
        
        return base_config
    
    def _load_data(self) -> pd.DataFrame:
        """Loads and preprocesses the Mark Six data."""
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        df = pd.read_csv(self.base_config["data_path"], header=None, skiprows=33, names=col_names)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        return df
    
    def _build_negative_pool(self, df: pd.DataFrame, pool_size: int, config: Dict[str, Any]) -> List[List[int]]:
        """Builds a pool of negative samples for training."""
        winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        historical_sets = {tuple(sorted(draw)) for draw in df[winning_num_cols].itertuples(index=False)}
        
        negative_pool = []
        while len(negative_pool) < pool_size:
            candidate = tuple(sorted(random.sample(range(1, config['num_lotto_numbers'] + 1), 6)))
            if candidate not in historical_sets:
                negative_pool.append(list(candidate))
        
        return negative_pool
    
    def _save_trial_result(self, config: Dict[str, Any], score: float, trial_num: int, method: str):
        """Saves the result of a trial."""
        result = {
            'trial_num': trial_num,
            'method': method,
            'config': config,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(result)
        
        # Save individual trial result
        filename = f"{self.results_dir}/trial_{trial_num}_{method}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _save_optimization_summary(self, method: str):
        """Saves a summary of the optimization process."""
        summary = {
            'method': method,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_trials': len(self.optimization_history),
            'timestamp': datetime.now().isoformat(),
            'all_results': self.optimization_history
        }
        
        filename = f"{self.results_dir}/optimization_summary_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n--- Optimization Summary ---")
        print(f"Method: {method}")
        print(f"Total trials: {len(self.optimization_history)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best configuration:")
        for key, value in self.best_config.items():
            if key in self.search_space:
                print(f"  {key}: {value}")
        print(f"Results saved to: {filename}")
    
    def _format_config_for_display(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Formats configuration for display, showing only tuned parameters."""
        display_config = {}
        for key in self.search_space.keys():
            if key in config:
                display_config[key] = config[key]
        return display_config
    
    def load_best_config(self, method: str = None) -> Dict[str, Any]:
        """Loads the best configuration from saved results."""
        if method:
            pattern = f"optimization_summary_{method}_*.json"
        else:
            pattern = "optimization_summary_*.json"
        
        import glob
        summary_files = glob.glob(f"{self.results_dir}/{pattern}")
        
        if not summary_files:
            print("No optimization summary files found.")
            return self.base_config
        
        # Get the most recent file
        latest_file = max(summary_files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Loaded best configuration from {latest_file}")
        print(f"Best score: {summary['best_score']:.4f}")
        
        return summary['best_config']

# Dataset class for hyperparameter optimization (same as in training pipeline)
class ContrastiveDataset(Dataset):
    def __init__(self, df, fe, config, negative_pool):
        self.df, self.fe, self.config, self.negative_pool = df, fe, config, negative_pool
        self.winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        positive_row = self.df.iloc[idx]
        positive_set = positive_row[self.winning_num_cols].astype(int).tolist()
        positive_features = self.fe.transform(positive_set, idx)
        neg_indices = np.random.choice(len(self.negative_pool), self.config['negative_samples'], replace=False)
        negative_features = np.array([self.fe.transform(self.negative_pool[i], idx) for i in neg_indices])

        return {"positive_features": torch.tensor(positive_features, dtype=torch.float32),
                "negative_features": torch.tensor(negative_features, dtype=torch.float32)}

def run_hyperparameter_optimization():
    """Main function to run hyperparameter optimization."""
    print("\n--- Hyperparameter Optimization Menu ---")
    print("1. Grid Search (Exhaustive, ~30-60 mins)")
    print("2. Random Search (Fast, ~15-30 mins)")
    print("3. Bayesian Optimization (Balanced, ~20-40 mins)")
    print("4. Load Best Configuration from Previous Run")
    print("5. Custom Quick Search (5 trials)")
    
    choice = input("Choose optimization method (1-5): ")
    
    optimizer = HyperparameterOptimizer(CONFIG)
    
    if choice == '1':
        print("Starting Grid Search...")
        max_combinations = int(input("Max combinations to try (recommended: 30-50): ") or "30")
        epochs_per_trial = int(input("Epochs per trial (recommended: 3-5): ") or "3")
        best_config, best_score = optimizer.grid_search(max_combinations, epochs_per_trial)
        
    elif choice == '2':
        print("Starting Random Search...")
        num_trials = int(input("Number of trials (recommended: 20-30): ") or "20")
        epochs_per_trial = int(input("Epochs per trial (recommended: 3-5): ") or "3")
        best_config, best_score = optimizer.random_search(num_trials, epochs_per_trial)
        
    elif choice == '3':
        print("Starting Bayesian Optimization...")
        num_trials = int(input("Number of trials (recommended: 15-25): ") or "20")
        epochs_per_trial = int(input("Epochs per trial (recommended: 3-5): ") or "3")
        best_config, best_score = optimizer.bayesian_optimization(num_trials, epochs_per_trial)
        
    elif choice == '4':
        method = input("Enter method name (grid_search/random_search/bayesian_optimization) or press Enter for latest: ").strip()
        best_config = optimizer.load_best_config(method if method else None)
        
        # Option to use this config for training
        if input("Train with this configuration? (y/n): ").lower() == 'y':
            print("Training with loaded configuration...")
            from src.training_pipeline import run_training
            
            # Temporarily update CONFIG with best parameters
            original_config = CONFIG.copy()
            CONFIG.update(best_config)
            
            try:
                run_training()
            finally:
                # Restore original config
                CONFIG.clear()
                CONFIG.update(original_config)
        return
        
    elif choice == '5':
        print("Starting Custom Quick Search...")
        best_config, best_score = optimizer.random_search(5, 2)  # 5 trials, 2 epochs each
        
    else:
        print("Invalid choice. Returning to main menu.")
        return
    
    print(f"\n🎉 Hyperparameter optimization completed!")
    print(f"Best score: {best_score:.4f}")
    print("\nBest configuration:")
    for key, value in best_config.items():
        if key in optimizer.search_space:
            print(f"  {key}: {value}")
    
    # Ask if user wants to train with the best configuration
    if input("\nTrain a full model with the best configuration? (y/n): ").lower() == 'y':
        print("Training with optimized hyperparameters...")
        
        # Update CONFIG with best parameters
        original_config = CONFIG.copy()
        CONFIG.update(best_config)
        
        try:
            from src.training_pipeline import run_training
            run_training()
        finally:
            # Restore original config
            CONFIG.clear()
            CONFIG.update(original_config)
        
        print("Training completed with optimized hyperparameters!")
    
    # Save best config to a separate file for easy loading
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print("Best hyperparameters saved to 'best_hyperparameters.json'")

if __name__ == "__main__":
    run_hyperparameter_optimization()