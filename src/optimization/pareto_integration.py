"""
Integration functions for applying Pareto Front parameters to training.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..infrastructure.logging.logger import get_logger
from ..config import CONFIG

logger = get_logger(__name__)


def load_pareto_parameters(pareto_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load Pareto Front parameters from file.
    
    Args:
        pareto_file: Specific Pareto Front file to load. If None, loads the most recent.
        
    Returns:
        Dictionary of parameters or None if not found.
    """
    try:
        if pareto_file and Path(pareto_file).exists():
            # Load specific file
            with open(pareto_file, 'r') as f:
                data = json.load(f)
            
            if 'selected_parameters' in data:
                return data['selected_parameters']
            elif 'parameters' in data:
                return data['parameters']
            elif isinstance(data, dict) and 'learning_rate' in data:
                return data
            else:
                logger.warning(f"No parameters found in {pareto_file}")
                return None
        
        # Look for saved parameters in standard locations
        search_paths = [
            "models/pareto_optimized/selected_parameters.json",  # Primary location
            "models/best_parameters/latest_pareto_selection.json",
            "models/best_parameters/selected_pareto_params.json", 
            "models/best_parameters/pareto_selected_*.json",
            "models/best_parameters/pareto_front_nsga2_*.json",
            "models/best_parameters/pareto_front_tpe_*.json",
            "optimization_results/best_parameters.json",
            "thorough_search_results/best_parameters.json"
        ]
        
        for pattern in search_paths:
            if '*' in pattern:
                # Handle glob pattern
                from glob import glob
                files = glob(pattern)
                if files:
                    # Get most recent file
                    most_recent = max(files, key=os.path.getmtime)
                    return load_pareto_parameters(most_recent)
            else:
                # Direct file check
                if Path(pattern).exists():
                    return load_pareto_parameters(pattern)
        
        logger.info("No Pareto Front parameters found")
        return None
        
    except Exception as e:
        logger.error(f"Error loading Pareto Front parameters: {e}")
        return None


def apply_pareto_parameters_to_config(
    base_config: Dict[str, Any], 
    pareto_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply Pareto Front parameters to a base configuration.
    
    Args:
        base_config: Base configuration dictionary
        pareto_params: Parameters from Pareto Front optimization
        
    Returns:
        Updated configuration with Pareto Front parameters applied
    """
    config = base_config.copy()
    
    # Map Pareto Front parameter names to config keys
    param_mapping = {
        'learning_rate': 'learning_rate',
        'batch_size': 'batch_size',
        'hidden_size': 'hidden_size',
        'num_layers': 'num_layers',
        'dropout': 'dropout',
        'weight_decay': 'weight_decay',
        'kl_weight': 'kl_weight',
        'contrastive_weight': 'contrastive_weight',
        'use_batch_norm': 'use_batch_norm',
        'optimizer_type': 'optimizer_type',
        'gradient_clip_norm': 'gradient_clip_norm',
        'early_stopping_patience': 'early_stopping_patience'
    }
    
    # Apply parameters with constraints
    applied_params = []
    for param_name, param_value in pareto_params.items():
        if param_name in param_mapping:
            config_key = param_mapping[param_name]
            
            # Apply batch size constraints for optimal training
            if param_name == 'batch_size':
                # Prefer smaller batch sizes: clamp to range [8, 16] for better convergence
                original_value = param_value
                param_value = min(16, max(8, param_value))
                logger.info(f"Constrained batch_size for optimal training: {original_value} → {param_value} (range: 8-16)")
            
            config[config_key] = param_value
            applied_params.append(f"{param_name}={param_value}")
    
    if applied_params:
        logger.info(f"Applied Pareto Front parameters: {', '.join(applied_params)}")
    else:
        logger.warning("No recognized parameters found in Pareto Front data")
    
    return config


def create_pareto_optimized_config(pareto_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a configuration optimized with Pareto Front parameters.
    
    Args:
        pareto_params: Specific Pareto parameters to use. If None, loads from file.
        
    Returns:
        Configuration dictionary with Pareto Front optimization applied
    """
    # Start with optimized base configuration
    config = CONFIG.copy()
    # config.update({
       # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        #'epochs': 6,
        #'use_mixed_precision': True,
        #'early_stopping_patience': 4,
        #'gradient_clip_norm': 0.5035892758042647,
    #})
    
    # Load Pareto Front parameters if not provided
    if pareto_params is None:
        pareto_params = load_pareto_parameters()
    
    # Apply Pareto Front parameters if available
    if pareto_params:
        config = apply_pareto_parameters_to_config(config, pareto_params)
        config['_pareto_optimized'] = True
        config['_pareto_params'] = pareto_params
        logger.info("Configuration enhanced with Pareto Front optimization")
    else:
        logger.info("No Pareto Front parameters available, using default optimized config")
        config['_pareto_optimized'] = False
    
    return config


def list_available_pareto_results() -> List[Dict[str, Any]]:
    """
    List all available Pareto Front result files.
    
    Returns:
        List of dictionaries with file info
    """
    results = []
    
    # Check Pareto Front directories
    pareto_dirs = [
        "models/pareto_front/nsga2",
        "models/pareto_front/tpe"
    ]
    
    for pareto_dir in pareto_dirs:
        pareto_path = Path(pareto_dir)
        if pareto_path.exists():
            for result_file in pareto_path.glob("pareto_results_*.json"):
                try:
                    stat = result_file.stat()
                    results.append({
                        'file': str(result_file),
                        'algorithm': pareto_path.name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'name': result_file.name
                    })
                except Exception as e:
                    logger.warning(f"Error reading {result_file}: {e}")
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x['modified'], reverse=True)
    
    return results


def save_pareto_selection_for_training(parameters: Dict[str, Any], algorithm: str) -> str:
    """
    Save selected Pareto Front parameters for easy loading by training system.
    
    Args:
        parameters: Selected parameters from Pareto Front
        algorithm: Algorithm that generated the parameters
        
    Returns:
        Path to saved file
    """
    output_dir = Path("models/best_parameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_file = output_dir / f"pareto_selected_{algorithm}_{timestamp}.json"
    
    data = {
        'algorithm': algorithm,
        'timestamp': timestamp,
        'selected_parameters': parameters,
        'note': 'Parameters selected from Pareto Front for training'
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also save as latest selection
    latest_file = output_dir / "latest_pareto_selection.json"
    with open(latest_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Pareto Front selection saved to {output_file}")
    return str(output_file)


# Import torch only when needed
def get_torch():
    try:
        import torch
        return torch
    except ImportError:
        logger.error("PyTorch not available")
        return None


# Make torch available conditionally
torch = get_torch()