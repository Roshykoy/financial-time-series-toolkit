# src/cvae_data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from collections import Counter
from tqdm import tqdm

class CVAEDataset(Dataset):
    """
    Dataset class for CVAE training that provides:
    - Historical winning combinations
    - Negative sample pools
    - Temporal context information
    - Statistical features
    """
    
    def __init__(self, df, feature_engineer, config, negative_pool=None, is_training=True):
        self.df = df
        self.feature_engineer = feature_engineer
        self.config = config
        self.is_training = is_training
        
        # Extract winning number columns
        self.winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        
        # Build negative pool if not provided
        if negative_pool is None:
            self.negative_pool = self._build_negative_pool()
        else:
            self.negative_pool = negative_pool
        
        # Precompute temporal sequences for efficiency
        self._precompute_temporal_sequences()
        
    def _build_negative_pool(self):
        """Builds a pool of negative samples (non-winning combinations)."""
        print("Building negative sample pool for CVAE training...")
        
        # Get all historical winning combinations
        historical_sets = set()
        for _, row in self.df.iterrows():
            combination = tuple(sorted(row[self.winning_num_cols].astype(int).tolist()))
            historical_sets.add(combination)
        
        negative_pool = []
        target_size = self.config['negative_pool_size']
        
        with tqdm(total=target_size, desc="Generating negatives") as pbar:
            while len(negative_pool) < target_size:
                # Generate random combination
                candidate = tuple(sorted(random.sample(
                    range(1, self.config['num_lotto_numbers'] + 1), 6
                )))
                
                # Only add if it's not a historical winner
                if candidate not in historical_sets:
                    negative_pool.append(list(candidate))
                    pbar.update(1)
        
        print(f"Generated {len(negative_pool)} negative samples")
        return negative_pool
    
    def _precompute_temporal_sequences(self):
        """Precomputes temporal sequences for each draw to speed up training."""
        print("Precomputing temporal sequences...")
        self.temporal_sequences = {}
        
        sequence_length = self.config['temporal_sequence_length']
        
        for idx in tqdm(range(len(self.df)), desc="Computing sequences"):
            # Get sequence of draws before current index
            start_idx = max(0, idx - sequence_length)
            end_idx = idx
            
            if start_idx == end_idx:
                # Beginning of dataset - create dummy sequence
                sequence = np.zeros((sequence_length, 6), dtype=int)
            else:
                # Extract actual sequence
                sequence_data = self.df.iloc[start_idx:end_idx][self.winning_num_cols].values
                
                # Pad if necessary
                if len(sequence_data) < sequence_length:
                    padding = np.zeros((sequence_length - len(sequence_data), 6), dtype=int)
                    sequence = np.vstack([padding, sequence_data])
                else:
                    sequence = sequence_data[-sequence_length:]
            
            self.temporal_sequences[idx] = torch.tensor(sequence, dtype=torch.long)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns a training sample with all necessary components.
        """
        # Get the positive combination (historical winner)
        row = self.df.iloc[idx]
        positive_combination = row[self.winning_num_cols].astype(int).tolist()
        
        # Get temporal sequence
        temporal_sequence = self.temporal_sequences[idx]
        
        # Sample negative combinations for this batch item
        num_negatives = self.config['negative_samples']
        sampled_negatives = random.sample(self.negative_pool, 
                                        min(num_negatives, len(self.negative_pool)))
        
        # Get pair counts from feature engineer
        pair_counts = self.feature_engineer.pair_counts
        
        return {
            'positive_combination': positive_combination,
            'temporal_sequence': temporal_sequence,
            'negative_samples': sampled_negatives,
            'pair_counts': pair_counts,
            'draw_index': idx,
            'date': row['Date'] if 'Date' in row else None
        }

def collate_cvae_batch(batch):
    """
    Custom collate function for CVAE training batches.
    Groups together all the components needed for training.
    """
    positive_combinations = [item['positive_combination'] for item in batch]
    temporal_sequences = torch.stack([item['temporal_sequence'] for item in batch])
    draw_indices = [item['draw_index'] for item in batch]
    
    # Collect all negative samples
    all_negatives = []
    for item in batch:
        all_negatives.extend(item['negative_samples'])
    
    # Get pair counts (same for all items in batch)
    pair_counts = batch[0]['pair_counts']
    
    return {
        'positive_combinations': positive_combinations,
        'temporal_sequences': temporal_sequences,
        'negative_pool': all_negatives,
        'pair_counts': pair_counts,
        'current_indices': draw_indices,
        'batch_size': len(batch)
    }

def create_cvae_data_loaders(df, feature_engineer, config):
    """
    Creates training and validation data loaders for CVAE.
    
    Args:
        df: Historical lottery data
        feature_engineer: Fitted feature engineering object
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Split data
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Creating CVAE datasets: {len(train_df)} train, {len(val_df)} validation")
    
    # Create datasets
    train_dataset = CVAEDataset(train_df, feature_engineer, config, is_training=True)
    
    # Share negative pool between train and validation for consistency
    val_dataset = CVAEDataset(val_df, feature_engineer, config, 
                             negative_pool=train_dataset.negative_pool, 
                             is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_cvae_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_cvae_batch,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

# Specialized batch class for easier handling
class CVAEBatch:
    """Wrapper class for CVAE training batches with convenient access methods."""
    
    def __init__(self, batch_dict, df, device):
        self.batch_dict = batch_dict
        self.df = df
        self.device = device
    
    @property
    def positive_combinations(self):
        return self.batch_dict['positive_combinations']
    
    @property
    def temporal_sequences(self):
        return self.batch_dict['temporal_sequences'].to(self.device)
    
    @property
    def negative_pool(self):
        return self.batch_dict['negative_pool']
    
    @property
    def pair_counts(self):
        return self.batch_dict['pair_counts']
    
    @property
    def current_indices(self):
        return self.batch_dict['current_indices']
    
    @property
    def batch_size(self):
        return self.batch_dict['batch_size']
    
    def to_device(self):
        """Moves appropriate tensors to the training device."""
        # Most data doesn't need to be moved as it's processed by the model
        return self
    
    def get_df_subset(self):
        """Returns the DataFrame subset relevant to this batch."""
        return self.df  # For simplicity, return full df (models handle indexing)