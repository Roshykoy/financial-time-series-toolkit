# src/cvae_data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import multiprocessing
import psutil
from collections import Counter
from tqdm import tqdm

# Phase 2 optimization imports
try:
    from src.optimization.parallel_feature_processor import ParallelFeatureProcessor
    from src.optimization.memory_pool_manager import get_memory_manager, MemoryPoolConfig
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False
    print("⚠️ Phase 2 optimizations not available, using standard processing")

def _get_optimal_num_workers():
    """Calculate optimal number of workers for DataLoader based on hardware."""
    try:
        cpu_count = multiprocessing.cpu_count()
        # Use CPU cores - 1 for optimal performance, minimum 1, maximum 8
        optimal_workers = max(1, min(cpu_count - 1, 8))
        return optimal_workers
    except Exception:
        return 2  # Fallback to safe default

def _get_smart_pin_memory():
    """Determine if pin_memory should be enabled based on system resources."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check available system RAM
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        # Only enable pin_memory if we have sufficient RAM (>4GB available)
        return available_ram_gb > 4.0
    except Exception:
        return torch.cuda.is_available()  # Fallback to basic CUDA check

def _calculate_optimal_batch_size(config):
    """
    Conservative batch size management for Pareto Front optimization.
    
    Only intervenes when:
    1. Batch size < 8 (minimum enforcement)
    2. OOM recovery is needed (handled separately in training pipeline)
    
    Args:
        config: Configuration dictionary with batch_size and optimization settings
        
    Returns:
        int: Adjusted batch size (minimum 8, otherwise unchanged)
    """
    base_batch_size = config.get('batch_size', 8)
    min_batch_size = 8  # Minimum viable batch size for model complexity
    
    # Conservative mode: only enforce minimum, don't scale up
    conservative_mode = config.get('conservative_batch_sizing', True)
    
    if conservative_mode:
        # Only enforce minimum constraint - don't interfere with Pareto optimization
        if base_batch_size < min_batch_size:
            print(f"🔧 Batch size constraint: {base_batch_size} → {min_batch_size} (minimum enforcement)")
            return min_batch_size
        else:
            # Leave batch size unchanged for Pareto Front testing
            return base_batch_size
    
    # Legacy aggressive optimization (when conservative_batch_sizing=False)
    if not torch.cuda.is_available() or config.get('optimized_batch_size') != 'auto':
        return max(min_batch_size, base_batch_size)
    
    try:
        # Get current GPU memory info
        if hasattr(torch.cuda, 'mem_get_info'):
            total_memory, free_memory = torch.cuda.mem_get_info()
            total_gb = total_memory / (1024**3)
            free_gb = free_memory / (1024**3)
            
            # Aggressive scaling (legacy behavior)
            if total_gb >= 8.0:  # High-end GPU
                scaling_factor = min(3.5, free_gb / 2.0)  # Conservative scaling
            elif total_gb >= 6.0:  # Mid-range GPU  
                scaling_factor = min(2.5, free_gb / 1.5)
            else:  # Lower-end GPU
                scaling_factor = min(2.0, free_gb / 1.0)
            
            optimal_batch_size = int(base_batch_size * scaling_factor)
            optimal_batch_size = min(optimal_batch_size, config.get('max_batch_size', 32))
            
            # Enforce minimum viable batch size
            optimal_batch_size = max(min_batch_size, optimal_batch_size)
            
            print(f"🚀 Hardware-optimized batch size: {base_batch_size} → {optimal_batch_size} "
                  f"(VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total, min={min_batch_size})")
            
            return optimal_batch_size
        else:
            return max(min_batch_size, base_batch_size)
    except Exception as e:
        print(f"⚠️  Batch size optimization failed: {e}, using minimum: {min_batch_size}")
        return min_batch_size

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
        """Precomputes temporal sequences with improved temporal integrity."""
        print("Precomputing temporal sequences with proper temporal ordering...")
        self.temporal_sequences = {}
        
        sequence_length = self.config['temporal_sequence_length']
        
        # Ensure we have Date column for proper temporal ordering
        if 'Date' in self.df.columns:
            # Sort by date to ensure proper temporal order
            df_sorted = self.df.sort_values('Date').reset_index(drop=True)
            print("Using date-sorted temporal sequences")
        else:
            df_sorted = self.df
            print("⚠️  Using row-order temporal sequences (not recommended)")
        
        for idx in tqdm(range(len(df_sorted)), desc="Computing temporal sequences"):
            # Get sequence of draws STRICTLY before current index (no data leakage)
            start_idx = max(0, idx - sequence_length)
            end_idx = idx  # Exclusive - current draw not included in its own sequence
            
            if start_idx == end_idx:
                # Beginning of dataset - create dummy sequence with padding
                sequence = np.ones((sequence_length, 6), dtype=int)  # Use 1s instead of 0s (valid lottery numbers)
                # Add some random variation to prevent model from learning padding pattern
                for i in range(sequence_length):
                    sequence[i] = sorted(np.random.choice(range(1, self.config['num_lotto_numbers'] + 1), 6, replace=False))
            else:
                # Extract actual historical sequence
                sequence_data = df_sorted.iloc[start_idx:end_idx][self.winning_num_cols].values
                
                # Pad if necessary (only at the beginning of the sequence)
                if len(sequence_data) < sequence_length:
                    needed_padding = sequence_length - len(sequence_data)
                    # Create realistic padding instead of zeros
                    padding = np.ones((needed_padding, 6), dtype=int)
                    for i in range(needed_padding):
                        padding[i] = sorted(np.random.choice(range(1, self.config['num_lotto_numbers'] + 1), 6, replace=False))
                    sequence = np.vstack([padding, sequence_data])
                else:
                    sequence = sequence_data[-sequence_length:]
            
            # Store the original index mapping for validation
            self.temporal_sequences[idx] = {
                'sequence': torch.tensor(sequence, dtype=torch.long),
                'original_idx': idx,
                'has_padding': start_idx == end_idx or len(sequence_data) < sequence_length if start_idx != end_idx else True
            }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns a training sample with improved temporal integrity.
        """
        # Get the positive combination (historical winner)
        row = self.df.iloc[idx]
        positive_combination = row[self.winning_num_cols].astype(int).tolist()
        
        # Validate combination
        if len(set(positive_combination)) != 6:
            print(f"⚠️  Invalid combination at index {idx}: {positive_combination}")
            # Create a valid combination as fallback
            positive_combination = sorted(random.sample(range(1, self.config['num_lotto_numbers'] + 1), 6))
        
        # Get temporal sequence with metadata
        temporal_data = self.temporal_sequences[idx]
        temporal_sequence = temporal_data['sequence'] if isinstance(temporal_data, dict) else temporal_data
        has_padding = temporal_data.get('has_padding', False) if isinstance(temporal_data, dict) else False
        
        # Sample negative combinations for this batch item
        num_negatives = self.config['negative_samples']
        sampled_negatives = random.sample(self.negative_pool, 
                                        min(num_negatives, len(self.negative_pool)))
        
        # Get pair counts from feature engineer
        pair_counts = self.feature_engineer.pair_counts
        
        # Add temporal validation info
        date_info = None
        if 'Date' in row:
            date_info = {
                'date': row['Date'],
                'has_temporal_padding': has_padding
            }
        
        return {
            'positive_combination': positive_combination,
            'temporal_sequence': temporal_sequence,
            'negative_samples': sampled_negatives,
            'pair_counts': pair_counts,
            'draw_index': idx,
            'date_info': date_info,
            'temporal_integrity': {
                'has_padding': has_padding,
                'sequence_length': len(temporal_sequence)
            }
        }

def collate_cvae_batch(batch):
    """
    Enhanced custom collate function for CVAE training batches with Phase 2 optimizations.
    Groups together all the components needed for training with parallel feature processing.
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
    
    # Phase 2 enhancement: Add batch metadata for parallel processing
    batch_metadata = {
        'batch_size': len(batch),
        'feature_ready': True,  # Mark that this batch can use parallel features
        'temporal_integrity': all(item.get('temporal_integrity', {}).get('has_padding', False) for item in batch)
    }
    
    return {
        'positive_combinations': positive_combinations,
        'temporal_sequences': temporal_sequences,
        'negative_pool': all_negatives,
        'pair_counts': pair_counts,
        'current_indices': draw_indices,
        'batch_size': len(batch),
        'batch_metadata': batch_metadata  # Phase 2 addition
    }


def enhanced_collate_cvae_batch_phase2(batch):
    """
    Phase 2 enhanced collate function with parallel feature processing integration.
    Uses parallel feature processor if available and enabled.
    """
    # Get basic batch data
    basic_batch = collate_cvae_batch(batch)
    
    # Check if Phase 2 parallel processing is available
    if hasattr(batch[0], '_parallel_processor') and batch[0]._parallel_processor is not None:
        try:
            # Extract data for parallel feature processing
            positive_combinations = basic_batch['positive_combinations']
            current_indices = basic_batch['current_indices']
            
            # Process features in parallel
            parallel_processor = batch[0]._parallel_processor
            feature_vectors = parallel_processor.process_batch_parallel(
                positive_combinations, current_indices
            )
            
            # Add processed features to batch
            basic_batch['parallel_features'] = torch.from_numpy(feature_vectors)
            basic_batch['feature_processing_stats'] = parallel_processor.get_performance_stats()
            
        except Exception as e:
            print(f"⚠️ Parallel feature processing failed, using fallback: {e}")
            # Fallback to standard processing
            basic_batch['parallel_features'] = None
    
    return basic_batch

def create_cvae_data_loaders(df, feature_engineer, config):
    """
    Creates training and validation data loaders for CVAE with proper temporal splitting.
    
    Args:
        df: Historical lottery data (must be sorted by Date)
        feature_engineer: Fitted feature engineering object
        config: Configuration dictionary
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Ensure data is sorted by date for proper temporal splitting
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"Data sorted by date: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    else:
        print("⚠️  No Date column found, using row order (not recommended)")
    
    # Temporal split with gap to prevent leakage
    total_size = len(df)
    # Use 75% for training, 5% gap, 20% for validation
    train_end = int(total_size * 0.75)
    gap_size = max(1, int(total_size * 0.05))  # At least 1 sample gap
    val_start = train_end + gap_size
    
    # Create temporal splits
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[val_start:].reset_index(drop=True)
    
    print(f"Temporal split: {len(train_df)} train, {gap_size} gap, {len(val_df)} validation")
    if 'Date' in df.columns:
        print(f"Train period: {train_df['Date'].iloc[0]} to {train_df['Date'].iloc[-1]}")
        print(f"Validation period: {val_df['Date'].iloc[0]} to {val_df['Date'].iloc[-1]}")
        
        # Verify no temporal leakage
        if train_df['Date'].iloc[-1] >= val_df['Date'].iloc[0]:
            print("⚠️  Warning: Potential temporal leakage detected")
    
    # Ensure minimum validation size
    if len(val_df) < 10:
        print("⚠️  Very small validation set, consider using more data")
    
    # Create datasets
    train_dataset = CVAEDataset(train_df, feature_engineer, config, is_training=True)
    
    # Share negative pool between train and validation for consistency
    val_dataset = CVAEDataset(val_df, feature_engineer, config, 
                             negative_pool=train_dataset.negative_pool, 
                             is_training=False)
    
    # === PHASE 1 PERFORMANCE OPTIMIZATIONS ===
    
    # Calculate optimal settings based on hardware
    optimal_batch_size = _calculate_optimal_batch_size(config)
    optimal_workers = _get_optimal_num_workers() if config.get('num_workers') == 'auto' else config.get('num_workers', 2)
    smart_pin_memory = _get_smart_pin_memory() if config.get('pin_memory') == 'auto' else config.get('pin_memory', torch.cuda.is_available())
    
    # === PHASE 2 MEDIUM-TERM IMPROVEMENTS ===
    
    # Initialize Phase 2 optimizations if enabled
    parallel_processor = None
    memory_manager = None
    
    if PHASE2_AVAILABLE and config.get('enable_performance_optimizations', False):
        if config.get('enable_parallel_features', False):
            try:
                # Initialize parallel feature processor
                parallel_config = {
                    'feature_cache_size_gb': config.get('feature_cache_size_gb', 6.0),
                    'max_workers': config.get('feature_parallel_workers', 'auto'),
                    'parallel_threshold': config.get('feature_batch_threshold', 16),
                    'use_threading': config.get('use_feature_threading', True)
                }
                parallel_processor = ParallelFeatureProcessor(feature_engineer, parallel_config)
                print(f"✅ Phase 2 Parallel Feature Processor initialized")
            except Exception as e:
                print(f"⚠️ Parallel feature processor initialization failed: {e}")
        
        if config.get('enable_memory_pools', False):
            try:
                # Initialize memory pool manager
                memory_config = MemoryPoolConfig(
                    tensor_pool_gb=config.get('tensor_pool_size_gb', 4.0),
                    batch_cache_gb=config.get('batch_cache_size_gb', 8.0),
                    feature_cache_gb=config.get('feature_cache_size_gb', 6.0),
                    enable_compression=config.get('enable_cache_compression', True),
                    cleanup_threshold=config.get('memory_pressure_threshold', 0.85)
                )
                memory_manager = get_memory_manager(memory_config)
                memory_manager.optimize_for_batch_size(optimal_batch_size)
                print(f"✅ Phase 2 Memory Pool Manager initialized")
            except Exception as e:
                print(f"⚠️ Memory pool manager initialization failed: {e}")
    
    # Enhanced dynamic batch sizing for Phase 2
    if config.get('enable_dynamic_batching', False) and memory_manager:
        try:
            # Get current memory stats for dynamic batch sizing
            memory_stats = memory_manager.get_comprehensive_stats()
            memory_pressure = memory_stats['memory_usage_pct'] / 100.0
            
            if memory_pressure < 0.7:  # Low memory pressure
                scaling_factor = config.get('batch_size_scaling_factor', 3.5)
                max_dynamic_size = config.get('max_dynamic_batch_size', 64)
                enhanced_batch_size = min(int(optimal_batch_size * scaling_factor), max_dynamic_size)
                # Enforce minimum batch size of 8 for viable model complexity
                min_batch_size = 8
                optimal_batch_size = max(min_batch_size, enhanced_batch_size)
                print(f"🚀 Dynamic batch sizing: {enhanced_batch_size} → {optimal_batch_size} (memory pressure: {memory_pressure:.1%}, min={min_batch_size})")
        except Exception as e:
            print(f"⚠️ Dynamic batch sizing failed: {e}")
    
    # Display optimization info
    if config.get('enable_performance_optimizations', False):
        available_ram = psutil.virtual_memory().available / (1024**3)
        print(f"🚀 Phase 1+2 DataLoader Optimizations:")
        print(f"   • Workers: {optimal_workers} (CPU cores: {multiprocessing.cpu_count()})")
        print(f"   • Batch size: {optimal_batch_size}")
        print(f"   • Pin memory: {smart_pin_memory} (RAM: {available_ram:.1f}GB available)")
        print(f"   • Persistent workers: {config.get('persistent_workers', True)}")
        print(f"   • Prefetch factor: {config.get('prefetch_factor', 4)}")
        
        if parallel_processor:
            print(f"   • Parallel features: ✅ {parallel_processor.num_workers} workers")
        if memory_manager:
            print(f"   • Memory pools: ✅ {memory_manager.config.tensor_pool_gb}GB tensor pool")
    
    # Choose appropriate collate function based on Phase 2 availability
    collate_function = collate_cvae_batch
    if PHASE2_AVAILABLE and config.get('enable_parallel_features', False) and parallel_processor:
        # Use enhanced collate function for Phase 2
        collate_function = enhanced_collate_cvae_batch_phase2
        # Attach parallel processor to datasets for collate function access
        train_dataset._parallel_processor = parallel_processor
        val_dataset._parallel_processor = parallel_processor
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimal_batch_size,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=optimal_workers,
        pin_memory=smart_pin_memory,
        persistent_workers=config.get('persistent_workers', True) if optimal_workers > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 4) if optimal_workers > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        collate_fn=collate_function,
        num_workers=optimal_workers,
        pin_memory=smart_pin_memory,
        persistent_workers=config.get('persistent_workers', True) if optimal_workers > 0 else False,
        prefetch_factor=config.get('prefetch_factor', 4) if optimal_workers > 0 else 2
    )
    
    # Store Phase 2 components in DataLoader for later access
    if parallel_processor:
        train_loader._parallel_processor = parallel_processor
        val_loader._parallel_processor = parallel_processor
    if memory_manager:
        train_loader._memory_manager = memory_manager
        val_loader._memory_manager = memory_manager
    
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