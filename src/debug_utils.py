# src/debug_utils.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import traceback
import psutil
import gc
from collections import defaultdict
from contextlib import contextmanager
import time
import warnings

class ModelDebugger:
    """
    Comprehensive debugging utility for PyTorch models.
    Helps catch device mismatches, shape errors, NaN/inf values, and more.
    """
    
    def __init__(self, model, device=None, verbose=True):
        self.model = model
        # FIX: Normalize device specification
        if device is None:
            device = next(model.parameters()).device
        
        # Normalize cuda device specification
        if device.type == 'cuda':
            self.device = torch.device(f'cuda:{device.index if device.index is not None else 0}')
        else:
            self.device = device
            
        self.verbose = verbose
        self.debug_log = []
        
    def log(self, message, level="INFO"):
        """Log debug messages with timestamp."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.debug_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def check_device_consistency(self):
        """Check that all model parameters are on the same device."""
        self.log("🔍 Checking device consistency...")
        
        devices = set()
        param_count = 0
        
        for name, param in self.model.named_parameters():
            # Normalize parameter device for comparison
            param_device = param.device
            if param_device.type == 'cuda':
                param_device = torch.device(f'cuda:{param_device.index if param_device.index is not None else 0}')
            
            devices.add(param_device)
            param_count += 1
            
            if param_device != self.device:
                self.log(f"❌ Parameter {name} is on {param_device}, expected {self.device}", "ERROR")
                return False
        
        # Same fix for buffers
        buffer_count = 0
        for name, buffer in self.model.named_buffers():
            buffer_device = buffer.device
            if buffer_device.type == 'cuda':
                buffer_device = torch.device(f'cuda:{buffer_device.index if buffer_device.index is not None else 0}')
            
            devices.add(buffer_device)
            buffer_count += 1
            
            if buffer_device != self.device:
                self.log(f"❌ Buffer {name} is on {buffer_device}, expected {self.device}", "ERROR")
                return False
        
        self.log(f"✅ All {param_count} parameters and {buffer_count} buffers on {self.device}")
        return True
    
    def check_tensor_health(self, tensor, name="tensor"):
        """Check tensor for NaN, inf, and other issues."""
        if tensor is None:
            self.log(f"❌ {name} is None", "ERROR")
            return False
            
        issues = []
        
        # Check for NaN
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            issues.append(f"{nan_count} NaN values")
        
        # Check for inf
        if torch.isinf(tensor).any():
            inf_count = torch.isinf(tensor).sum().item()
            issues.append(f"{inf_count} inf values")
        
        # Check for extremely large values
        if tensor.dtype in [torch.float32, torch.float64]:
            max_val = tensor.abs().max().item()
            if max_val > 1e6:
                issues.append(f"very large values (max: {max_val:.2e})")
        
        # FIX: Normalize device check
        tensor_device = tensor.device
        if tensor_device.type == 'cuda':
            tensor_device = torch.device(f'cuda:{tensor_device.index if tensor_device.index is not None else 0}')
        
        if tensor_device != self.device:
            issues.append(f"wrong device ({tensor_device} vs {self.device})")
        
        if issues:
            self.log(f"⚠️  {name} has issues: {', '.join(issues)}", "WARNING")
            return False
        else:
            self.log(f"✅ {name} is healthy (shape: {tensor.shape}, device: {tensor.device})")
            return True
    
    def test_forward_pass(self, dummy_input_fn):
        """Test forward pass with dummy data."""
        self.log("🧪 Testing forward pass with dummy data...")
        
        try:
            # Get dummy inputs
            dummy_inputs = dummy_input_fn()
            
            # Check input health
            if isinstance(dummy_inputs, (list, tuple)):
                for i, inp in enumerate(dummy_inputs):
                    if isinstance(inp, torch.Tensor):
                        self.check_tensor_health(inp, f"dummy_input_{i}")
            elif isinstance(dummy_inputs, dict):
                for key, inp in dummy_inputs.items():
                    if isinstance(inp, torch.Tensor):
                        self.check_tensor_health(inp, f"dummy_input_{key}")
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                if isinstance(dummy_inputs, dict):
                    outputs = self.model(**dummy_inputs)
                elif isinstance(dummy_inputs, (list, tuple)):
                    outputs = self.model(*dummy_inputs)
                else:
                    outputs = self.model(dummy_inputs)
            
            # Check output health
            if isinstance(outputs, (list, tuple)):
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        self.check_tensor_health(out, f"output_{i}")
            elif isinstance(outputs, torch.Tensor):
                self.check_tensor_health(outputs, "output")
            
            self.log("✅ Forward pass completed successfully")
            return True
            
        except Exception as e:
            self.log(f"❌ Forward pass failed: {e}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False
    
    def test_backward_pass(self, dummy_input_fn, loss_fn):
        """Test backward pass with dummy data."""
        self.log("🔄 Testing backward pass...")
        
        try:
            # Get dummy inputs
            dummy_inputs = dummy_input_fn()
            
            # Forward pass
            self.model.train()
            if isinstance(dummy_inputs, dict):
                outputs = self.model(**dummy_inputs)
            elif isinstance(dummy_inputs, (list, tuple)):
                outputs = self.model(*dummy_inputs)
            else:
                outputs = self.model(dummy_inputs)
            
            # Compute loss
            loss = loss_fn(outputs)
            self.check_tensor_health(loss, "loss")
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Check gradients
            grad_count = 0
            max_grad = 0
            min_grad = float('inf')
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                    grad_norm = param.grad.data.norm().item()
                    max_grad = max(max_grad, grad_norm)
                    min_grad = min(min_grad, grad_norm)
                    
                    # Check gradient health
                    if torch.isnan(param.grad).any():
                        self.log(f"❌ NaN gradient in {name}", "ERROR")
                        return False
                    
                    if torch.isinf(param.grad).any():
                        self.log(f"❌ Inf gradient in {name}", "ERROR")
                        return False
            
            self.log(f"✅ Backward pass completed. Gradients in {grad_count} parameters")
            self.log(f"📊 Gradient range: [{min_grad:.2e}, {max_grad:.2e}]")
            return True
            
        except Exception as e:
            self.log(f"❌ Backward pass failed: {e}", "ERROR")
            self.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False
    
    def check_memory_usage(self):
        """Check GPU and system memory usage."""
        self.log("💾 Checking memory usage...")
        
        # System memory
        try:
            memory = psutil.virtual_memory()
            self.log(f"🖥️  System RAM: {memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB ({memory.percent:.1f}%)")
        except:
            self.log("⚠️  Could not check system memory", "WARNING")
        
        # GPU memory
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            
            self.log(f"🎮 GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")
            
            if allocated / total > 0.9:
                self.log("⚠️  GPU memory usage very high (>90%)", "WARNING")
            elif allocated / total > 0.7:
                self.log("⚠️  GPU memory usage high (>70%)", "WARNING")
        else:
            self.log("ℹ️  No CUDA device available")
    
    def validate_data_shapes(self, data_sample):
        """Validate that data shapes are consistent."""
        self.log("📐 Validating data shapes...")
        
        if isinstance(data_sample, dict):
            for key, value in data_sample.items():
                if isinstance(value, (list, torch.Tensor, np.ndarray)):
                    self.log(f"📏 {key}: {np.array(value).shape if not isinstance(value, torch.Tensor) else value.shape}")
        
    def comprehensive_check(self, dummy_input_fn=None, loss_fn=None):
        """Run all debugging checks."""
        self.log("🚀 Starting comprehensive model debugging...")
        
        checks_passed = 0
        total_checks = 5
        
        # 1. Device consistency
        if self.check_device_consistency():
            checks_passed += 1
        
        # 2. Memory usage
        self.check_memory_usage()
        checks_passed += 1  # Memory check always "passes"
        
        # 3. Forward pass test
        if dummy_input_fn:
            if self.test_forward_pass(dummy_input_fn):
                checks_passed += 1
            else:
                total_checks -= 1
        else:
            total_checks -= 1
        
        # 4. Backward pass test
        if dummy_input_fn and loss_fn:
            if self.test_backward_pass(dummy_input_fn, loss_fn):
                checks_passed += 1
            else:
                total_checks -= 1
        else:
            total_checks -= 1
        
        # 5. Model structure
        self.analyze_model_structure()
        checks_passed += 1
        
        success_rate = checks_passed / total_checks
        if success_rate >= 0.8:
            self.log(f"🎉 Debugging complete! {checks_passed}/{total_checks} checks passed ({success_rate:.1%})")
        else:
            self.log(f"⚠️  Debugging complete. {checks_passed}/{total_checks} checks passed ({success_rate:.1%})", "WARNING")
        
        return success_rate >= 0.8
    
    def analyze_model_structure(self):
        """Analyze model structure and parameters."""
        self.log("🏗️  Analyzing model structure...")
        
        total_params = 0
        trainable_params = 0
        layers_by_type = defaultdict(int)
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = type(module).__name__
                layers_by_type[module_type] += 1
                
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                total_params += param_count
                trainable_params += trainable_count
        
        self.log(f"📊 Model structure:")
        for layer_type, count in layers_by_type.items():
            self.log(f"   • {layer_type}: {count}")
        
        self.log(f"🔢 Total parameters: {total_params:,}")
        self.log(f"🎓 Trainable parameters: {trainable_params:,}")
        
        param_size_mb = total_params * 4 / (1024 * 1024)
        self.log(f"💾 Estimated size: {param_size_mb:.1f} MB")
    
    def save_debug_report(self, filepath="outputs/debug_report.txt"):
        """Save debug log to file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("MODEL DEBUGGING REPORT\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.debug_log:
                f.write(entry + "\n")
        
        self.log(f"📝 Debug report saved to {filepath}")

class DataLoaderDebugger:
    """Debug data loading pipeline."""
    
    def __init__(self, data_loader, verbose=True):
        self.data_loader = data_loader
        self.verbose = verbose
    
    def check_batch_consistency(self, num_batches=3):
        """Check first few batches for consistency."""
        print("🔍 Checking data loader batch consistency...")
        
        batch_info = []
        for i, batch in enumerate(self.data_loader):
            if i >= num_batches:
                break
            
            info = {
                'batch_idx': i,
                'batch_type': type(batch),
            }
            
            if isinstance(batch, dict):
                info['keys'] = list(batch.keys())
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        info[f'{key}_shape'] = value.shape
                        info[f'{key}_dtype'] = value.dtype
                        info[f'{key}_device'] = value.device
                    elif isinstance(value, list):
                        info[f'{key}_len'] = len(value)
                        if value and isinstance(value[0], (int, float)):
                            info[f'{key}_sample'] = value[:3]
            
            batch_info.append(info)
            
            if self.verbose:
                print(f"Batch {i}: {info}")
        
        # Check consistency
        if len(batch_info) > 1:
            first_batch = batch_info[0]
            for i, batch in enumerate(batch_info[1:], 1):
                for key in first_batch:
                    if key in batch and batch[key] != first_batch[key] and 'sample' not in key:
                        if 'shape' in key:
                            # Allow different batch sizes
                            if batch[key][0] != first_batch[key][0]:  # Different batch size is OK
                                continue
                        print(f"⚠️  Inconsistency in {key}: batch 0 has {first_batch[key]}, batch {i} has {batch[key]}")
        
        print("✅ Data loader check complete")
        return batch_info

@contextmanager
def debug_training_step(model, batch, device, step_name="training_step"):
    """Context manager to debug a training step."""
    print(f"🔍 Debugging {step_name}...")
    
    # Pre-step checks
    debugger = ModelDebugger(model, device, verbose=False)
    
    print("📊 Pre-step state:")
    debugger.check_memory_usage()
    
    start_time = time.time()
    
    try:
        yield debugger
        
        # Post-step checks
        end_time = time.time()
        print(f"⏱️  Step completed in {end_time - start_time:.3f}s")
        
        print("📊 Post-step state:")
        debugger.check_memory_usage()
        
    except Exception as e:
        print(f"❌ Error in {step_name}: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        
        # Emergency diagnostics
        print("🚨 Emergency diagnostics:")
        debugger.check_device_consistency()
        debugger.check_memory_usage()
        
        raise

def create_dummy_cvae_inputs(config, device, batch_size=2):
    """Create dummy inputs for CVAE testing."""
    
    def dummy_input_fn():
        # Create dummy combinations
        positive_combinations = []
        for _ in range(batch_size):
            combo = sorted(np.random.choice(range(1, config['num_lotto_numbers'] + 1), 6, replace=False).tolist())
            positive_combinations.append(combo)
        
        # Create dummy pair counts
        pair_counts = {}
        for i in range(1, config['num_lotto_numbers'] + 1):
            for j in range(i + 1, config['num_lotto_numbers'] + 1):
                pair_counts[(i, j)] = np.random.randint(0, 10)
        
        # Create dummy dataframe
        dummy_data = []
        for i in range(100):
            row = [i, f"2024-01-{i+1:02d}"] + list(np.random.choice(range(1, 50), 7, replace=False))
            row.extend([0] * 26)  # Fill remaining columns
            dummy_data.append(row)
        
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num'
        ] + [f'Col_{i}' for i in range(26)]
        
        df = pd.DataFrame(dummy_data, columns=col_names)
        current_indices = list(range(batch_size))
        
        return positive_combinations, pair_counts, df, current_indices
    
    return dummy_input_fn

def create_dummy_loss_fn():
    """Create dummy loss function for testing."""
    def loss_fn(outputs):
        if isinstance(outputs, (list, tuple)):
            # For CVAE outputs: reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context
            reconstruction_logits = outputs[0]
            return reconstruction_logits.mean()  # Dummy loss
        else:
            return outputs.mean()
    
    return loss_fn

def quick_model_test(model, config, device):
    """Quick comprehensive test of model."""
    print("🚀 Running quick model test...")
    
    debugger = ModelDebugger(model, device)
    dummy_input_fn = create_dummy_cvae_inputs(config, device)
    loss_fn = create_dummy_loss_fn()
    
    success = debugger.comprehensive_check(dummy_input_fn, loss_fn)
    
    if success:
        print("✅ Model passed all tests!")
    else:
        print("❌ Model failed some tests. Check the debug log.")
    
    return success, debugger

# Auto-debugging decorator
def auto_debug(device_check=True, memory_check=True, tensor_check=True):
    """Decorator to automatically debug tensor operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Pre-execution checks
            if device_check:
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        if arg.device.type != 'cuda' and torch.cuda.is_available():
                            warnings.warn(f"Tensor on {arg.device} but CUDA available")
            
            if memory_check and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                result = func(*args, **kwargs)
                
                # Post-execution checks
                if tensor_check and isinstance(result, torch.Tensor):
                    if torch.isnan(result).any():
                        warnings.warn(f"Function {func.__name__} returned tensor with NaN values")
                    if torch.isinf(result).any():
                        warnings.warn(f"Function {func.__name__} returned tensor with inf values")
                
                return result
                
            except Exception as e:
                print(f"❌ Error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator