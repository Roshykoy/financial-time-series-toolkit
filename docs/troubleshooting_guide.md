# MarkSix Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with the MarkSix Probabilistic Forecasting System. Issues are organized by category with step-by-step solutions.

---

## 🚨 Critical Issues (System Won't Start)

### 1. Import Errors on Startup

**Symptoms:**
```
ImportError: cannot import name 'configure_logging' from 'src.infrastructure.config'
ModuleNotFoundError: No module named 'torch'
```

**Diagnosis:**
- Environment not activated
- Missing dependencies
- Incorrect Python path

**Solutions:**

#### Option A: Environment Issues
```bash
# Check current environment
conda info --envs

# Activate MarkSix environment
conda activate marksix_ai

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

#### Option B: Dependency Issues
```bash
# Reinstall environment
conda env remove -n marksix_ai
conda env create -f environment.yml
conda activate marksix_ai

# Or update existing environment
conda env update -f environment.yml
```

#### Option C: Import Path Issues
```bash
# Run from project root directory
cd /path/to/MarkSix-Probabilistic-Forecasting
python main.py

# Or use the improved version
python main_improved.py
```

### 2. Configuration Loading Errors

**Symptoms:**
```
ConfigurationError: CONFIG not found
NameError: name 'CONFIG' is not defined
```

**Solutions:**

#### Quick Fix:
```bash
# Use the improved main file with better error handling
python main_improved.py
```

#### Manual Fix:
```python
# Check if config files exist
ls src/config*.py

# If missing, create minimal config
cat > src/config_minimal.py << 'EOF'
import torch

CONFIG = {
    "data_path": "data/raw/Mark_Six.csv",
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "models/conservative_cvae_model.pth",
    "meta_learner_save_path": "models/conservative_meta_learner.pth",
    "feature_engineer_path": "models/conservative_feature_engineer.pkl"
}
EOF
```

---

## ⚠️ Runtime Errors

### 3. CUDA Out of Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Diagnosis:**
- Batch size too large for GPU
- Model too complex for available VRAM
- Memory leak in training loop

**Solutions:**

#### Immediate Fixes:
```python
# Reduce batch size in config
CONFIG['batch_size'] = 4  # or even 2

# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Use CPU fallback
CONFIG['device'] = 'cpu'
```

#### Permanent Solutions:
1. **Edit configuration file:**
```python
# In src/config.py, reduce memory usage:
CONFIG.update({
    'batch_size': 4,
    'latent_dim': 32,          # Reduced from 64
    'graph_hidden_dim': 32,    # Reduced from 64
    'temporal_hidden_dim': 32, # Reduced from 64
})
```

2. **Use memory-efficient training:**
```bash
# Run with memory optimization
python main_improved.py  # Has automatic memory management
```

### 4. Training Crashes with NaN/Inf Values

**Symptoms:**
```
RuntimeError: Function 'LogSoftmaxBackward' returned nan values
Loss contains NaN or Inf values
```

**Diagnosis:**
- Numerical instability in loss computation
- Learning rate too high
- Gradient explosion

**Solutions:**

#### Quick Fixes:
```python
# Reduce learning rate
CONFIG['learning_rate'] = 1e-5  # Much smaller

# Enable gradient clipping
CONFIG['gradient_clip_norm'] = 0.1  # Very conservative

# Add numerical stability
CONFIG['numerical_stability_eps'] = 1e-6
```

#### Use Improved Engine:
```python
# Replace cvae_engine.py imports with:
from src.cvae_engine_improved import ImprovedCVAELossComputer, SafeCVAETrainer
```

### 5. Data Loading Errors

**Symptoms:**
```
FileNotFoundError: data/raw/Mark_Six.csv not found
pd.errors.EmptyDataError: No columns to parse
```

**Solutions:**

#### Check Data File:
```bash
# Verify data file exists
ls -la data/raw/Mark_Six.csv

# Check file format
head -5 data/raw/Mark_Six.csv

# Check file permissions
ls -la data/raw/
```

#### Fix Data Issues:
```bash
# Create missing directories
mkdir -p data/raw data/processed

# Download sample data (if available)
# Or create dummy data for testing
python -c "
import pandas as pd
import numpy as np

# Create dummy lottery data
np.random.seed(42)
dummy_data = []
for i in range(100):
    numbers = sorted(np.random.choice(49, 6, replace=False) + 1)
    extra = np.random.randint(1, 50)
    dummy_data.append([i+1, f'2024-01-{i%30+1:02d}'] + numbers + [extra])

df = pd.DataFrame(dummy_data, columns=['Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3', 'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num'])
df.to_csv('data/raw/Mark_Six.csv', index=False)
print('Dummy data created')
"
```

---

## 🔧 Performance Issues

### 6. Very Slow Training

**Symptoms:**
- Training takes hours on CPU
- GPU utilization low
- System becomes unresponsive

**Diagnosis & Solutions:**

#### Check GPU Usage:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check PyTorch GPU access
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

#### Optimize Configuration:
```python
# Use fast training preset
CONFIG.update({
    'epochs': 5,
    'batch_size': 16,
    'latent_dim': 32,
    'num_gat_layers': 1,
    'temporal_sequence_length': 5
})
```

#### Performance Tips:
1. **Use compiled mode (PyTorch 2.0+):**
```python
model = torch.compile(model)  # Add this after model creation
```

2. **Enable optimizations:**
```python
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # For newer GPUs
```

### 7. Memory Leaks During Training

**Symptoms:**
- Memory usage constantly increasing
- System becomes slower over time
- Eventually crashes with OOM

**Solutions:**

#### Enable Memory Monitoring:
```python
# Add to training loop
if epoch % 5 == 0:
    import psutil
    memory_percent = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Memory: {memory_percent:.1f}% RAM, {gpu_memory:.1f}GB GPU")
```

#### Fix Memory Leaks:
```python
# Clear cache regularly
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()

# Use context managers for temporary tensors
with torch.no_grad():
    # Evaluation code here
    pass

# Delete large variables explicitly
del large_tensor
torch.cuda.empty_cache()
```

---

## 🐛 Model & Algorithm Issues

### 8. Poor Model Performance

**Symptoms:**
- Win rate below 50%
- Generated numbers are not diverse
- Model seems to overfit quickly

**Diagnosis & Solutions:**

#### Check Training Data:
```python
# Verify data quality
import pandas as pd
df = pd.read_csv('data/raw/Mark_Six.csv')
print(f"Data shape: {df.shape}")
print(f"Unique draws: {df['Draw'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Check for duplicates or anomalies
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
```

#### Hyperparameter Optimization:
```python
# Run hyperparameter optimization
python main.py  # Choose option 4

# Or use grid search manually
from src.hyperparameter_optimizer import HyperparameterOptimizer
optimizer = HyperparameterOptimizer(config=CONFIG)
best_params = optimizer.random_search(n_trials=20)
```

#### Model Architecture Issues:
```python
# Check model output shapes
model = create_model()
dummy_input = create_dummy_input()
output = model(dummy_input)
print(f"Model output shapes: {[v.shape for v in output.values()]}")

# Verify gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

### 9. Inference Generates Invalid Combinations

**Symptoms:**
- Numbers outside 1-49 range
- Duplicate numbers in combinations
- Empty or malformed output

**Solutions:**

#### Add Output Validation:
```python
def validate_combination(combo):
    """Validate generated combination."""
    if len(combo) != 6:
        raise ValueError(f"Expected 6 numbers, got {len(combo)}")
    
    if not all(1 <= n <= 49 for n in combo):
        raise ValueError(f"Numbers out of range: {combo}")
    
    if len(set(combo)) != 6:
        raise ValueError(f"Duplicate numbers: {combo}")
    
    return sorted(combo)

# Use in inference
generated_combo = model.generate()
validated_combo = validate_combination(generated_combo)
```

#### Fix Generation Logic:
```python
# Add constraints to generation
def constrained_generation(model, num_samples=10):
    valid_combos = []
    max_attempts = num_samples * 10  # Prevent infinite loops
    
    for attempt in range(max_attempts):
        try:
            combo = model.generate_one()
            validated = validate_combination(combo)
            if validated not in valid_combos:  # Ensure uniqueness
                valid_combos.append(validated)
                
            if len(valid_combos) >= num_samples:
                break
        except ValueError:
            continue  # Skip invalid combinations
    
    return valid_combos
```

---

## 🔌 Environment & System Issues

### 10. Package Version Conflicts

**Symptoms:**
```
ImportError: cannot import name 'ABC' from 'collections'
AttributeError: module 'torch' has no attribute 'compile'
```

**Solutions:**

#### Check Package Versions:
```bash
# Check Python version
python --version  # Should be 3.10+

# Check key packages
pip list | grep -E "(torch|numpy|pandas|scikit-learn)"

# Check for conflicts
pip check
```

#### Fix Version Issues:
```bash
# Update to compatible versions
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Fix numpy issues
pip install "numpy>=1.24,<2.0"

# Resolve conflicts
pip install --upgrade pip setuptools wheel
pip install --force-reinstall -r requirements/dev.txt
```

### 11. File Permission Issues

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
OSError: [Errno 28] No space left on device
```

**Solutions:**

#### Fix Permissions:
```bash
# Check permissions
ls -la models/ outputs/ data/

# Fix directory permissions
chmod -R 755 models outputs data
mkdir -p models outputs data/processed

# Check disk space
df -h
```

#### Alternative Locations:
```python
# Use temporary directory if needed
import tempfile
import os

temp_dir = tempfile.mkdtemp()
CONFIG['model_save_path'] = os.path.join(temp_dir, 'model.pth')
CONFIG['output_dir'] = temp_dir
```

---

## 🔍 Debugging Tools

### 12. Enable Detailed Debugging

#### System Diagnostics:
```python
# Run comprehensive system check
python main_improved.py  # Choose option 5 for diagnostics

# Or run manually
from src.utils.error_handling import log_system_info
import logging
logger = logging.getLogger(__name__)
log_system_info(logger)
```

#### Model Debugging:
```python
# Test model architecture
python test_model_debug.py

# Check training pipeline
python test_hyperparameter_optimization.py
```

#### Enable Verbose Logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use structured logging
from src.infrastructure.logging import configure_logging
configure_logging(log_level="DEBUG", log_file="debug.log")
```

### 13. Performance Profiling

#### Memory Profiling:
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler main.py
```

#### GPU Profiling:
```bash
# Monitor GPU during training
nvidia-smi dmon -s pucvmet -d 1

# Or use PyTorch profiler
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your training code here
    pass

print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```

---

## 📞 Getting Help

### When to Seek Help

Contact support if you encounter:
1. **Critical bugs** that prevent system startup
2. **Data loss** or corruption issues  
3. **Security concerns** or suspicious behavior
4. **Performance degradation** that can't be resolved

### Information to Provide

When reporting issues, include:

1. **System Information:**
```bash
python --version
pip list | grep torch
nvidia-smi  # If using GPU
uname -a    # Linux/Mac
```

2. **Error Details:**
- Full error traceback
- Steps to reproduce
- Configuration used
- Data characteristics

3. **Log Files:**
- `outputs/marksix.log`
- `outputs/debug.log`
- Console output

### Quick Health Check

Run this comprehensive health check:

```python
#!/usr/bin/env python3
"""MarkSix System Health Check"""

def health_check():
    print("🔍 MarkSix System Health Check")
    print("=" * 50)
    
    # 1. Python environment
    import sys
    print(f"✓ Python {sys.version}")
    
    # 2. Core packages
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False
    
    # 3. GPU check
    if torch.cuda.is_available():
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠ No GPU available (CPU mode)")
    
    # 4. File system
    import os
    required_dirs = ['data/raw', 'models', 'outputs']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"⚠ Missing: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created: {dir_path}")
    
    # 5. Data file
    data_file = 'data/raw/Mark_Six.csv'
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / 1e6
        print(f"✓ Data file: {data_file} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Missing data file: {data_file}")
        return False
    
    # 6. Memory check
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / 1e9:.1f} GB")
        if memory.available < 2e9:  # Less than 2GB available
            print("⚠ Low available memory")
    except ImportError:
        print("⚠ Cannot check memory (install psutil)")
    
    print("=" * 50)
    print("✅ Health check completed!")
    return True

if __name__ == "__main__":
    health_check()
```

Save this as `health_check.py` and run it to quickly diagnose issues.

---

## 🚀 Quick Recovery Commands

### Emergency Reset:
```bash
# Reset to clean state
rm -rf models/* outputs/*
conda activate marksix_ai
python main_improved.py
```

### Memory Emergency:
```bash
# Clear all caches
python -c "
import torch
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print('Memory cleared')
"
```

### Configuration Reset:
```bash
# Backup current config
cp src/config.py src/config_backup.py

# Use minimal config
python -c "
import torch
config = '''
CONFIG = {
    \"data_path\": \"data/raw/Mark_Six.csv\",
    \"epochs\": 5,
    \"batch_size\": 4,
    \"learning_rate\": 1e-4,
    \"device\": \"cpu\",
    \"model_save_path\": \"models/safe_model.pth\"
}
'''
with open('src/config_safe.py', 'w') as f:
    f.write(config)
print('Safe config created')
"
```

This troubleshooting guide should help you resolve most common issues. For persistent problems, consider using the `main_improved.py` version which includes comprehensive error handling and recovery mechanisms.