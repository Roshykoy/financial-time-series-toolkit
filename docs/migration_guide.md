# MarkSix Project Migration Guide

## Overview

This guide helps you migrate from the old monolithic structure to the new modular architecture of the MarkSix Probabilistic Forecasting System. The migration is designed to be **gradual and safe** to avoid breaking existing functionality.

## What Changed

### **Old Structure Problems**
- **Monolithic files**: 900+ line training/inference pipelines
- **Tight coupling**: Global CONFIG imported everywhere
- **Mixed concerns**: Training, debugging, visualization in single files
- **Repeated code**: Model loading, data processing patterns duplicated

### **New Structure Benefits**
- **Modular architecture**: Clear separation of concerns
- **Dependency injection**: Configurable components
- **Structured logging**: Better debugging and monitoring
- **Configuration management**: Environment-specific settings
- **Testability**: Isolated components easier to test

## Directory Structure Changes

```
Old Structure:                  New Structure:
src/                           src/
├── config.py                  ├── core/                    # Business logic
├── cvae_model.py             │   ├── models/              # Model definitions
├── training_pipeline.py      │   ├── data/                # Data processing
├── inference_pipeline.py     │   ├── training/            # Training components
├── feature_engineering.py    │   └── inference/           # Inference components
├── meta_learner.py           ├── infrastructure/          # Infrastructure
├── graph_encoder.py          │   ├── config/              # Configuration
├── temporal_context.py       │   ├── logging/             # Logging system
└── ...                       │   ├── storage/             # Persistence
                              │   └── monitoring/          # Monitoring
                              ├── application/             # Application layer
                              │   ├── services/            # Business services
                              │   └── cli/                 # CLI interface
                              └── utils/                   # Shared utilities

config/                        config/                      # Configuration files
├── default.yml               ├── environments/            # Environment configs
├── model_presets/            └── model_presets/           # Model presets
```

## Migration Steps

### Step 1: Automated Migration

Run the migration script to set up the new structure:

```bash
# Dry run first (recommended)
python scripts/migrate.py --dry-run

# Run actual migration
python scripts/migrate.py
```

This will:
- ✅ Create backup of existing code
- ✅ Set up new directory structure  
- ✅ Create compatibility layer
- ✅ Update main.py imports

### Step 2: Test Compatibility

After migration, test that everything still works:

```bash
# Test the system
python main.py

# Run existing tests
python test_hyperparameter_optimization.py
python test_model_debug.py
```

### Step 3: Gradual Code Updates

Update your code gradually to use the new architecture:

#### **Configuration Updates**

**Old way:**
```python
from src.config import CONFIG
batch_size = CONFIG['batch_size']
```

**New way:**
```python
from src.infrastructure.config import get_config
config = get_config()
batch_size = config['training'].batch_size
```

#### **Logging Updates**

**Old way:**
```python
print(f"Training epoch {epoch}")
```

**New way:**
```python
from src.infrastructure.logging import get_logger
logger = get_logger(__name__)
logger.info(f"Training epoch {epoch}")
```

#### **Model Creation Updates**

**Old way:**
```python
from src.cvae_model import ConditionalVAE
model = ConditionalVAE(CONFIG)
```

**New way:**
```python
from src.core.models import ModelFactory
from src.infrastructure.config import get_config

config = get_config()
model = ModelFactory.create('cvae', config['model'])
```

### Step 4: Move Model Implementations

Gradually move your model files to the new structure:

1. **Create new model file** in `src/core/models/`
2. **Inherit from BaseModel** for consistency
3. **Register with ModelFactory** using `@register_model` decorator
4. **Update imports** in files that use the model

Example:
```python
# src/core/models/cvae.py
from src.core.models.base import BaseModel, register_model

@register_model('cvae')
class ConditionalVAE(BaseModel):
    def __init__(self, config, device="cpu"):
        super().__init__(config, device)
        # Your existing implementation
```

### Step 5: Create Services

Convert your pipeline functions to service classes:

**Old way:**
```python
def run_training():
    # 200+ lines of training logic
    pass
```

**New way:**
```python
# src/application/services/training_service.py
from src.application.services.base import BaseService

class TrainingService(BaseService):
    def train_model(self, training_config=None):
        # Focused training logic
        # Returns ServiceResult
        pass
```

## Backward Compatibility

The migration maintains **full backward compatibility** during transition:

### **Configuration Compatibility**
- Old `from src.config import CONFIG` still works
- Deprecation warnings guide you to new system
- All existing parameter names preserved

### **Import Compatibility**
- Existing imports continue working
- Gradual migration allows testing each change
- No breaking changes to public interfaces

### **Functionality Compatibility**
- All CLI options work unchanged
- Model files and outputs unchanged
- Training/inference behavior identical

## Configuration Management

### **Environment-Specific Settings**

Create environment configs for different use cases:

```yaml
# config/environments/development.yml
training:
  epochs: 5
  batch_size: 16
system:
  device: "cpu"  # Force CPU for development

# config/environments/production.yml  
training:
  epochs: 25
  batch_size: 8
system:
  device: "auto"  # Use GPU if available
```

### **Model Presets**

Use model presets for different scenarios:

```python
from src.infrastructure.config import get_config_manager

# Load fast training preset
config_manager = get_config_manager()
config_manager.config_path = Path("config/model_presets/fast_training.yml")
config = config_manager.load_config()
```

## Testing Strategy

### **Unit Testing**
```python
# tests/unit/test_models.py
from src.core.models.base import ModelFactory
from src.infrastructure.config import ModelConfig

def test_model_creation():
    config = ModelConfig()
    model = ModelFactory.create('cvae', config)
    assert model is not None
```

### **Integration Testing**
```python
# tests/integration/test_training_pipeline.py
from src.application.services.training_service import TrainingService

def test_full_training():
    service = TrainingService(config_manager)
    result = service.train_model()
    assert result.success
```

## Troubleshooting

### **Import Errors**
If you get import errors:
1. Check that `__init__.py` files exist in new directories
2. Verify Python path includes project root
3. Use absolute imports: `from src.infrastructure.config import ...`

### **Configuration Errors**
If configuration isn't loading:
1. Check `config/default.yml` exists and is valid YAML
2. Verify environment variable `MARKSIX_ENV` if using environments
3. Check file permissions on config directory

### **Compatibility Issues**
If old code stops working:
1. Check the compatibility layer in `src/config_legacy.py`
2. Look for deprecation warnings in console output
3. Restore from backup: files are in `backup_old_structure/`

## Migration Checklist

- [ ] **Backup created** - Original files safely stored
- [ ] **New structure created** - All directories and base files
- [ ] **Tests pass** - Existing functionality works
- [ ] **Configuration migrated** - Using new config system
- [ ] **Logging added** - Better debugging and monitoring
- [ ] **Services created** - Business logic in service classes
- [ ] **Models refactored** - Using base classes and factory
- [ ] **Documentation updated** - New structure documented

## Rollback Procedure

If you need to rollback:

```bash
# Restore original files from backup
cp -r backup_old_structure/* .

# Remove new directories if desired
rm -rf src/core src/infrastructure src/application
rm -rf config/environments config/model_presets
```

## Benefits After Migration

### **Improved Maintainability**
- **Smaller files**: Each file has single responsibility
- **Clear dependencies**: Explicit configuration injection
- **Better testing**: Isolated components easier to test

### **Enhanced Flexibility**
- **Environment configs**: Different settings for dev/prod
- **Model presets**: Quick switching between configurations
- **Plugin architecture**: Easy to add new models/scorers

### **Better Development Experience**
- **Structured logging**: Better debugging information
- **Error handling**: Consistent error patterns
- **IDE support**: Better code completion and navigation

### **Future-Proofing**
- **Extensible architecture**: Easy to add new features
- **Clean interfaces**: Well-defined component boundaries
- **Configuration management**: Easy parameter tuning

## Getting Help

If you encounter issues during migration:

1. **Check logs**: Look in `outputs/marksix.log` for detailed error information
2. **Run diagnostics**: Use `python main.py` → Option 5 (System Diagnostics)
3. **Review backup**: Original files are preserved in `backup_old_structure/`
4. **Test incrementally**: Migrate one component at a time

The migration is designed to be safe and reversible. Take your time and test thoroughly at each step.