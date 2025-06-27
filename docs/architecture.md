# MarkSix System Architecture

## Overview

The MarkSix Probabilistic Forecasting System uses a **layered architecture** with clear separation of concerns, dependency injection, and modular design. The system combines deep learning techniques (CVAE, Graph Neural Networks, Meta-Learning) with robust software engineering practices.

## Architecture Principles

### **Separation of Concerns**
- **Core**: Business logic and model implementations
- **Infrastructure**: Configuration, logging, storage, monitoring  
- **Application**: Services and user interfaces
- **Utils**: Shared utilities and helpers

### **Dependency Injection**
- Configuration passed as constructor parameters
- No global state or singleton dependencies
- Easy to test and mock components
- Runtime configuration switching

### **Interface-Based Design**
- Abstract base classes define contracts
- Concrete implementations focus on specific functionality
- Easy to swap implementations (e.g., different model types)
- Plugin architecture for extensibility

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │  CLI Interface  │  │    Services     │  │   Web API     │
│  │                 │  │                 │  │   (Future)    │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │     Models      │  │      Data       │  │   Training    │
│  │                 │  │                 │  │               │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │   Inference     │  │   Evaluation    │                   │
│  │                 │  │                 │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │  Configuration  │  │     Logging     │  │    Storage    │
│  │                 │  │                 │  │               │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │   Monitoring    │  │   Device Mgmt   │                   │
│  │                 │  │                 │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### **Models Module** (`src/core/models/`)

**Base Classes:**
- `BaseModel`: Abstract base for all models
- `BaseEncoder`: Abstract base for encoder models
- `BaseDecoder`: Abstract base for decoder models
- `BaseScorer`: Abstract base for scoring components

**Model Factory:**
```python
from src.core.models import ModelFactory, register_model

@register_model('cvae')
class ConditionalVAE(BaseModel):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        super().__init__(config, device)
        # Implementation...

# Usage
model = ModelFactory.create('cvae', config.model)
```

**Key Features:**
- Dependency injection through constructor
- Standardized checkpoint saving/loading
- Parameter counting and device management
- Consistent error handling and logging

### **Data Module** (`src/core/data/`)

**Components:**
- `DataLoader`: Abstract interface for data loading
- `FeatureProcessor`: Feature engineering and preprocessing
- `DataValidator`: Input validation and sanitization

**Example:**
```python
from src.core.data import DataLoaderFactory

loader = DataLoaderFactory.create('mark_six', config.data)
train_data, val_data = loader.load_and_split()
```

### **Training Module** (`src/core/training/`)

**Components:**
- `Trainer`: Core training logic (epoch loops, validation)
- `TrainingEngine`: Model-specific training engines
- `TrainingCallbacks`: Progress monitoring, early stopping

**Example:**
```python
from src.core.training import Trainer

trainer = Trainer(
    model=model,
    config=config.training,
    device=device
)
result = trainer.train(train_data, val_data)
```

### **Inference Module** (`src/core/inference/`)

**Components:**
- `Generator`: Number combination generation
- `Scorer`: Individual scoring methods
- `Ensemble`: Ensemble combination methods

**Example:**
```python
from src.core.inference import EnsembleGenerator

generator = EnsembleGenerator(
    model=model,
    scorers=scorers,
    config=config.inference
)
combinations = generator.generate(num_sets=10)
```

## Infrastructure Components

### **Configuration System** (`src/infrastructure/config/`)

**Structured Configuration:**
```python
@dataclass
class ModelConfig:
    latent_dim: int = 64
    learning_rate: float = 5e-5
    # ... other parameters

@dataclass  
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    # ... other parameters
```

**Environment Support:**
```python
from src.infrastructure.config import get_config

# Load environment-specific config
config = get_config(environment='production')
config = get_config(environment='development')
```

**Configuration Presets:**
```python
config_manager = get_config_manager()
config_manager.config_path = Path("config/model_presets/fast_training.yml")
config = config_manager.load_config()
```

### **Logging System** (`src/infrastructure/logging/`)

**Structured Logging:**
```python
from src.infrastructure.logging import get_logger, log_with_context

logger = get_logger(__name__)

# Simple logging
logger.info("Training started")

# Structured logging with context
with log_with_context(logger, epoch=5, batch_size=16) as log:
    log.info("Processing batch", batch_id=123, loss=0.45)
```

**Output Formats:**
- **Console**: Colored, human-readable format
- **File**: JSON format for parsing and analysis
- **Structured**: Includes metadata, timestamps, context

### **Storage System** (`src/infrastructure/storage/`)

**Model Persistence:**
```python
from src.infrastructure.storage import ModelRepository

repo = ModelRepository(config.paths)
repo.save_model(model, 'cvae_trained.pth', metadata={'epoch': 10})
model = repo.load_model('cvae_trained.pth')
```

**Data Persistence:**
```python
from src.infrastructure.storage import DataRepository

data_repo = DataRepository(config.paths)
data_repo.save_processed_data(data, 'features.pkl')
```

## Application Layer

### **Services** (`src/application/services/`)

Business logic encapsulated in service classes:

```python
from src.application.services.base import BaseService

class TrainingService(BaseService):
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        
    def train_model(self, model_type: str = 'cvae') -> ServiceResult:
        try:
            # Initialize components
            model = ModelFactory.create(model_type, self.config['model'])
            trainer = Trainer(model, self.config['training'])
            
            # Execute training
            result = trainer.train(train_data, val_data)
            
            return self.success_result(data=result)
        except Exception as e:
            return self.handle_error(e, "Training failed")
```

**Service Benefits:**
- **Consistent error handling**: All services return ServiceResult
- **Logging integration**: Automatic logging of service operations
- **Configuration injection**: Services receive configuration
- **Testability**: Easy to mock and test in isolation

### **CLI Interface** (`src/application/cli/`)

**Command Pattern:**
```python
from src.application.cli.commands.base import BaseCommand

class TrainCommand(BaseCommand):
    def execute(self, args) -> int:
        service = TrainingService(self.config_manager)
        result = service.train_model()
        
        if result.success:
            self.logger.info("Training completed successfully")
            return 0
        else:
            self.logger.error(f"Training failed: {result.error}")
            return 1
```

## Data Flow

### **Training Flow**
```
Raw Data → DataLoader → FeatureProcessor → Model → Trainer → Checkpoint
    ↑                                                     ↓
Config ←→ ConfigManager ←→ Logger ←→ ModelRepository ←→ Storage
```

### **Inference Flow**
```
Historical Data → FeatureProcessor → Model → Generator → Ensemble → Results
         ↑                                                    ↓
   Config ←→ ConfigManager ←→ Logger ←→ Scorers ←→ Formatter ←→ Output
```

### **Configuration Flow**
```
YAML Files → ConfigManager → Dataclass Objects → Components
     ↑              ↓
Environment    Validation
Variables      
```

## Design Patterns Used

### **Factory Pattern**
- `ModelFactory`: Creates model instances
- `DataLoaderFactory`: Creates data loaders
- `ScorerFactory`: Creates scoring components

### **Strategy Pattern**
- `BaseScorer`: Different scoring strategies
- `BaseEnsemble`: Different ensemble methods
- `TrainingEngine`: Different training strategies

### **Observer Pattern**
- `TrainingCallbacks`: Monitor training progress
- `LoggingHandlers`: Handle different log outputs

### **Repository Pattern**
- `ModelRepository`: Model persistence
- `DataRepository`: Data persistence
- `ConfigRepository`: Configuration storage

### **Service Layer Pattern**
- `TrainingService`: Training business logic
- `InferenceService`: Inference business logic
- `EvaluationService`: Evaluation business logic

## Extension Points

### **Adding New Models**
1. Inherit from `BaseModel`
2. Register with `@register_model` decorator
3. Implement required abstract methods
4. Add configuration section if needed

### **Adding New Scorers**
1. Inherit from `BaseScorer`
2. Implement `score()` and `get_name()` methods
3. Register with `ScorerFactory`
4. Add to ensemble configuration

### **Adding New Data Sources**
1. Inherit from `BaseDataLoader`
2. Implement data loading and preprocessing
3. Register with `DataLoaderFactory`
4. Add configuration section

### **Adding New Services**
1. Inherit from `BaseService`
2. Implement business logic methods
3. Return `ServiceResult` objects
4. Add CLI command if needed

## Error Handling Strategy

### **Layered Error Handling**
- **Infrastructure**: Technical errors (device, I/O, configuration)
- **Core**: Business logic errors (model training, data processing)
- **Application**: User interface errors (validation, formatting)

### **Error Propagation**
```python
# Service layer - convert exceptions to ServiceResult
try:
    result = model.train()
    return ServiceResult(success=True, data=result)
except ModelError as e:
    return ServiceResult(success=False, error=str(e))

# CLI layer - convert ServiceResult to exit codes
result = service.train_model()
if result.success:
    print("Training completed")
    return 0
else:
    print(f"Error: {result.error}")
    return 1
```

### **Graceful Degradation**
- GPU failures → CPU fallback
- Scorer failures → Exclude from ensemble
- Config errors → Use defaults with warnings

## Performance Considerations

### **Memory Management**
- Explicit device management in models
- CUDA cache clearing at configured intervals
- Batch size adaptation based on available memory

### **Lazy Loading**
- Configuration loaded on demand
- Models loaded only when needed
- Data loaded in batches

### **Caching**
- Configuration caching per environment
- Model checkpoint caching
- Feature engineering result caching

## Testing Strategy

### **Unit Tests**
- Test individual components in isolation
- Mock dependencies using interfaces
- Focus on business logic correctness

### **Integration Tests**
- Test component interactions
- Use real configuration but test data
- Verify data flow end-to-end

### **System Tests**
- Test complete workflows
- Use production-like configuration
- Verify performance characteristics

This architecture provides a solid foundation for long-term maintainability, extensibility, and testability while preserving the sophisticated machine learning capabilities of the original system.