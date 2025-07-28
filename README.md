# MarkSix AI: Unified Probabilistic Forecasting System

> **A sophisticated, unified AI system for Mark Six lottery analysis using deep learning, statistical modeling, and advanced optimization techniques.**

---

## 🎯 Project Overview

This project implements a comprehensive, multi-stage pipeline to analyze historical Mark Six lottery data and generate probabilistically-informed number combinations. It leverages a **Conditional Variational Autoencoder (CVAE)** with graph neural networks, temporal context modeling, meta-learning, and **automated hyperparameter optimization** to identify high-scoring number sets through advanced generative modeling.

### Key Features
- **🤖 AI-Powered Generation**: CVAE with graph neural networks and temporal modeling
- **📊 Statistical Analysis**: Frequency-based pattern analysis requiring no trained models
- **🔄 Hybrid Approach**: Combined AI and statistical prediction methods
- **🎯 Pareto Front Optimization**: Advanced multi-objective hyperparameter optimization with NSGA-II and TPE
- **⚡ Checkpoint System**: Resume interrupted optimizations with full state preservation
- **🧪 Comprehensive Testing**: Integrated diagnostic and validation tools
- **🎛️ Unified Interface**: All features accessible through single `main.py` entry point
- **📊 Enhanced Training Monitoring**: Comprehensive loss tracking with overfitting detection
- **🚫 Zero Loss Prevention**: Advanced techniques to prevent training loss collapse

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Conda package manager
- An NVIDIA GPU with CUDA is highly recommended for training and optimization

### Installation
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd MarkSix-Probabilistic-Forecasting

# 2. Automated setup (recommended)
python setup.py

# 3. Alternative manual setup
conda env create -f environment.yml
conda activate marksix_ai

# 4. Launch the unified interface
python main.py
```

### First Run
1. **Run the unified interface**: `python main.py`
2. **Start with diagnostics**: Choose Option 6 → Basic System Check
3. **Try statistical predictions**: Choose Option 2 → Statistical Pattern Analysis
4. **For AI predictions**: First train a model with Option 1 → Quick Training
5. **🎯 NEW: Try Pareto Front optimization**: Choose Option 4 → Option 5 for advanced multi-objective optimization

---

## 🎮 Unified Main Menu Interface

The system provides a comprehensive menu-driven interface with all features integrated:

```
MAIN MENU - UNIFIED MARK SIX PREDICTION SYSTEM v4.1
============================================================
1. Train New Model (Optimized/Quick/Ultra-Quick/Standard)
2. Generate Predictions (AI/Statistical/Hybrid)
3. Evaluate Trained Model
4. Optimize Hyperparameters (including NEW Pareto Front Multi-Objective)
5. View Model Information
6. System Diagnostics & Testing
7. Exit
============================================================
```

### 🧠 Option 1: Enhanced Training Modes
Choose from multiple training approaches based on your needs:

- **🔧 Optimized Training**: 20 epochs, ~94 min, hardware-optimized for best results ✅ **FIXED**
- **⚡ Quick Training**: 5 epochs, ~15 min, perfect for testing and development
- **🏃 Ultra-Quick Training**: 3 epochs, ~5 min, minimal model for immediate testing
- **⚙️ Standard Training**: Fully configurable using original pipeline

### 🎯 Option 2: Unified Prediction Methods
Generate number combinations using three different approaches:

- **🤖 AI Model Inference**: 
  - Uses trained CVAE + Meta-Learner models
  - Temperature control for creativity vs. conservatism
  - Optional I-Ching scorer integration
  - Confidence-based selection

- **📊 Statistical Pattern Analysis**: 
  - No trained models required
  - Frequency-based historical analysis
  - Conservative, Balanced, or Creative modes
  - Immediate results without training

- **🔄 Hybrid Approach**: 
  - Combines AI and statistical methods
  - Generates sets using both approaches
  - Provides diverse prediction strategies

### 📈 Option 3: Model Evaluation
Comprehensive model performance assessment:
- Generation quality testing
- Ensemble ranking performance
- Reconstruction accuracy analysis
- Latent space quality evaluation
- Win rate calculations

### ⚙️ Option 4: Advanced Hyperparameter Optimization
Professional-grade optimization with bulletproof safeguards:

- **🔍 Quick Validation**: Pre-flight pipeline testing (30 seconds)
- **🚀 Thorough Search**: 8+ hour production optimization with full safeguards
- **⚖️ Standard Optimization**: 1-2 hour balanced search
- **🎛️ Custom Configuration**: Manual preset selection with expert options
- **🎯 Pareto Front Multi-Objective**: Advanced NSGA-II and TPE optimization with checkpoint system

#### 🎯 NEW: Pareto Front Multi-Objective Optimization (Option 4.5)
The most advanced optimization method featuring:

- **NSGA-II (Evolutionary Algorithm)**: Global search with population-based optimization
- **TPE/Optuna (Multi-Objective Bayesian)**: Sample-efficient optimization with learning
- **Multi-Objective Functions** (prioritized by weight): 
  - Model Complexity (minimize overfitting risk) - **Weight: 1.0 (HIGH)**
  - JSD Alignment Fidelity (statistical realism with historical data) - **Weight: 1.0 (HIGH)**
  - Training Time (minimize computational cost) - **Weight: 0.8 (MEDIUM-HIGH)**
  - Accuracy (maximize model prediction performance) - **Weight: 0.6 (MEDIUM)**
- **Interactive Pareto Front**: Choose from multiple optimal trade-off solutions
- **Checkpoint System**: Resume interrupted optimizations with full state preservation
- **Automatic Integration**: Selected parameters flow seamlessly to training pipeline

### 📋 Option 5: Model Information Dashboard
Complete model status and information:
- Standard model availability
- Alternative model variants
- Model sizes and modification dates
- Optimization results summary
- Training history

### 🧪 Option 6: System Diagnostics & Testing
Comprehensive system validation and testing:

- **🔍 Basic System Check**: Hardware, data, and environment validation
- **🧬 Model Compatibility Test**: Cross-version model testing and validation
- **🔬 Full System Validation**: End-to-end testing of all components

---

## 🏗️ System Architecture

### Core Neural Network Components

#### 1. CVAE Core Model (`src/cvae_model.py`)
- **Architecture**: Conditional Variational Autoencoder with encoder-decoder structure
- **Latent Dimensions**: 64-dimensional compressed representation space
- **Function**: Learns to reconstruct number combinations while discovering latent patterns
- **Training**: Multi-component loss combining reconstruction, KL divergence, and contrastive learning

#### 2. Graph Neural Network (`src/graph_encoder.py`)
- **Architecture**: Graph Attention Network (GAT) for modeling number relationships
- **Function**: Captures complex co-occurrence patterns between lottery numbers
- **Features**: Multi-head attention mechanism for relationship learning

#### 3. Temporal Context Module (`src/temporal_context.py`)
- **Architecture**: LSTM with attention mechanism for sequence modeling
- **Function**: Processes historical lottery draw sequences to learn temporal patterns
- **Features**: Bidirectional LSTM with attention-based context aggregation

#### 4. Meta-Learning Component (`src/meta_learner.py`)
- **Architecture**: Attention-based neural network for ensemble weight optimization
- **Function**: Dynamically adapts scoring weights based on input patterns
- **Features**: Confidence estimation and uncertainty quantification

#### 5. Enhanced Feature Engineering (`src/feature_engineering.py`)
- **Temporal sequences and graph embeddings**
- **Statistical properties (sum, mean, variance, odd/even ratios)**
- **Historical frequencies and pair analysis**
- **Delta features and number group distributions**
- **Graph-based relationship features**

#### 6. JSD Alignment Fidelity (`src/evaluation_pipeline.py`)
- **Statistical Metric**: Jensen-Shannon Distance alignment between model and historical data
- **Objective**: Minimize |Sample_JSD - Historical_JSD| to ensure statistical realism
- **Integration**: Seamlessly integrated into Pareto Front multi-objective optimization
- **Benefits**: Ensures models replicate true lottery statistical properties, not just accuracy

### Advanced Pipeline Components

#### 7. Training Engine (`src/cvae_engine.py`)
- Conservative training with stability checks
- Error recovery and comprehensive logging
- Mixed precision handling with overflow detection
- Gradient management and numerical stability

#### 8. Inference Pipeline (`src/inference_pipeline.py`)
- Sophisticated number generation using CVAE sampling
- Meta-learned ensemble weights
- Confidence-based selection and iterative refinement
- Local search with CVAE-guided exploration

#### 9. Multi-Objective Optimization System (`src/optimization/`)
- **Pareto Front Optimization**: NSGA-II and TPE/Optuna algorithms
- **Four-Objective Optimization**: Model Complexity, Statistical Fidelity, Training Time, Accuracy
- **Interactive Selection**: Choose optimal trade-offs from Pareto Front
- **Checkpoint System**: Resume interrupted optimizations with full state preservation

---

## 📊 Project Structure

```
MarkSix-Probabilistic-Forecasting/
├── README.md                    # 📖 This unified documentation
├── main.py                      # 🚀 Unified entry point interface
├── CLAUDE.md                    # 🤖 Claude Code development guidance
├── setup.py                     # 📦 Automated environment setup
├── environment.yml              # 🐍 Conda environment specification
├── 
├── src/                         # 💻 Core source code
│   ├── config.py               # ⚙️ Centralized configuration
│   ├── cvae_model.py           # 🧠 CVAE architecture
│   ├── cvae_engine.py          # 🔧 CVAE training engine
│   ├── graph_encoder.py        # 🕸️ Graph neural networks
│   ├── temporal_context.py     # ⏰ Temporal modeling
│   ├── meta_learner.py         # 🎯 Meta-learning ensemble
│   ├── feature_engineering.py  # 📊 Feature extraction
│   ├── training_pipeline.py    # 🚂 Training orchestration
│   ├── inference_pipeline.py   # 🎲 Number generation
│   ├── evaluation_pipeline.py  # 📈 Model evaluation
│   ├── hyperparameter_optimizer.py # ⚙️ Auto-optimization
│   └── optimization/           # 🔧 Optimization modules
│
├── data/                       # 📊 Data storage
│   ├── raw/Mark_Six.csv       # 🎰 Historical lottery data
│   └── processed/             # 📈 Processed datasets
├── models/                     # 🤖 Trained model artifacts
├── outputs/                    # 📋 Training logs and plots
├── optimization_results/       # 📊 Optimization outputs
├── thorough_search_results/    # 🎯 Production optimization results
├── hyperparameter_results/     # ⚙️ Hyperparameter trials
├── backup_standalone_scripts/  # 🗄️ Archived legacy scripts
├── 
├── tests/                      # 🧪 Test suite
├── docs/                       # 📚 Additional documentation
├── notebooks/                  # 📓 Analysis notebooks
├── scripts/                    # 🛠️ Utility scripts
├── config/                     # ⚙️ Configuration files
└── requirements/               # 📋 Dependencies
```

---

## 🎮 Usage Workflows

### 🚀 Quick Prediction Workflow (No Training Required)
```bash
python main.py
# Choose: 2. Generate Predictions
# Select: Statistical Pattern Analysis
# Configure: Balanced mode, 5 sets
# Result: Instant predictions based on historical patterns
```

### 🧠 AI Model Workflow
```bash
python main.py
# Step 1: Choose 1. Train New Model → Quick Training (15 min)
# Step 2: Choose 2. Generate Predictions → AI Model Inference
# Step 3: Choose 3. Evaluate Trained Model (optional)
```

### ⚙️ Production Optimization Workflow
```bash
python main.py
# Step 1: Choose 4. Optimize Hyperparameters → Quick Validation
# Step 2: Choose 4. Optimize Hyperparameters → Thorough Search (8+ hours)
# Step 3: Choose 1. Train New Model → Use optimized parameters
# Step 4: Choose 2. Generate Predictions → High-quality AI inference
```

### 🧪 Development and Testing Workflow
```bash
python main.py
# Step 1: Choose 6. System Diagnostics → Basic System Check
# Step 2: Choose 6. System Diagnostics → Model Compatibility Test
# Step 3: Choose 1. Train New Model → Ultra-Quick Training (5 min)
# Step 4: Choose 6. System Diagnostics → Full System Validation
```

---

## 🛡️ Bulletproof Optimization System

### 🚀 Thorough Search Features
- **Pre-flight Validation**: Comprehensive checks before starting
- **Checkpoint System**: Automatic saving every 5 trials + emergency checkpoints
- **Recovery Capability**: Resume from interruptions without losing progress
- **Model Validation**: Ensures optimized models work with inference pipeline
- **Resource Monitoring**: Tracks memory, disk space, and performance
- **Error Handling**: Graceful degradation and detailed error reporting

### 📋 Optimization Validation Checklist
- ✅ **Environment**: PyTorch + CUDA available
- ✅ **Data File**: `data/raw/Mark_Six.csv` exists and readable
- ✅ **Disk Space**: At least 5GB free space
- ✅ **Optimization Setup**: All presets and algorithms available
- ✅ **Model Compatibility**: CVAE and Meta-learner can be instantiated
- ✅ **Inference Pipeline**: All components importable and functional

### 🔄 Recovery and Resumption
```bash
# Automatic recovery
python main.py → Option 4 → Thorough Search (detects existing checkpoints)

# Check optimization status
cat thorough_search_results/optimization_status.json

# Manual checkpoint management
ls thorough_search_results/checkpoints/
```

### 📈 Expected Performance Improvements
With proper optimization, expect:
- **15-30% better model performance**
- **More stable training convergence**
- **Better number generation quality**
- **Reduced overfitting**
- **Improved ensemble weights**

---

## 🎯 Integration and Cleanup Summary

### ✅ Completed Major Integrations

#### 🔄 Standalone Script Integration
All temporal and feature-specific scripts have been successfully integrated:

| Original Script | Integration Status | Menu Location |
|---|---|---|
| `train_optimized.py` | ✅ Integrated | Option 1 → Optimized Training |
| `quick_train.py` | ✅ Integrated | Option 1 → Quick Training |
| `quick_predict.py` | ✅ Integrated | Option 2 → Statistical Analysis |
| `bulletproof_thorough_search.py` | ✅ Integrated | Option 4 → Thorough Search |
| `validate_thorough_search_pipeline.py` | ✅ Integrated | Option 4 → Validation |
| `test_new_model.py` | ✅ Integrated | Option 6 → Model Compatibility |
| `test_main_inference.py` | ✅ Integrated | Option 6 → System Diagnostics |
| `verify_fix.py` | ✅ Integrated | Option 6 → System Diagnostics |

#### 🧹 Project Cleanup Achievements
- **✅ Unified Test System**: All tests organized in `/tests/` with interactive runner
- **✅ Documentation Organization**: All docs consolidated in `/docs/` directory
- **✅ Clean Root Structure**: Eliminated duplicate files and redundancy
- **✅ Script Integration**: 14 standalone scripts integrated into main.py
- **✅ Backup Strategy**: All removed files safely archived

### 💡 Key Benefits

#### For Users
- **Single Entry Point**: All functionality accessible through `python main.py`
- **Clean Interface**: No confusion about which script to run
- **Enhanced Options**: More training, prediction, and optimization modes
- **Better Diagnostics**: Comprehensive testing and validation tools

#### For Developers
- **Maintainable Code**: Centralized functionality in unified interface
- **Reduced Complexity**: No scattered temporal scripts
- **Better Organization**: Clear separation of concerns within main.py
- **Easy Extension**: New features integrate into existing menu structure

---

## 📚 Configuration and Development Guide

### 🔧 Development Commands

#### Environment Setup
```bash
# Automated setup (recommended)
python setup.py

# Manual setup
conda env create -f environment.yml
conda activate marksix_ai
```

#### Main Application
```bash
# Primary entry point - interactive CLI with 7 options
python main.py
```

#### Testing and Validation
```bash
# Test hyperparameter optimization functionality
python test_hyperparameter_optimization.py

# Debug model architecture and training pipeline
python test_model_debug.py

# Run unified test suite
python run_tests.py
```

### 🎛️ Configuration System

#### Main Configuration
- All parameters centralized in `src/config.py`
- Conservative settings for stability
- Device auto-detection with CUDA/CPU fallback

#### Configuration Presets
- `fast_training`: Quick results, lower quality
- `balanced`: Good balance of speed and quality (default)
- `high_quality`: Best results, longer training time
- `experimental`: Cutting-edge parameters for research

### 📊 Data Requirements

#### Input Data
- **Required**: `data/raw/Mark_Six.csv` - Historical Mark Six lottery data
- **Format**: CSV with columns for Draw, Date, Winning numbers, Extra number, and statistics
- **Minimum**: 100+ historical draws for meaningful training

#### Generated Data
- `data/processed/`: Automatically processed data files
- `models/`: Trained model artifacts (.pth files)
- `models/pareto_front/`: NEW - Pareto Front optimization results
  - `models/pareto_front/nsga2/`: NSGA-II algorithm results  
  - `models/pareto_front/tpe/`: TPE/Optuna algorithm results
- `models/best_parameters/`: Selected parameters from optimization
- `models/optimization_trials/`: Trial history and checkpoints
- `outputs/`: Training logs, plots, and debug reports
- `optimization_results/`: Legacy optimization results (maintained for compatibility)

## 🧹 Project Cleanup & Consolidation (Version 4.1)

This version represents a major cleanup and consolidation effort:

### ✅ Removed Standalone Scripts
**Deleted 8+ standalone scripts** that were previously scattered in the root directory:
- `quick_train.py`, `quick_predict.py`, `ultra_quick_train.py`
- `train_optimized.py`, `use_ultra_model.py`, `legacy_inference.py`
- `check_dependencies.py`, `run_tests.py`, `setup_quick_model.py`

### ✅ Unified Interface
**All functionality consolidated** into the main menu system (`main.py`):
- Single entry point for all features
- Consistent user experience across all operations
- Integrated error handling and validation
- No more scattered scripts to manage

### ✅ Enhanced Architecture  
**Major additions** to the codebase:
- 5 new Pareto Front optimization modules with 1000+ lines of advanced algorithms
- Comprehensive checkpoint system with interrupt handling
- Multi-objective optimization with NSGA-II and TPE algorithms
- Automatic parameter flow from optimization to training

### 🛠️ Development Best Practices

1. **Unified Interface Only**: Use `python main.py` - no standalone scripts
2. **Pareto Front First**: Try Option 4.5 for advanced optimization  
3. **Always test first**: Run diagnostic tools before making changes
4. **Monitor memory usage**: Watch for OOM errors on GPU
5. **Validate data**: Ensure CSV format matches expected structure
6. **Clean development**: All temporary scripts auto-deleted after use

### 🐛 Common Issues and Solutions

#### Memory Issues
- Reduce `batch_size` in config (default: 8)
- Enable CPU fallback in config
- Reduce model parameters (already conservative)

#### Training Instability
- Use conservative learning rate (5e-5)
- Enable gradient clipping (0.5)
- Reduce KL and contrastive loss weights

#### GPU Issues
- System auto-detects and falls back to CPU
- Test GPU functionality with test scripts
- Check CUDA memory with diagnostic tools

---

## 🎯 Performance and System Requirements

### 💻 System Requirements
- **Minimum**: Python 3.10+, 8GB RAM, CPU-only training
- **Recommended**: Python 3.10+, 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: Python 3.10+, 32GB+ RAM, NVIDIA GPU with 10GB+ VRAM

### ⚡ Performance Tips
1. **Use GPU when available** - Significantly faster training and optimization
2. **Start with hyperparameter optimization** - Can improve performance by 15-30%
3. **Use configuration presets** - Pre-tuned settings for different hardware
4. **Monitor system resources** - Adjust batch size if memory errors occur

### 🔧 Troubleshooting
- **CUDA out of memory**: Reduce batch size in configuration
- **Slow training**: Use CPU presets or reduce model size
- **Poor performance**: Run hyperparameter optimization
- **Import errors**: Check conda environment activation

### 📈 Expected Results and Validation

#### Success Criteria
- ✅ **Model Training**: Successful convergence with stable loss curves
- ✅ **Generation Quality**: Diverse, valid number combinations
- ✅ **Optimization**: Performance improvements with optimized parameters
- ✅ **Statistical Analysis**: Immediate predictions without model requirements
- ✅ **System Integration**: All components working together seamlessly

---

## 🚀 Getting the Best Results

### 🎯 Recommended Workflow
1. **🔍 First Run**: Start with system diagnostics and statistical predictions
2. **⚡ Quick Training**: Train a quick model to test the AI pipeline
3. **⚙️ Optimization**: Run hyperparameter optimization for best results
4. **🧠 Production Training**: Train with optimized parameters
5. **📊 Evaluation**: Validate model performance
6. **🎲 Generation**: Use trained models for high-quality predictions

### 💡 Pro Tips
- **Start Simple**: Begin with statistical analysis to understand the system
- **Test Everything**: Use the diagnostic tools to validate your setup
- **Optimize First**: Hyperparameter optimization significantly improves results
- **Monitor Resources**: Keep an eye on GPU memory and disk space
- **Save Configurations**: Backup successful parameter combinations

### 🎉 Success Metrics
- **Training**: Models converge without errors
- **Generation**: Produces valid number combinations
- **Performance**: Win rate above 50% in evaluation
- **Stability**: Consistent results across multiple runs

---

## 📝 License and Disclaimer

This project is provided as-is for educational and research purposes. The system finds patterns in historical data but **cannot guarantee future lottery outcomes**. Please:

- Use responsibly and within your means
- Ensure compliance with local regulations regarding lottery systems
- Remember that lottery outcomes are fundamentally random
- Consider this as a learning tool for AI and statistical modeling

---

## 🌟 Summary

The **MarkSix AI Unified Probabilistic Forecasting System** provides:

- 🎯 **Complete Integration**: All features accessible through single interface
- 🤖 **Advanced AI**: State-of-the-art CVAE with graph and temporal modeling
- 📊 **Statistical Analysis**: Immediate predictions without model training
- ⚙️ **Bulletproof Optimization**: Production-grade hyperparameter tuning
- 🧪 **Comprehensive Testing**: Extensive validation and diagnostic tools
- 🔧 **Professional Quality**: Clean architecture and maintainable codebase

**Ready to explore? Run `python main.py` and discover the power of unified AI-driven lottery analysis!**

## 🔧 Recent Updates (July 2025)

### 🆕 Major Overfitting Fix (July 28, 2025)
Comprehensive solution to zero training loss and overfitting issues:

- **🚫 Zero Loss Prevention**: Fixed root causes of training loss collapse (overflow masking, aggressive clamping, temporal leakage)
- **📊 Advanced Monitoring**: New LossMonitor class with real-time pattern detection and diagnostic reports
- **🔄 Proper Data Splitting**: Temporal splitting with 75%/5%/20% train/gap/validation prevents data leakage
- **🧠 KL Collapse Prevention**: β-VAE annealing (5 epochs 0.0→1.0) with diversity bonuses and regularization
- **🔍 Enhanced Debugging**: Comprehensive loss component analysis with automatic problem detection

### ✅ Fixed Optimized Training Mode (July 24, 2025)
The **Optimized Training** option (Menu Option 1.1) has been fully debugged and fixed:

- **Fixed function signature mismatches** that caused training crashes
- **Corrected optimizer structure** to use separate optimizers for CVAE and meta-learner components  
- **Enhanced model saving** to save to standard paths for seamless inference integration
- **Added comprehensive error handling** and validation

**Now Ready**: Complete workflow from Pareto Front Optimization → Optimized Training → AI Model Inference works seamlessly!

---

*Last updated: July 2025 | Version 4.2 - Comprehensive Overfitting Prevention and Training Stability*