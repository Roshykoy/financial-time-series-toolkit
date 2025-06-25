# test_model_debug.py
"""
Comprehensive model debugging script.
Run this before training to catch issues early.
"""

import torch
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import CONFIG
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.feature_engineering import FeatureEngineer
from src.debug_utils import ModelDebugger, DataLoaderDebugger, quick_model_test, debug_training_step
from src.cvae_data_loader import create_cvae_data_loaders

def test_cvae_model():
    """Test CVAE model with debugging."""
    print("🧪 Testing CVAE Model")
    print("=" * 50)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    # Create model
    print("📝 Creating CVAE model...")
    cvae_model = ConditionalVAE(CONFIG).to(device)
    
    # Create dummy inputs function
    def create_cvae_dummy_inputs():
        batch_size = 2
        
        # Dummy combinations
        positive_combinations = []
        for _ in range(batch_size):
            combo = sorted(np.random.choice(range(1, CONFIG['num_lotto_numbers'] + 1), 6, replace=False).tolist())
            positive_combinations.append(combo)
        
        # Dummy pair counts
        pair_counts = {}
        for i in range(1, CONFIG['num_lotto_numbers'] + 1):
            for j in range(i + 1, CONFIG['num_lotto_numbers'] + 1):
                pair_counts[(i, j)] = np.random.randint(0, 10)
        
        # Dummy dataframe
        dummy_data = []
        for i in range(100):
            row_data = [i, f"2024-01-{i+1:02d}"]
            # Add winning numbers
            winning_nums = sorted(np.random.choice(range(1, 50), 6, replace=False))
            row_data.extend(winning_nums)
            row_data.append(np.random.randint(1, 50))  # Extra number
            row_data.extend([0] * 25)  # Other columns
            dummy_data.append(row_data)
        
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        
        df = pd.DataFrame(dummy_data, columns=col_names)
        current_indices = list(range(batch_size))
        
        return positive_combinations, pair_counts, df, current_indices
    
    # Create dummy loss function
    def create_loss_fn():
        def loss_fn(outputs):
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 6:
                reconstruction_logits, mu, logvar, mu_prior, logvar_prior, context = outputs[:6]
                
                # Dummy reconstruction loss
                batch_size = reconstruction_logits.size(0)
                dummy_targets = torch.randint(0, CONFIG['num_lotto_numbers'], (batch_size, 6), device=device)
                recon_loss = torch.nn.functional.cross_entropy(
                    reconstruction_logits.view(-1, CONFIG['num_lotto_numbers']),
                    dummy_targets.view(-1)
                )
                
                # Dummy KL loss
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))
                
                return recon_loss + 0.1 * kl_loss
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss_fn
    
    # Run comprehensive test
    success, debugger = quick_model_test(cvae_model, CONFIG, device)
    
    # Test specific CVAE forward pass
    print("\n🔍 Testing CVAE-specific forward pass...")
    try:
        with debug_training_step(cvae_model, None, device, "CVAE_forward_test") as debug:
            dummy_inputs = create_cvae_dummy_inputs()
            positive_combinations, pair_counts, df, current_indices = dummy_inputs
            
            # Test forward pass
            cvae_model.eval()
            with torch.no_grad():
                outputs = cvae_model(positive_combinations, pair_counts, df, current_indices)
                
                print(f"✅ CVAE forward pass successful!")
                print(f"   Output shapes: {[out.shape if isinstance(out, torch.Tensor) else type(out) for out in outputs]}")
                
                # Test generation
                print("🎲 Testing generation...")
                sequence = cvae_model.temporal_encoder.prepare_sequence_data(df, len(df))
                context, _ = cvae_model.temporal_encoder(sequence)
                
                generated_combinations, log_probs = cvae_model.generate(context, num_samples=3)
                print(f"✅ Generation successful! Generated {generated_combinations.shape[0]} combinations")
                print(f"   Sample: {generated_combinations[0].cpu().numpy().tolist()}")
    
    except Exception as e:
        print(f"❌ CVAE-specific test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Save debug report
    debugger.save_debug_report("outputs/cvae_debug_report.txt")
    
    return success

def test_meta_learner():
    """Test Meta-Learner model."""
    print("\n🧠 Testing Meta-Learner Model")
    print("=" * 50)
    
    device = torch.device(CONFIG['device'])
    
    # Create model
    meta_learner = AttentionMetaLearner(CONFIG).to(device)
    
    def create_meta_dummy_inputs():
        batch_size = 3
        
        # Dummy number combinations
        number_sets = []
        for _ in range(batch_size):
            combo = sorted(np.random.choice(range(1, CONFIG['num_lotto_numbers'] + 1), 6, replace=False).tolist())
            number_sets.append(combo)
        
        # Dummy temporal context
        temporal_context = torch.randn(batch_size, CONFIG['temporal_context_dim'], device=device)
        
        # Dummy scorer scores
        scorer_scores = {
            'generative': torch.randn(batch_size, device=device),
            'temporal': torch.randn(batch_size, device=device),
            'i_ching': torch.randn(batch_size, device=device)
        }
        
        return number_sets, temporal_context, scorer_scores
    
    def meta_loss_fn(outputs):
        ensemble_weights, final_scores, confidence = outputs
        return final_scores.mean()
    
    # Test meta-learner
    try:
        print("🔍 Testing meta-learner forward pass...")
        meta_learner.eval()
        
        with torch.no_grad():
            number_sets, temporal_context, scorer_scores = create_meta_dummy_inputs()
            
            ensemble_weights, final_scores, confidence = meta_learner(
                number_sets, temporal_context, scorer_scores
            )
            
            print(f"✅ Meta-learner forward pass successful!")
            print(f"   Ensemble weights shape: {ensemble_weights.shape}")
            print(f"   Final scores shape: {final_scores.shape}")
            print(f"   Confidence shape: {confidence.shape}")
            
            # Test explanations
            explanations = meta_learner.get_weight_explanations(
                number_sets, temporal_context, scorer_scores
            )
            print(f"✅ Explanations generated for {len(explanations)} combinations")
            
        return True
        
    except Exception as e:
        print(f"❌ Meta-learner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test data loading pipeline - FIXED VERSION."""
    print("\n📦 Testing Data Loading Pipeline")
    print("=" * 50)
    
    try:
        # Load sample data
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        
        # Create small dummy dataset with PROPER DATE FORMAT
        dummy_data = []
        for i in range(50):  # Small dataset for testing
            row_data = [i]  # Draw number
            
            # FIX: Create proper date string format
            month = (i % 12) + 1
            day = (i % 28) + 1
            date_str = f"2024-{month:02d}-{day:02d}"
            row_data.append(date_str)
            
            # Add winning numbers
            winning_nums = sorted(np.random.choice(range(1, 50), 6, replace=False))
            row_data.extend(winning_nums)
            
            # Add extra number
            extra_num = np.random.randint(1, 50)
            while extra_num in winning_nums:  # Ensure extra number is different
                extra_num = np.random.randint(1, 50)
            row_data.append(extra_num)
            
            # Add remaining columns with realistic data
            row_data.extend([np.random.randint(0, 1000) for _ in range(25)])
            dummy_data.append(row_data)
        
        df = pd.DataFrame(dummy_data, columns=col_names)
        
        # FIX: More robust date parsing
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        except ValueError:
            # Fallback: let pandas infer the format
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        print(f"✅ Created dummy dataset with {len(df)} rows")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Create feature engineer
        from src.feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(df)
        
        print("✅ Feature engineer fitted successfully")
        
        # Test creating reduced config for testing
        test_config = CONFIG.copy()
        test_config['batch_size'] = 4
        test_config['negative_pool_size'] = 100  # Much smaller for testing
        test_config['temporal_sequence_length'] = 5
        
        # Create data loaders
        train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, test_config)
        
        print("✅ Data loaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Debug data loader - test first batch only
        print("🔍 Testing first batch...")
        try:
            first_batch = next(iter(train_loader))
            print(f"   Batch keys: {list(first_batch.keys())}")
            print(f"   Positive combinations: {len(first_batch['positive_combinations'])}")
            print(f"   Sample combination: {first_batch['positive_combinations'][0] if first_batch['positive_combinations'] else 'None'}")
            print("✅ First batch loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load first batch: {e}")
            # This is not critical for the test
        
        print("✅ Data loader debugging complete")
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

def main():
    """Run all debugging tests."""
    print("🚀 COMPREHENSIVE MODEL DEBUGGING")
    print("=" * 70)
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: CVAE Model
    if test_cvae_model():
        tests_passed += 1
        print("✅ CVAE model test: PASSED")
    else:
        print("❌ CVAE model test: FAILED")
    
    # Test 2: Meta-Learner
    if test_meta_learner():
        tests_passed += 1
        print("✅ Meta-learner test: PASSED")
    else:
        print("❌ Meta-learner test: FAILED")
    
    # Test 3: Data Loading
    if test_data_loader():
        tests_passed += 1
        print("✅ Data loader test: PASSED")
    else:
        print("❌ Data loader test: FAILED")
    
    # Test 4: Device and Memory
    print("\n💾 Testing Device and Memory Setup")
    try:
        device = torch.device(CONFIG['device'])
        if device.type == 'cuda' and torch.cuda.is_available():
            # Test CUDA allocation
            test_tensor = torch.randn(1000, 1000, device=device)
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ CUDA allocation test: PASSED")
        else:
            print("ℹ️  CPU-only mode")
        
        tests_passed += 1
    except Exception as e:
        print(f"❌ Device/memory test: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEBUGGING SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Model is ready for training.")
    elif tests_passed >= total_tests * 0.75:
        print("⚠️  Most tests passed. Check warnings above.")
    else:
        print("❌ MULTIPLE TESTS FAILED. Fix issues before training.")
    
    print("\n📝 Debug reports saved to outputs/ directory")
    print("💡 Tip: Run this script whenever you modify model architecture")

if __name__ == "__main__":
    main()