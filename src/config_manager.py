# src/config_manager.py
import json
import os
from typing import Dict, Any
from src.config import CONFIG

class ConfigurationManager:
    """Manages configuration files and hyperparameter presets."""
    
    def __init__(self):
        self.config_dir = "config"
        self.presets_file = os.path.join(self.config_dir, "presets.json")
        os.makedirs(self.config_dir, exist_ok=True)
        self._load_presets()
    
    def _load_presets(self):
        """Load predefined configuration presets."""
        self.presets = {
            "fast_training": {
                "learning_rate": 1e-3,
                "hidden_size": 256,
                "num_layers": 4,
                "dropout": 0.2,
                "batch_size": 64,
                "epochs": 5,
                "margin": 0.5,
                "negative_samples": 16,
                "use_sam_optimizer": False
            },
            "balanced": {
                "learning_rate": 1e-4,
                "hidden_size": 256,
                "num_layers": 6,
                "dropout": 0.15,
                "batch_size": 64,
                "epochs": 15,
                "margin": 0.5,
                "negative_samples": 32,
                "use_sam_optimizer": True,
                "rho": 0.05
            },
            "high_quality": {
                "learning_rate": 5e-5,
                "hidden_size": 512,
                "num_layers": 8,
                "dropout": 0.1,
                "batch_size": 32,
                "epochs": 25,
                "margin": 0.7,
                "negative_samples": 64,
                "use_sam_optimizer": True,
                "rho": 0.05
            },
            "experimental": {
                "learning_rate": 1e-4,
                "hidden_size": 768,
                "num_layers": 10,
                "dropout": 0.3,
                "batch_size": 128,
                "epochs": 20,
                "margin": 1.0,
                "negative_samples": 48,
                "use_sam_optimizer": True,
                "rho": 0.1
            }
        }
        
        # Save presets to file if it doesn't exist
        if not os.path.exists(self.presets_file):
            self.save_presets()
    
    def save_presets(self):
        """Save presets to file."""
        with open(self.presets_file, 'w') as f:
            json.dump(self.presets, f, indent=2)
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a configuration preset."""
        if preset_name not in self.presets:
            raise ValueError(f"Preset '{preset_name}' not found. Available: {list(self.presets.keys())}")
        return self.presets[preset_name].copy()
    
    def apply_preset(self, preset_name: str):
        """Apply a preset to the current configuration."""
        preset_config = self.get_preset(preset_name)
        CONFIG.update(preset_config)
        print(f"✅ Applied preset '{preset_name}' to current configuration.")
    
    def save_current_as_preset(self, preset_name: str, description: str = ""):
        """Save the current configuration as a new preset."""
        # Extract tunable parameters
        tunable_params = [
            'learning_rate', 'hidden_size', 'num_layers', 'dropout', 
            'batch_size', 'epochs', 'margin', 'negative_samples', 
            'use_sam_optimizer', 'rho'
        ]
        
        new_preset = {}
        for param in tunable_params:
            if param in CONFIG:
                new_preset[param] = CONFIG[param]
        
        if description:
            new_preset['_description'] = description
        
        self.presets[preset_name] = new_preset
        self.save_presets()
        print(f"✅ Saved current configuration as preset '{preset_name}'.")
    
    def list_presets(self):
        """List all available presets."""
        print("\n--- Available Configuration Presets ---")
        for name, config in self.presets.items():
            description = config.get('_description', 'No description')
            print(f"\n📋 {name.upper()}")
            print(f"   Description: {description}")
            print(f"   Key parameters:")
            for key, value in config.items():
                if not key.startswith('_'):
                    print(f"     {key}: {value}")
    
    def compare_with_current(self, preset_name: str):
        """Compare a preset with the current configuration."""
        if preset_name not in self.presets:
            print(f"❌ Preset '{preset_name}' not found.")
            return
        
        preset = self.presets[preset_name]
        print(f"\n--- Comparing Current Config with '{preset_name}' Preset ---")
        
        for key, preset_value in preset.items():
            if key.startswith('_'):
                continue
            
            current_value = CONFIG.get(key, 'Not set')
            status = "✅ Same" if current_value == preset_value else "🔄 Different"
            print(f"{status} {key}: {current_value} → {preset_value}")
    
    def interactive_config_editor(self):
        """Interactive configuration editor."""
        print("\n--- Interactive Configuration Editor ---")
        
        tunable_params = {
            'learning_rate': {'type': float, 'range': (1e-6, 1e-2), 'description': 'Learning rate for optimizer'},
            'hidden_size': {'type': int, 'options': [128, 256, 512, 768, 1024], 'description': 'Hidden layer size'},
            'num_layers': {'type': int, 'range': (2, 12), 'description': 'Number of transformer layers'},
            'dropout': {'type': float, 'range': (0.0, 0.5), 'description': 'Dropout rate for regularization'},
            'batch_size': {'type': int, 'options': [16, 32, 64, 128, 256], 'description': 'Training batch size'},
            'epochs': {'type': int, 'range': (1, 50), 'description': 'Number of training epochs'},
            'margin': {'type': float, 'range': (0.1, 2.0), 'description': 'Margin for ranking loss'},
            'negative_samples': {'type': int, 'range': (8, 128), 'description': 'Number of negative samples'},
            'use_sam_optimizer': {'type': bool, 'description': 'Use Sharpness-Aware Minimization'},
            'rho': {'type': float, 'range': (0.01, 0.2), 'description': 'SAM sharpness parameter'}
        }
        
        while True:
            print(f"\n--- Current Configuration ---")
            for i, (param, info) in enumerate(tunable_params.items(), 1):
                current_value = CONFIG.get(param, 'Not set')
                print(f"{i:2d}. {param}: {current_value} - {info['description']}")
            
            print(f"\n{len(tunable_params)+1:2d}. Save current configuration as preset")
            print(f"{len(tunable_params)+2:2d}. Load preset")
            print(f"{len(tunable_params)+3:2d}. Exit editor")
            
            try:
                choice = int(input(f"\nSelect parameter to edit (1-{len(tunable_params)+3}): "))
                
                if choice <= len(tunable_params):
                    param_name = list(tunable_params.keys())[choice - 1]
                    self._edit_parameter(param_name, tunable_params[param_name])
                
                elif choice == len(tunable_params) + 1:
                    preset_name = input("Enter preset name: ").strip()
                    description = input("Enter description (optional): ").strip()
                    self.save_current_as_preset(preset_name, description)
                
                elif choice == len(tunable_params) + 2:
                    self.list_presets()
                    preset_name = input("Enter preset name to load: ").strip()
                    if preset_name in self.presets:
                        self.apply_preset(preset_name)
                    else:
                        print(f"❌ Preset '{preset_name}' not found.")
                
                elif choice == len(tunable_params) + 3:
                    break
                
                else:
                    print("❌ Invalid choice.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
    
    def _edit_parameter(self, param_name: str, param_info: Dict[str, Any]):
        """Edit a single parameter."""
        current_value = CONFIG.get(param_name, None)
        param_type = param_info['type']
        
        print(f"\nEditing: {param_name}")
        print(f"Description: {param_info['description']}")
        print(f"Current value: {current_value}")
        
        if 'options' in param_info:
            print(f"Available options: {param_info['options']}")
        elif 'range' in param_info:
            print(f"Valid range: {param_info['range']}")
        
        try:
            if param_type == bool:
                new_value = input("Enter new value (true/false): ").strip().lower()
                new_value = new_value in ['true', 't', 'yes', 'y', '1']
            elif param_type == int:
                new_value = int(input("Enter new value: "))
                if 'range' in param_info:
                    min_val, max_val = param_info['range']
                    if not (min_val <= new_value <= max_val):
                        print(f"❌ Value must be between {min_val} and {max_val}")
                        return
                elif 'options' in param_info:
                    if new_value not in param_info['options']:
                        print(f"❌ Value must be one of {param_info['options']}")
                        return
            elif param_type == float:
                new_value = float(input("Enter new value: "))
                if 'range' in param_info:
                    min_val, max_val = param_info['range']
                    if not (min_val <= new_value <= max_val):
                        print(f"❌ Value must be between {min_val} and {max_val}")
                        return
            
            CONFIG[param_name] = new_value
            print(f"✅ Updated {param_name} to {new_value}")
            
        except ValueError:
            print(f"❌ Invalid value for {param_type.__name__}")

def run_config_manager():
    """Main function to run the configuration manager."""
    manager = ConfigurationManager()
    
    while True:
        print("\n" + "="*50)
        print("         CONFIGURATION MANAGER")
        print("="*50)
        print("1. 📋 List Available Presets")
        print("2. 🔧 Apply Preset")
        print("3. 💾 Save Current as Preset")
        print("4. 🔍 Compare with Preset")
        print("5. ✏️  Interactive Editor")
        print("6. 📊 View Current Configuration")
        print("7. ⬅️  Back to Main Menu")
        print("="*50)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        try:
            if choice == '1':
                manager.list_presets()
            
            elif choice == '2':
                manager.list_presets()
                preset_name = input("\nEnter preset name to apply: ").strip()
                if preset_name in manager.presets:
                    manager.apply_preset(preset_name)
                else:
                    print(f"❌ Preset '{preset_name}' not found.")
            
            elif choice == '3':
                preset_name = input("Enter name for new preset: ").strip()
                description = input("Enter description (optional): ").strip()
                manager.save_current_as_preset(preset_name, description)
            
            elif choice == '4':
                manager.list_presets()
                preset_name = input("\nEnter preset name to compare: ").strip()
                manager.compare_with_current(preset_name)
            
            elif choice == '5':
                manager.interactive_config_editor()
            
            elif choice == '6':
                print("\n--- Current Configuration ---")
                for key, value in sorted(CONFIG.items()):
                    print(f"  {key}: {value}")
            
            elif choice == '7':
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 7.")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_config_manager()