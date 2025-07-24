#!/usr/bin/env python3
"""
Quick inference using the working ultra-quick model.
Provides the same interface as main.py but uses compatible model.
"""

import os
import sys

def main():
    print("🎯 MARK SIX AI PREDICTIONS")
    print("=" * 50)
    print("Using ultra-quick AI model (compatible architecture)")
    print()
    
    # Check if models exist
    if not os.path.exists("models/ultra_quick_model.pth"):
        print("❌ Ultra-quick model not found!")
        print("   Run: python ultra_quick_train.py")
        return
    
    print("Available prediction methods:")
    print("1. AI Predictions (Ultra-Quick Model)")
    print("2. Statistical Pattern Analysis")
    print("3. Exit")
    print()
    
    try:
        choice = input("Choose method (1-3): ").strip()
        
        if choice == "1":
            print("\n🤖 Loading AI model...")
            os.system("python use_ultra_model.py")
        elif choice == "2":
            print("\n📊 Running statistical analysis...")
            os.system("python quick_predict.py")
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except EOFError:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()