#!/usr/bin/env python3
"""
Quick prediction generator using statistical patterns while new model trains.
This provides immediate functionality without model loading issues.
"""

import pandas as pd
import numpy as np
import random
from collections import Counter
from datetime import datetime

def load_data():
    """Load Mark Six historical data."""
    data_path = "data/raw/Mark_Six.csv"
    col_names = [
        'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
        'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
        'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
        '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
        'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
        'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
        'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
    ]
    
    df = pd.read_csv(data_path, header=None, skiprows=33, names=col_names)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def analyze_patterns(df):
    """Analyze historical patterns for intelligent generation."""
    winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
    
    # Number frequency analysis
    all_numbers = []
    for col in winning_cols:
        all_numbers.extend(df[col].tolist())
    
    number_freq = Counter(all_numbers)
    
    # Recent trends (last 20 draws)
    recent_numbers = []
    for _, row in df.tail(20).iterrows():
        recent_numbers.extend(row[winning_cols].tolist())
    
    recent_freq = Counter(recent_numbers)
    
    # Pair frequency analysis
    pair_freq = Counter()
    for _, row in df.iterrows():
        numbers = sorted(row[winning_cols].tolist())
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                pair_freq[(numbers[i], numbers[j])] += 1
    
    return {
        'number_freq': number_freq,
        'recent_freq': recent_freq,
        'pair_freq': pair_freq,
        'total_draws': len(df)
    }

def generate_intelligent_combination(patterns, diversity_level='balanced'):
    """Generate a combination based on historical patterns."""
    
    number_freq = patterns['number_freq']
    recent_freq = patterns['recent_freq']
    pair_freq = patterns['pair_freq']
    
    # Create weighted probabilities
    weights = {}
    for num in range(1, 50):  # Mark Six numbers 1-49
        # Base frequency weight
        freq_weight = number_freq.get(num, 0) / patterns['total_draws']
        
        # Recent trend weight
        recent_weight = recent_freq.get(num, 0) / 20
        
        # Combine weights based on diversity level
        if diversity_level == 'conservative':
            # Favor frequently drawn numbers
            weights[num] = freq_weight * 0.8 + recent_weight * 0.2
        elif diversity_level == 'creative':
            # More exploration, less exploitation
            weights[num] = freq_weight * 0.3 + recent_weight * 0.1 + 0.6 * (1/49)
        else:  # balanced
            weights[num] = freq_weight * 0.5 + recent_weight * 0.3 + 0.2 * (1/49)
    
    # Generate combination
    combination = []
    available_numbers = list(range(1, 50))
    
    # Select 6 numbers using weighted random selection
    for _ in range(6):
        # Create probability distribution
        probs = [weights.get(num, 0.01) for num in available_numbers]
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        
        # Select number
        selected = np.random.choice(available_numbers, p=probs)
        combination.append(selected)
        available_numbers.remove(selected)
        
        # Update weights to favor numbers that form good pairs
        for num in available_numbers:
            pair_key = tuple(sorted([selected, num]))
            if pair_key in pair_freq:
                weights[num] *= 1.1  # Slight boost for good pairs
    
    return sorted(combination)

def main():
    """Generate intelligent predictions."""
    
    print("🎯 QUICK MARK SIX PREDICTIONS")
    print("=" * 50)
    print("🔄 Using statistical pattern analysis...")
    print("⏳ (While new AI model trains in background)")
    print()
    
    # Load and analyze data
    df = load_data()
    patterns = analyze_patterns(df)
    
    print(f"📊 Analyzed {len(df)} historical draws")
    print(f"🔥 Most frequent numbers: {', '.join(map(str, [k for k, v in patterns['number_freq'].most_common(10)]))}")
    print(f"📈 Recent hot numbers: {', '.join(map(str, [k for k, v in patterns['recent_freq'].most_common(6)]))}")
    print()
    
    # Generate predictions with different strategies
    strategies = [
        ('Conservative', 'conservative'),
        ('Balanced', 'balanced'), 
        ('Creative', 'creative')
    ]
    
    print("🎲 Generated Predictions:")
    print("-" * 30)
    
    all_predictions = []
    for strategy_name, strategy_mode in strategies:
        print(f"\n{strategy_name} Strategy:")
        for i in range(3):
            pred = generate_intelligent_combination(patterns, strategy_mode)
            all_predictions.append(pred)
            print(f"  {len(all_predictions):2d}. {' '.join(f'{n:2d}' for n in pred)}")
    
    print(f"\n📋 Summary - 9 Intelligent Combinations:")
    print("=" * 40)
    for i, pred in enumerate(all_predictions, 1):
        # Calculate confidence score based on patterns
        freq_score = sum(patterns['number_freq'].get(n, 0) for n in pred) / (patterns['total_draws'] * 6)
        recent_score = sum(patterns['recent_freq'].get(n, 0) for n in pred) / (20 * 6)
        confidence = (freq_score + recent_score) * 50  # Scale to percentage
        
        print(f"{i:2d}. {' '.join(f'{n:2d}' for n in pred)} (Confidence: {confidence:.1f}%)")
    
    print(f"\n💡 Strategy Notes:")
    print(f"   • Conservative: Favors historically frequent numbers")
    print(f"   • Balanced: Mix of patterns and exploration") 
    print(f"   • Creative: More diverse, less predictable")
    print(f"\n⚙️  Advanced AI model training in progress...")
    print(f"   Check: tail -f training.log")
    print(f"   When complete, use: python main.py → Option 2")

if __name__ == "__main__":
    # Set random seed based on current time for variety
    random.seed(int(datetime.now().timestamp()))
    np.random.seed(int(datetime.now().timestamp()) % 2**32)
    
    main()