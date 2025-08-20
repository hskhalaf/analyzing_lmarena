#!/usr/bin/env python3
"""
Step 1: Download Arena dataset from Hugging Face and explore its structure.
This will help us understand the data format before proceeding with analysis.
"""

from datasets import load_dataset
import json
import pickle
from pathlib import Path

def download_and_explore_arena_dataset():
    """Download Arena dataset and explore its structure."""
    print("="*80)
    print("DOWNLOADING AND EXPLORING ARENA DATASET")
    print("="*80)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        print("Loading dataset from Hugging Face...")
        dataset = load_dataset("lmarena-ai/arena-human-preference-140k")
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Save dataset locally
        cache_file = data_dir / "arena_dataset.pkl"
        print(f"\n💾 Saving dataset to {cache_file}...")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Dataset saved to {cache_file}")
        
        # Examine train split (main data)
        train_data = dataset['train']
        print(f"\n📊 Train split: {len(train_data)} examples")
        
        # Get the first example to understand structure
        first_example = train_data[0]
        
        print(f"\n🔍 FIRST EXAMPLE STRUCTURE:")
        print(f"Keys: {list(first_example.keys())}")
        
        print(f"\n📝 SAMPLE DATA:")
        for key, value in first_example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"{key}: {value[:200]}...")
            else:
                print(f"{key}: {value}")
        
        # Show data types
        print(f"\n📋 DATA TYPES:")
        for key, value in first_example.items():
            print(f"{key}: {type(value).__name__}")
        
        # Check for any nested structures
        print(f"\n🔍 CHECKING FOR NESTED STRUCTURES:")
        for key, value in first_example.items():
            if isinstance(value, dict):
                print(f"{key}: {type(value).__name__} with keys: {list(value.keys())}")
            elif isinstance(value, list):
                print(f"{key}: {type(value).__name__} with {len(value)} items")
        
        # Show a few more examples to see variation
        print(f"\n📝 ADDITIONAL EXAMPLES:")
        for i in range(1, min(4, len(train_data))):
            ex = train_data[i]
            print(f"\n--- Example {i+1} ---")
            print(f"model_a: {ex.get('model_a', 'N/A')}")
            print(f"model_b: {ex.get('model_b', 'N/A')}")
            print(f"winner: {ex.get('winner', 'N/A')}")
            if 'prompt' in ex:
                prompt = ex['prompt']
                if isinstance(prompt, str):
                    print(f"prompt: {prompt[:100]}...")
                else:
                    print(f"prompt: {type(prompt).__name__}")
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total examples: {len(train_data):,}")
        
        # Check for missing values
        missing_counts = {}
        for key in first_example.keys():
            missing = 0
            for i in range(min(1000, len(train_data))):
                ex = train_data[i]
                if not ex.get(key):
                    missing += 1
            missing_counts[key] = missing
        
        print(f"\nMissing values (first 1000 examples):")
        for key, count in missing_counts.items():
            percentage = count / 1000 * 100
            print(f"  {key}: {count} ({percentage:.1f}%)")
        
        print(f"\n✅ Data exploration complete!")
        print(f"Dataset saved to: {cache_file}")
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset = download_and_explore_arena_dataset()
