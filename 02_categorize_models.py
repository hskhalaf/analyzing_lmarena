#!/usr/bin/env python3
"""
Step 2: Categorize all models in the Arena dataset into 3 tiers with release dates.
This creates our foundation for temporal analysis.
"""

import pickle
from pathlib import Path
from collections import defaultdict, Counter

class ModelCategorizer:
    def __init__(self):
        # 3-tier classification system
        self.proprietary_models = {
            # Google models
            'gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18', 'gpt-4.1-2025-04-14', 
            'gpt-4.1-mini-2025-04-14', 'chatgpt-4o-latest-20250326',
            
            # Anthropic models
            'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022',
            'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking-32k',
            'claude-opus-4-20250514', 'claude-opus-4-20250514-thinking-16k',
            'claude-sonnet-4-20250514', 'claude-sonnet-4-20250514-thinking-32k',
            
            # Google models
            'gemini-2.0-flash-thinking-exp-01-21', 'gemini-2.0-flash-001',
            'gemini-2.5-pro-preview-03-25', 'gemini-2.5-flash-preview-04-17',
            'gemini-2.5-pro-preview-05-06', 'gemini-2.5-flash', 'gemini-2.5-pro',
            'gemini-2.5-flash-lite-preview-06-17-thinking',
            
            # Amazon models
            'amazon.nova-pro-v1:0', 'amazon-nova-experimental-chat-05-14',
            
            # xAI models
            'grok-3-preview-02-24', 'grok-3-mini-beta', 'grok-3-mini-high', 'grok-4-0709',
            
            # Other proprietary
            'o3-mini', 'o3-2025-04-16', 'o4-mini-2025-04-16', 'command-a-03-2025',
            'hunyuan-turbos-20250416', 'minimax-m1', 'kimi-k2-0711-preview'
        }
        
        self.open_weight_models = {
            'llama-3.3-70b-instruct', 'llama-4-maverick-03-26-experimental',
            'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct',
            'mistral-small-3.1-24b-instruct-2503', 'mistral-medium-2505', 'mistral-small-2506',
            'deepseek-v3-0324', 'deepseek-r1-0528', 'gemma-3-27b-it', 'gemma-3n-e4b-it',
            'qwen-max-2025-01-25', 'qwen3-30b-a3b', 'qwen3-235b-a22b',
            'qwen3-235b-a22b-instruct-2507', 'qwen3-coder-480b-a35b-instruct',
            'magistral-medium-2506', 'qwq-32b'
        }
        
        self.open_source_models = {
            'deepseek-v3-0324', 'deepseek-r1-0528', 'gemma-3-27b-it', 'gemma-3n-e4b-it',
            'llama-3.3-70b-instruct', 'llama-4-maverick-03-26-experimental',
            'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct',
            'mistral-small-3.1-24b-instruct-2503', 'mistral-medium-2505', 'mistral-small-2506',
            'qwen3-30b-a3b', 'qwen3-235b-a22b', 'qwen3-235b-a22b-instruct-2507',
            'qwen3-coder-480b-a35b-instruct', 'magistral-medium-2506', 'qwen3-235b-a22b-no-thinking'
        }
        
        # Release cohorts by month
        self.release_cohorts = {
            'Pre-2025': [
                'llama-3.3-70b-instruct', 'amazon.nova-pro-v1:0', 'gpt-4o-2024-11-20',
                'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'gpt-4o-mini-2024-07-18'
            ],
            '2025-01': [
                'qwen-max-2025-01-25', 'gemini-2.0-flash-thinking-exp-01-21', 'o3-mini'
            ],
            '2025-02': [
                'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking-32k',
                'gemini-2.0-flash-001', 'grok-3-preview-02-24'
            ],
            '2025-03': [
                'command-a-03-2025', 'deepseek-v3-0324', 'gemma-3-27b-it',
                'llama-4-maverick-03-26-experimental', 'mistral-small-3.1-24b-instruct-2503',
                'chatgpt-4o-latest-20250326', 'gemini-2.5-pro-preview-03-25', 'qwq-32b'
            ],
            '2025-04': [
                'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct',
                'qwen3-30b-a3b', 'qwen3-235b-a22b', 'qwen3-235b-a22b-no-thinking', 'gpt-4.1-2025-04-14',
                'gpt-4.1-mini-2025-04-14', 'o3-2025-04-16', 'o4-mini-2025-04-16',
                'gemini-2.5-flash-preview-04-17', 'hunyuan-turbos-20250416'
            ],
            '2025-05': [
                'deepseek-r1-0528', 'mistral-medium-2505', 'amazon-nova-experimental-chat-05-14',
                'claude-opus-4-20250514', 'claude-opus-4-20250514-thinking-16k',
                'claude-sonnet-4-20250514', 'claude-sonnet-4-20250514-thinking-32k',
                'gemini-2.5-pro-preview-05-06', 'grok-3-mini-beta', 'grok-3-mini-high'
            ],
            '2025-06': [
                'gemma-3n-e4b-it', 'magistral-medium-2506', 'mistral-small-2506',
                'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17-thinking',
                'gemini-2.5-pro', 'minimax-m1'
            ],
            '2025-07': [
                'qwen3-235b-a22b-instruct-2507', 'qwen3-coder-480b-a35b-instruct',
                'grok-4-0709', 'kimi-k2-0711-preview'
            ]
        }
        
        # Create reverse lookups
        self.model_to_tier = {}
        self.model_to_cohort = {}
        
        for model in self.proprietary_models:
            self.model_to_tier[model] = 'P'
        for model in self.open_weight_models:
            self.model_to_tier[model] = 'W'
        for model in self.open_source_models:
            self.model_to_tier[model] = 'S'
        
        for cohort, models in self.release_cohorts.items():
            for model in models:
                self.model_to_cohort[model] = cohort
    
    def get_model_tier(self, model_name):
        """Get the tier classification for a model."""
        return self.model_to_tier.get(model_name, 'Unknown')
    
    def get_model_cohort(self, model_name):
        """Get the release cohort for a model."""
        return self.model_to_cohort.get(model_name, 'Unknown')
    
    def analyze_dataset_models(self):
        """Analyze all models in the dataset and categorize them."""
        print("="*80)
        print("ANALYZING AND CATEGORIZING MODELS IN ARENA DATASET")
        print("="*80)
        
        # Load cached dataset
        cache_file = Path("data/arena_dataset.pkl")
        if not cache_file.exists():
            print("❌ No cached dataset found. Please run 01_explore_data.py first.")
            return None
        
        try:
            print("Loading cached dataset...")
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
            
            train_data = dataset['train']
            print(f"✅ Dataset loaded: {len(train_data):,} examples")
            
            # Collect all unique models
            all_models = set()
            model_battle_counts = Counter()
            
            print("\n🔍 Analyzing battles to find all models...")
            for i, ex in enumerate(train_data):
                if i % 20000 == 0:
                    print(f"Processing {i}/{len(train_data)} battles...")
                
                model_a = ex.get('model_a', '')
                model_b = ex.get('model_b', '')
                
                if model_a:
                    all_models.add(model_a)
                    model_battle_counts[model_a] += 1
                if model_b:
                    all_models.add(model_b)
                    model_battle_counts[model_b] += 1
            
            print(f"✅ Found {len(all_models)} unique models")
            
            # Categorize all models
            categorized_models = {
                'P': [],  # Proprietary
                'W': [],  # Open-weight
                'S': [],  # Open-source
                'Unknown': []
            }
            
            for model in sorted(all_models):
                tier = self.get_model_tier(model)
                cohort = self.get_model_cohort(model)
                battle_count = model_battle_counts[model]
                
                categorized_models[tier].append({
                    'name': model,
                    'cohort': cohort,
                    'battles': battle_count
                })
            
            # Display results
            self.display_categorization_results(categorized_models, model_battle_counts)
            
            return categorized_models
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return None
    
    def display_categorization_results(self, categorized_models, model_battle_counts):
        """Display the categorization results."""
        print("\n" + "="*80)
        print("MODEL CATEGORIZATION RESULTS")
        print("="*80)
        
        total_models = sum(len(models) for models in categorized_models.values())
        
        print(f"\n📊 OVERVIEW:")
        print(f"Total models found: {total_models}")
        print(f"Proprietary (P): {len(categorized_models['P'])}")
        print(f"Open-weight (W): {len(categorized_models['W'])}")
        print(f"Open-source (S): {len(categorized_models['S'])}")
        print(f"Unknown: {len(categorized_models['Unknown'])}")
        
        # Display by tier
        for tier, tier_name in [('P', 'PROPRIETARY'), ('W', 'OPEN-WEIGHT'), ('S', 'OPEN-SOURCE'), ('Unknown', 'UNKNOWN')]:
            models = categorized_models[tier]
            if not models:
                continue
                
            print(f"\n{'='*60}")
            print(f"{tier_name} MODELS ({len(models)} total)")
            print(f"{'='*60}")
            
            # Sort by battle count
            models.sort(key=lambda x: x['battles'], reverse=True)
            
            for model_info in models:
                name = model_info['name']
                cohort = model_info['cohort']
                battles = model_info['battles']
                
                print(f"{name:<50} | {cohort:<12} | {battles:>6,} battles")
        
        # Display by cohort
        print(f"\n{'='*80}")
        print("MODELS BY RELEASE COHORT")
        print(f"{'='*80}")
        
        for cohort in self.release_cohorts.keys():
            cohort_models = []
            for tier in ['P', 'W', 'S']:
                for model_info in categorized_models[tier]:
                    if model_info['cohort'] == cohort:
                        cohort_models.append(model_info)
            
            if cohort_models:
                print(f"\n{cohort} ({len(cohort_models)} models):")
                cohort_models.sort(key=lambda x: x['battles'], reverse=True)
                for model_info in cohort_models:
                    name = model_info['name']
                    tier = self.get_model_tier(name)
                    battles = model_info['battles']
                    print(f"  {tier} | {name:<45} | {battles:>6,} battles")
        
        # Data quality check
        print(f"\n{'='*80}")
        print("DATA QUALITY CHECK")
        print(f"{'='*80}")
        
        unknown_models = categorized_models['Unknown']
        if unknown_models:
            print(f"⚠️  WARNING: {len(unknown_models)} models could not be categorized!")
            print("These models may need to be added to the classification system:")
            for model_info in unknown_models:
                print(f"  - {model_info['name']} ({model_info['battles']:,} battles)")
        else:
            print("✅ All models successfully categorized!")
        
        # Coverage statistics
        total_battles = sum(model_battle_counts.values())
        categorized_battles = sum(
            model_battle_counts[model_info['name']] 
            for tier in ['P', 'W', 'S'] 
            for model_info in categorized_models[tier]
        )
        
        coverage = categorized_battles / total_battles * 100 if total_battles > 0 else 0
        print(f"\n📊 Coverage: {coverage:.1f}% of all battles involve categorized models")

def main():
    categorizer = ModelCategorizer()
    categorized_models = categorizer.analyze_dataset_models()
    
    if categorized_models:
        print(f"\n✅ Model categorization complete!")
        print(f"Results saved in memory for next analysis steps.")

if __name__ == "__main__":
    main()
