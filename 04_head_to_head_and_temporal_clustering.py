#!/usr/bin/env python3
"""
Step 4: Analyze head-to-head performance between categories across months
and temporal clustering patterns (which models battle each other by release date).
"""

import pickle
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class HeadToHeadAndTemporalAnalyzer:
    def __init__(self):
        # 3-tier classification system
        self.proprietary_models = {
            'gpt-4o-2024-11-20', 'gpt-4o-mini-2024-07-18', 'gpt-4.1-2025-04-14', 
            'gpt-4.1-mini-2025-04-14', 'chatgpt-4o-latest-20250326',
            'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022',
            'claude-3-7-sonnet-20250219', 'claude-3-7-sonnet-20250219-thinking-32k',
            'claude-opus-4-20250514', 'claude-opus-4-20250514-thinking-16k',
            'claude-sonnet-4-20250514', 'claude-sonnet-4-20250514-thinking-32k',
            'gemini-2.0-flash-thinking-exp-01-21', 'gemini-2.0-flash-001',
            'gemini-2.5-pro-preview-03-25', 'gemini-2.5-flash-preview-04-17',
            'gemini-2.5-pro-preview-05-06', 'gemini-2.5-flash', 'gemini-2.5-pro',
            'gemini-2.5-flash-lite-preview-06-17-thinking',
            'amazon.nova-pro-v1:0', 'amazon-nova-experimental-chat-05-14',
            'grok-3-preview-02-24', 'grok-3-mini-beta', 'grok-3-mini-high', 'grok-4-0709',
            'o3-mini', 'o3-2025-04-16', 'o4-mini-2025-04-16', 'command-a-03-2025',
            'hunyuan-turbos-20250416', 'minimax-m1', 'kimi-k2-0711-preview'
        }
        
        self.open_weight_models = {
            'qwen-max-2025-01-25', 'qwq-32b'
        }
        
        self.open_source_models = {
            'deepseek-v3-0324', 'deepseek-r1-0528', 'gemma-3-27b-it', 'gemma-3n-e4b-it',
            'llama-3.3-70b-instruct', 'llama-4-maverick-03-26-experimental',
            'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct',
            'mistral-small-3.1-24b-instruct-2503', 'mistral-medium-2505', 'mistral-small-2506',
            'qwen3-30b-a3b', 'qwen3-235b-a22b', 'qwen3-235b-a22b-instruct-2507',
            'qwen3-coder-480b-a35b-instruct', 'magistral-medium-2506', 'qwen3-235b-a22b-no-thinking'
        }
        
        # Release cohorts
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
        self.model_to_category = {}
        self.model_to_cohort = {}
        
        for model in self.proprietary_models:
            self.model_to_category[model] = 'Proprietary'
        for model in self.open_weight_models:
            self.model_to_category[model] = 'Open-Weight'
        for model in self.open_source_models:
            self.model_to_category[model] = 'Open-Source'
        
        for cohort, models in self.release_cohorts.items():
            for model in models:
                self.model_to_cohort[model] = cohort
        
        # Data structures for analysis
        # Head-to-head: month -> category1 -> category2 -> {wins, total}
        self.monthly_head_to_head = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0})))
        
        # Temporal clustering: cohort1 -> cohort2 -> battle_count
        self.cohort_battle_matrix = defaultdict(lambda: defaultdict(int))
        
        # Cohort ordering for distance calculation
        self.cohort_order = ['Pre-2025', '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06', '2025-07']
        self.cohort_to_index = {cohort: i for i, cohort in enumerate(self.cohort_order)}
    
    def get_model_category(self, model_name):
        """Get the category for a given model."""
        return self.model_to_category.get(model_name, 'Unknown')
    
    def get_model_cohort(self, model_name):
        """Get the release cohort for a model."""
        return self.model_to_cohort.get(model_name, 'Unknown')
    
    def get_month_key(self, timestamp):
        """Convert timestamp to month key (YYYY-MM format)."""
        if hasattr(timestamp, 'year') and hasattr(timestamp, 'month'):
            return f"{timestamp.year}-{timestamp.month:02d}"
        return None
    
    def calculate_temporal_distance(self, cohort1, cohort2):
        """Calculate temporal distance between two cohorts."""
        if cohort1 == 'Unknown' or cohort2 == 'Unknown':
            return -1
        
        idx1 = self.cohort_to_index.get(cohort1, -1)
        idx2 = self.cohort_to_index.get(cohort2, -1)
        
        if idx1 == -1 or idx2 == -1:
            return -1
        
        return abs(idx1 - idx2)
    
    def analyze_battles(self):
        """Analyze battles for both head-to-head and temporal clustering."""
        print("="*80)
        print("ANALYZING HEAD-TO-HEAD PERFORMANCE AND TEMPORAL CLUSTERING")
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
            
            print("\n🔍 Processing battles for head-to-head and temporal analysis...")
            
            valid_battles = 0
            invalid_battles = 0
            
            for i, ex in enumerate(train_data):
                if i % 20000 == 0:
                    print(f"Processing {i}/{len(train_data)} battles...")
                
                model_a = ex.get('model_a', '')
                model_b = ex.get('model_b', '')
                winner = ex.get('winner', '')
                timestamp = ex.get('timestamp')
                
                if not model_a or not model_b or winner not in ['model_a', 'model_b', 'tie', 'both_bad']:
                    invalid_battles += 1
                    continue
                
                # Get month, categories, and cohorts
                month_key = self.get_month_key(timestamp)
                cat_a = self.get_model_category(model_a)
                cat_b = self.get_model_category(model_b)
                cohort_a = self.get_model_cohort(model_a)
                cohort_b = self.get_model_cohort(model_b)
                
                if not month_key or cat_a == 'Unknown' or cat_b == 'Unknown':
                    invalid_battles += 1
                    continue
                
                valid_battles += 1
                
                # Head-to-head analysis (only for cross-category battles)
                if cat_a != cat_b and winner in ['model_a', 'model_b']:
                    # Sort categories for consistent indexing
                    cat1, cat2 = sorted([cat_a, cat_b])
                    
                    # Determine winner category
                    if (cat_a == cat1 and winner == 'model_a') or (cat_b == cat1 and winner == 'model_b'):
                        winner_cat = cat1
                    else:
                        winner_cat = cat2
                    
                    # Record the battle
                    self.monthly_head_to_head[month_key][cat1][cat2]['total'] += 1
                    if winner_cat == cat1:
                        self.monthly_head_to_head[month_key][cat1][cat2]['wins'] += 1
                
                # Temporal clustering analysis
                if cohort_a != 'Unknown' and cohort_b != 'Unknown':
                    # Sort cohorts for consistent indexing
                    cohort1, cohort2 = sorted([cohort_a, cohort_b])
                    self.cohort_battle_matrix[cohort1][cohort2] += 1
            
            print(f"✅ Analysis complete!")
            print(f"Valid battles processed: {valid_battles:,}")
            print(f"Invalid battles skipped: {invalid_battles:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False
    
    def display_head_to_head_analysis(self):
        """Display head-to-head performance analysis across months."""
        print("\n" + "="*80)
        print("HEAD-TO-HEAD PERFORMANCE ANALYSIS BY MONTH")
        print("="*80)
        
        # Get all months and sort them
        all_months = sorted(self.monthly_head_to_head.keys())
        
        print(f"\nMonths analyzed: {', '.join(all_months)}")
        
        # Category pairs to analyze
        category_pairs = [
            ('Open-Source', 'Proprietary'),
            ('Open-Source', 'Open-Weight'),
            ('Open-Weight', 'Proprietary')
        ]
        
        for cat1, cat2 in category_pairs:
            print(f"\n{'='*60}")
            print(f"{cat1.upper()} vs {cat2.upper()} PERFORMANCE")
            print(f"{'='*60}")
            
            print(f"{'Month':<10} {'Battles':<8} {cat1+' Wins':<15} {'Win Rate':<10} {'Trend':<10}")
            print("-" * 60)
            
            prev_win_rate = None
            
            for month in all_months:
                monthly_data = self.monthly_head_to_head[month]
                battle_data = monthly_data[cat1][cat2]
                
                total_battles = battle_data['total']
                cat1_wins = battle_data['wins']
                
                if total_battles > 0:
                    win_rate = (cat1_wins / total_battles) * 100
                    
                    # Determine trend
                    if prev_win_rate is not None:
                        if win_rate > prev_win_rate + 2:
                            trend = "📈"
                        elif win_rate < prev_win_rate - 2:
                            trend = "📉"
                        else:
                            trend = "➡️"
                    else:
                        trend = "🆕"
                    
                    print(f"{month:<10} {total_battles:<8} {cat1_wins:<15} {win_rate:<10.1f}% {trend}")
                    prev_win_rate = win_rate
                else:
                    print(f"{month:<10} {total_battles:<8} {cat1_wins:<15} {'N/A':<10} {'❌'}")
        
        # Overall summary
        print(f"\n{'='*80}")
        print("OVERALL HEAD-TO-HEAD SUMMARY")
        print(f"{'='*80}")
        
        for cat1, cat2 in category_pairs:
            total_battles = 0
            total_cat1_wins = 0
            
            for month in all_months:
                battle_data = self.monthly_head_to_head[month][cat1][cat2]
                total_battles += battle_data['total']
                total_cat1_wins += battle_data['wins']
            
            if total_battles > 0:
                overall_win_rate = (total_cat1_wins / total_battles) * 100
                print(f"\n{cat1} vs {cat2}:")
                print(f"  Total battles: {total_battles:,}")
                print(f"  {cat1} wins: {total_cat1_wins:,} ({overall_win_rate:.1f}%)")
                print(f"  {cat2} wins: {total_battles - total_cat1_wins:,} ({100 - overall_win_rate:.1f}%)")
                
                if overall_win_rate > 55:
                    print(f"  Result: 🏆 {cat1} dominates")
                elif overall_win_rate < 45:
                    print(f"  Result: 🏆 {cat2} dominates")
                else:
                    print(f"  Result: ⚖️ Competitive balance")
    
    def display_temporal_clustering_analysis(self):
        """Display temporal clustering analysis."""
        print("\n" + "="*80)
        print("TEMPORAL CLUSTERING ANALYSIS")
        print("="*80)
        
        print(f"\n📊 BATTLE MATRIX BY RELEASE COHORT:")
        print(f"{'Cohort':<12}", end="")
        for cohort in self.cohort_order:
            print(f"{cohort:<10}", end="")
        print()
        
        print("-" * (12 + 10 * len(self.cohort_order)))
        
        # Create the grid matrix
        grid_matrix = []
        for cohort1 in self.cohort_order:
            row = []
            print(f"{cohort1:<12}", end="")
            for cohort2 in self.cohort_order:
                # For symmetric matrix, use sorted cohorts
                c1, c2 = sorted([cohort1, cohort2])
                count = self.cohort_battle_matrix[c1][c2]
                row.append(count)
                
                if count == 0:
                    print(f"{'.':<10}", end="")
                else:
                    print(f"{count:<10}", end="")
            print()
            grid_matrix.append(row)
        
        # Analyze diagonal dominance
        print(f"\n📈 TEMPORAL DISTANCE ANALYSIS:")
        
        distance_battles = defaultdict(int)
        
        for cohort1 in self.cohort_order:
            for cohort2 in self.cohort_order:
                c1, c2 = sorted([cohort1, cohort2])
                count = self.cohort_battle_matrix[c1][c2]
                
                distance = self.calculate_temporal_distance(cohort1, cohort2)
                if distance >= 0:
                    distance_battles[distance] += count
        
        total_battles = sum(distance_battles.values())
        
        print(f"{'Distance':<10} {'Battles':<10} {'Percentage':<12} {'Description':<30}")
        print("-" * 72)
        
        for distance in sorted(distance_battles.keys()):
            count = distance_battles[distance]
            percentage = (count / total_battles * 100) if total_battles > 0 else 0
            
            if distance == 0:
                description = "Same cohort (diagonal)"
            elif distance == 1:
                description = "Adjacent cohorts"
            elif distance <= 2:
                description = "Close cohorts"
            else:
                description = "Distant cohorts"
            
            print(f"{distance:<10} {count:<10,} {percentage:<12.1f}% {description:<30}")
        
        # Analyze clustering strength
        print(f"\n🔍 CLUSTERING ANALYSIS:")
        
        same_cohort_battles = distance_battles.get(0, 0)
        adjacent_battles = distance_battles.get(1, 0)
        distant_battles = sum(distance_battles.get(d, 0) for d in range(2, 8))
        
        if total_battles > 0:
            same_cohort_pct = (same_cohort_battles / total_battles) * 100
            adjacent_pct = (adjacent_battles / total_battles) * 100
            distant_pct = (distant_battles / total_battles) * 100
            
            print(f"Same cohort battles: {same_cohort_battles:,} ({same_cohort_pct:.1f}%)")
            print(f"Adjacent cohort battles: {adjacent_battles:,} ({adjacent_pct:.1f}%)")
            print(f"Distant cohort battles: {distant_battles:,} ({distant_pct:.1f}%)")
            
            # Determine clustering strength
            if same_cohort_pct > 40:
                clustering_strength = "🔴 EXTREME temporal clustering"
            elif same_cohort_pct > 25:
                clustering_strength = "🟠 STRONG temporal clustering"
            elif same_cohort_pct > 15:
                clustering_strength = "🟡 MODERATE temporal clustering"
            else:
                clustering_strength = "🟢 MINIMAL temporal clustering"
            
            print(f"\nClustering assessment: {clustering_strength}")
            
            if same_cohort_pct > 25:
                print("⚠️  This suggests potential matchmaking bias!")
                print("   Models are more likely to battle others from their release cohort.")
            else:
                print("✅ Matchmaking appears relatively unbiased across release dates.")
    
    def save_analysis_to_file(self):
        """Save the complete analysis to a text file."""
        output_file = Path("head_to_head_and_temporal_analysis.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("HEAD-TO-HEAD PERFORMANCE AND TEMPORAL CLUSTERING ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                # Write head-to-head analysis
                f.write("HEAD-TO-HEAD PERFORMANCE ANALYSIS BY MONTH\n")
                f.write("="*60 + "\n\n")
                
                all_months = sorted(self.monthly_head_to_head.keys())
                f.write(f"Months analyzed: {', '.join(all_months)}\n\n")
                
                category_pairs = [
                    ('Open-Source', 'Proprietary'),
                    ('Open-Source', 'Open-Weight'),
                    ('Open-Weight', 'Proprietary')
                ]
                
                for cat1, cat2 in category_pairs:
                    f.write(f"{cat1.upper()} vs {cat2.upper()} PERFORMANCE\n")
                    f.write("="*50 + "\n")
                    
                    f.write(f"{'Month':<10} {'Battles':<8} {cat1+' Wins':<15} {'Win Rate':<10} {'Trend':<10}\n")
                    f.write("-" * 60 + "\n")
                    
                    prev_win_rate = None
                    
                    for month in all_months:
                        monthly_data = self.monthly_head_to_head[month]
                        battle_data = monthly_data[cat1][cat2]
                        
                        total_battles = battle_data['total']
                        cat1_wins = battle_data['wins']
                        
                        if total_battles > 0:
                            win_rate = (cat1_wins / total_battles) * 100
                            
                            if prev_win_rate is not None:
                                if win_rate > prev_win_rate + 2:
                                    trend = "📈"
                                elif win_rate < prev_win_rate - 2:
                                    trend = "📉"
                                else:
                                    trend = "➡️"
                            else:
                                trend = "🆕"
                            
                            f.write(f"{month:<10} {total_battles:<8} {cat1_wins:<15} {win_rate:<10.1f}% {trend}\n")
                            prev_win_rate = win_rate
                        else:
                            f.write(f"{month:<10} {total_battles:<8} {cat1_wins:<15} {'N/A':<10} {'❌'}\n")
                    
                    f.write("\n")
                
                # Write overall summary
                f.write("OVERALL HEAD-TO-HEAD SUMMARY\n")
                f.write("="*40 + "\n\n")
                
                for cat1, cat2 in category_pairs:
                    total_battles = 0
                    total_cat1_wins = 0
                    
                    for month in all_months:
                        battle_data = self.monthly_head_to_head[month][cat1][cat2]
                        total_battles += battle_data['total']
                        total_cat1_wins += battle_data['wins']
                    
                    if total_battles > 0:
                        overall_win_rate = (total_cat1_wins / total_battles) * 100
                        f.write(f"{cat1} vs {cat2}:\n")
                        f.write(f"  Total battles: {total_battles:,}\n")
                        f.write(f"  {cat1} wins: {total_cat1_wins:,} ({overall_win_rate:.1f}%)\n")
                        f.write(f"  {cat2} wins: {total_battles - total_cat1_wins:,} ({100 - overall_win_rate:.1f}%)\n")
                        
                        if overall_win_rate > 55:
                            f.write(f"  Result: 🏆 {cat1} dominates\n")
                        elif overall_win_rate < 45:
                            f.write(f"  Result: 🏆 {cat2} dominates\n")
                        else:
                            f.write(f"  Result: ⚖️ Competitive balance\n")
                        f.write("\n")
                
                # Write temporal clustering analysis
                f.write("\n" + "="*80 + "\n")
                f.write("TEMPORAL CLUSTERING ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                f.write("BATTLE MATRIX BY RELEASE COHORT:\n")
                f.write(f"{'Cohort':<12}")
                for cohort in self.cohort_order:
                    f.write(f"{cohort:<10}")
                f.write("\n")
                
                f.write("-" * (12 + 10 * len(self.cohort_order)) + "\n")
                
                for cohort1 in self.cohort_order:
                    f.write(f"{cohort1:<12}")
                    for cohort2 in self.cohort_order:
                        c1, c2 = sorted([cohort1, cohort2])
                        count = self.cohort_battle_matrix[c1][c2]
                        
                        if count == 0:
                            f.write(f"{'.':<10}")
                        else:
                            f.write(f"{count:<10}")
                    f.write("\n")
                
                # Write distance analysis
                f.write("\nTEMPORAL DISTANCE ANALYSIS:\n")
                
                distance_battles = defaultdict(int)
                
                for cohort1 in self.cohort_order:
                    for cohort2 in self.cohort_order:
                        c1, c2 = sorted([cohort1, cohort2])
                        count = self.cohort_battle_matrix[c1][c2]
                        
                        distance = self.calculate_temporal_distance(cohort1, cohort2)
                        if distance >= 0:
                            distance_battles[distance] += count
                
                total_battles = sum(distance_battles.values())
                
                f.write(f"{'Distance':<10} {'Battles':<10} {'Percentage':<12} {'Description':<30}\n")
                f.write("-" * 72 + "\n")
                
                for distance in sorted(distance_battles.keys()):
                    count = distance_battles[distance]
                    percentage = (count / total_battles * 100) if total_battles > 0 else 0
                    
                    if distance == 0:
                        description = "Same cohort (diagonal)"
                    elif distance == 1:
                        description = "Adjacent cohorts"
                    elif distance <= 2:
                        description = "Close cohorts"
                    else:
                        description = "Distant cohorts"
                    
                    f.write(f"{distance:<10} {count:<10,} {percentage:<12.1f}% {description:<30}\n")
                
                # Write clustering assessment
                same_cohort_battles = distance_battles.get(0, 0)
                if total_battles > 0:
                    same_cohort_pct = (same_cohort_battles / total_battles) * 100
                    
                    f.write(f"\nCLUSTERING ASSESSMENT:\n")
                    f.write(f"Same cohort battles: {same_cohort_battles:,} ({same_cohort_pct:.1f}%)\n")
                    
                    if same_cohort_pct > 40:
                        f.write("🔴 EXTREME temporal clustering detected!\n")
                        f.write("⚠️  Strong evidence of matchmaking bias.\n")
                    elif same_cohort_pct > 25:
                        f.write("🟠 STRONG temporal clustering detected!\n")
                        f.write("⚠️  Moderate evidence of matchmaking bias.\n")
                    elif same_cohort_pct > 15:
                        f.write("🟡 MODERATE temporal clustering detected.\n")
                    else:
                        f.write("🟢 MINIMAL temporal clustering - matchmaking appears unbiased.\n")
                
                f.write("\n✅ Head-to-head and temporal clustering analysis complete!\n")
            
            print(f"\n💾 Analysis saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
    
    def run_analysis(self):
        """Run the complete head-to-head and temporal clustering analysis."""
        print("="*100)
        print("HEAD-TO-HEAD PERFORMANCE AND TEMPORAL CLUSTERING ANALYSIS")
        print("Analyzing competitive dynamics between categories and matchmaking patterns")
        print("="*100)
        
        # Analyze battles
        if not self.analyze_battles():
            print("Failed to analyze dataset. Exiting.")
            return
        
        # Display results
        self.display_head_to_head_analysis()
        self.display_temporal_clustering_analysis()
        
        # Save to file
        self.save_analysis_to_file()
        
        print(f"\n✅ Head-to-head and temporal clustering analysis complete!")

if __name__ == "__main__":
    analyzer = HeadToHeadAndTemporalAnalyzer()
    analyzer.run_analysis()
