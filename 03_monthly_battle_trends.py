#!/usr/bin/env python3
"""
Step 3: Track battles per month for each model and analyze trends across categories.
This will show how model usage and category dominance evolves over time.
"""

import pickle
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import sys

class MonthlyBattleTrendAnalyzer:
    def __init__(self):
        # 3-tier classification system (from 02_categorize_models.py)
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
        
        # Data structures for analysis
        self.model_monthly_battles = defaultdict(lambda: defaultdict(int))  # model -> month -> count
        self.category_monthly_battles = defaultdict(lambda: defaultdict(int))  # category -> month -> count
        self.monthly_totals = defaultdict(int)  # month -> total battles
        
    def get_model_category(self, model_name):
        """Get the category for a given model."""
        if model_name in self.proprietary_models:
            return 'Proprietary'
        elif model_name in self.open_weight_models:
            return 'Open-Weight'
        elif model_name in self.open_source_models:
            return 'Open-Source'
        else:
            return 'Unknown'
    
    def get_month_key(self, timestamp):
        """Convert timestamp to month key (YYYY-MM format)."""
        if hasattr(timestamp, 'year') and hasattr(timestamp, 'month'):
            return f"{timestamp.year}-{timestamp.month:02d}"
        return None
    
    def analyze_monthly_trends(self):
        """Analyze monthly battle trends for each model and category."""
        print("="*80)
        print("ANALYZING MONTHLY BATTLE TRENDS")
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
            
            print("\n🔍 Processing battles by month...")
            
            valid_battles = 0
            invalid_timestamps = 0
            
            for i, ex in enumerate(train_data):
                if i % 20000 == 0:
                    print(f"Processing {i}/{len(train_data)} battles...")
                
                model_a = ex.get('model_a', '')
                model_b = ex.get('model_b', '')
                timestamp = ex.get('timestamp')
                winner = ex.get('winner', '')
                
                if not model_a or not model_b or winner not in ['model_a', 'model_b', 'tie', 'both_bad']:
                    continue
                
                # Get month from timestamp
                month_key = self.get_month_key(timestamp)
                if not month_key:
                    invalid_timestamps += 1
                    continue
                
                valid_battles += 1
                
                # Count battles for each model
                self.model_monthly_battles[model_a][month_key] += 1
                self.model_monthly_battles[model_b][month_key] += 1
                
                # Count battles for each category
                cat_a = self.get_model_category(model_a)
                cat_b = self.get_model_category(model_b)
                
                self.category_monthly_battles[cat_a][month_key] += 1
                self.category_monthly_battles[cat_b][month_key] += 1
                
                # Total battles per month
                self.monthly_totals[month_key] += 1
            
            print(f"✅ Analysis complete!")
            print(f"Valid battles processed: {valid_battles:,}")
            print(f"Invalid timestamps: {invalid_timestamps:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False
    
    def display_monthly_model_trends(self):
        """Display monthly battle trends for individual models."""
        print("\n" + "="*100)
        print("MONTHLY BATTLE TRENDS BY MODEL")
        print("="*100)
        
        # Get all months and sort them
        all_months = sorted(set().union(*[battles.keys() for battles in self.model_monthly_battles.values()]))
        
        print(f"\nMonths available: {', '.join(all_months)}")
        
        # Display top models by total battles
        model_totals = {}
        for model, monthly_battles in self.model_monthly_battles.items():
            model_totals[model] = sum(monthly_battles.values())
        
        # Sort models by total battles (ALL models)
        all_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 ALL {len(all_models)} MODELS - MONTHLY BREAKDOWN:")
        print(f"{'Model':<50} {'Category':<12} {'Total':<8} {' | '.join(f'{month:<8}' for month in all_months)}")
        print("-" * (50 + 12 + 8 + len(all_months) * 11))
        
        for model, total_battles in all_models:
            category = self.get_model_category(model)
            monthly_data = self.model_monthly_battles[model]
            
            monthly_counts = []
            for month in all_months:
                count = monthly_data.get(month, 0)
                monthly_counts.append(f"{count:<8}")
            
            print(f"{model:<50} {category:<12} {total_battles:<8,} {' | '.join(monthly_counts)}")
    
    def display_category_trends(self):
        """Display monthly battle trends by category."""
        print("\n" + "="*80)
        print("MONTHLY BATTLE TRENDS BY CATEGORY")
        print("="*80)
        
        # Get all months and sort them
        all_months = sorted(set().union(*[battles.keys() for battles in self.category_monthly_battles.values()]))
        
        categories = ['Proprietary', 'Open-Source', 'Open-Weight']
        
        print(f"\n📊 CATEGORY BATTLE COUNTS BY MONTH:")
        print(f"{'Category':<15} {'Total':<10} {' | '.join(f'{month:<10}' for month in all_months)}")
        print("-" * (15 + 10 + len(all_months) * 13))
        
        category_totals = {}
        
        for category in categories:
            monthly_data = self.category_monthly_battles[category]
            total_battles = sum(monthly_data.values())
            category_totals[category] = total_battles
            
            monthly_counts = []
            for month in all_months:
                count = monthly_data.get(month, 0)
                monthly_counts.append(f"{count:<10,}")
            
            print(f"{category:<15} {total_battles:<10,} {' | '.join(monthly_counts)}")
        
        # Calculate percentages
        print(f"\n📊 CATEGORY PERCENTAGES BY MONTH:")
        print(f"{'Category':<15} {'Avg %':<8} {' | '.join(f'{month:<10}' for month in all_months)}")
        print("-" * (15 + 8 + len(all_months) * 13))
        
        for category in categories:
            monthly_data = self.category_monthly_battles[category]
            total_battles = sum(monthly_data.values())
            
            monthly_percentages = []
            total_percentage = 0
            valid_months = 0
            
            for month in all_months:
                count = monthly_data.get(month, 0)
                month_total = self.monthly_totals.get(month, 1)  # Avoid division by zero
                percentage = (count / month_total * 100) if month_total > 0 else 0
                monthly_percentages.append(f"{percentage:<10.1f}%")
                
                if month_total > 0:
                    total_percentage += percentage
                    valid_months += 1
            
            avg_percentage = total_percentage / valid_months if valid_months > 0 else 0
            print(f"{category:<15} {avg_percentage:<8.1f}% {' | '.join(monthly_percentages)}")
    
    def analyze_category_growth(self):
        """Analyze growth trends for each category."""
        print("\n" + "="*80)
        print("CATEGORY GROWTH ANALYSIS")
        print("="*80)
        
        all_months = sorted(set().union(*[battles.keys() for battles in self.category_monthly_battles.values()]))
        categories = ['Proprietary', 'Open-Source', 'Open-Weight']
        
        print(f"\n📈 MONTH-OVER-MONTH GROWTH RATES:")
        
        for category in categories:
            print(f"\n{category} Growth:")
            monthly_data = self.category_monthly_battles[category]
            
            prev_count = None
            for month in all_months:
                count = monthly_data.get(month, 0)
                
                if prev_count is not None and prev_count > 0:
                    growth_rate = ((count - prev_count) / prev_count) * 100
                    direction = "📈" if growth_rate > 0 else "📉" if growth_rate < 0 else "➡️"
                    print(f"  {month}: {count:,} battles ({growth_rate:+.1f}%) {direction}")
                else:
                    print(f"  {month}: {count:,} battles (baseline)")
                
                prev_count = count
        
        # Summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        
        for category in categories:
            monthly_data = self.category_monthly_battles[category]
            monthly_counts = [monthly_data.get(month, 0) for month in all_months]
            
            if monthly_counts:
                total = sum(monthly_counts)
                avg = total / len(monthly_counts)
                max_month = all_months[monthly_counts.index(max(monthly_counts))]
                min_month = all_months[monthly_counts.index(min(monthly_counts))]
                
                print(f"\n{category}:")
                print(f"  Total battles: {total:,}")
                print(f"  Average per month: {avg:.0f}")
                print(f"  Peak month: {max_month} ({max(monthly_counts):,} battles)")
                print(f"  Lowest month: {min_month} ({min(monthly_counts):,} battles)")
                
                # Calculate trend (first vs last month)
                if len(monthly_counts) >= 2 and monthly_counts[0] > 0:
                    overall_growth = ((monthly_counts[-1] - monthly_counts[0]) / monthly_counts[0]) * 100
                    trend = "Growing" if overall_growth > 10 else "Declining" if overall_growth < -10 else "Stable"
                    print(f"  Overall trend: {trend} ({overall_growth:+.1f}%)")
    
    def run_analysis(self):
        """Run the complete monthly battle trend analysis."""
        print("="*100)
        print("MONTHLY BATTLE TRENDS ANALYSIS")
        print("Tracking battles per month for each model and analyzing category trends")
        print("="*100)
        
        # Analyze monthly trends
        if not self.analyze_monthly_trends():
            print("Failed to analyze dataset. Exiting.")
            return
        
        # Display results
        self.display_monthly_model_trends()
        self.display_category_trends()
        self.analyze_category_growth()
        
        print(f"\n✅ Monthly battle trends analysis complete!")
        
        # Save output to file
        self.save_output_to_file()
    
    def save_output_to_file(self):
        """Save all the analysis output to a text file."""
        output_file = Path("monthly_battle_trends_analysis.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write the analysis output to file
                f.write("="*100 + "\n")
                f.write("MONTHLY BATTLE TRENDS ANALYSIS - SAVED OUTPUT\n")
                f.write("Tracking battles per month for each model and analyzing category trends\n")
                f.write("="*100 + "\n\n")
                
                # Write model trends
                f.write("MONTHLY BATTLE TRENDS BY MODEL\n")
                f.write("="*80 + "\n\n")
                
                # Get all months and sort them
                all_months = sorted(set().union(*[battles.keys() for battles in self.model_monthly_battles.values()]))
                f.write(f"Months available: {', '.join(all_months)}\n\n")
                
                # Display top models by total battles
                model_totals = {}
                for model, monthly_battles in self.model_monthly_battles.items():
                    model_totals[model] = sum(monthly_battles.values())
                
                # Sort models by total battles (ALL models)
                all_models_save = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
                
                f.write(f"ALL {len(all_models_save)} MODELS - MONTHLY BREAKDOWN:\n")
                f.write(f"{'Model':<50} {'Category':<12} {'Total':<8} {' | '.join(f'{month:<8}' for month in all_months)}\n")
                f.write("-" * (50 + 12 + 8 + len(all_months) * 11) + "\n")
                
                for model, total_battles in all_models_save:
                    category = self.get_model_category(model)
                    monthly_data = self.model_monthly_battles[model]
                    
                    monthly_counts = []
                    for month in all_months:
                        count = monthly_data.get(month, 0)
                        monthly_counts.append(f"{count:<8}")
                    
                    f.write(f"{model:<50} {category:<12} {total_battles:<8,} {' | '.join(monthly_counts)}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("MONTHLY BATTLE TRENDS BY CATEGORY\n")
                f.write("="*80 + "\n\n")
                
                # Write category trends
                categories = ['Proprietary', 'Open-Source', 'Open-Weight']
                
                f.write("CATEGORY BATTLE COUNTS BY MONTH:\n")
                f.write(f"{'Category':<15} {'Total':<10} {' | '.join(f'{month:<10}' for month in all_months)}\n")
                f.write("-" * (15 + 10 + len(all_months) * 13) + "\n")
                
                for category in categories:
                    monthly_data = self.category_monthly_battles[category]
                    total_battles = sum(monthly_data.values())
                    
                    monthly_counts = []
                    for month in all_months:
                        count = monthly_data.get(month, 0)
                        monthly_counts.append(f"{count:<10,}")
                    
                    f.write(f"{category:<15} {total_battles:<10,} {' | '.join(monthly_counts)}\n")
                
                # Write percentages
                f.write("\nCATEGORY PERCENTAGES BY MONTH:\n")
                f.write(f"{'Category':<15} {'Avg %':<8} {' | '.join(f'{month:<10}' for month in all_months)}\n")
                f.write("-" * (15 + 8 + len(all_months) * 13) + "\n")
                
                for category in categories:
                    monthly_data = self.category_monthly_battles[category]
                    total_battles = sum(monthly_data.values())
                    
                    monthly_percentages = []
                    total_percentage = 0
                    valid_months = 0
                    
                    for month in all_months:
                        count = monthly_data.get(month, 0)
                        month_total = self.monthly_totals.get(month, 1)
                        percentage = (count / month_total * 100) if month_total > 0 else 0
                        monthly_percentages.append(f"{percentage:<10.1f}%")
                        
                        if month_total > 0:
                            total_percentage += percentage
                            valid_months += 1
                    
                    avg_percentage = total_percentage / valid_months if valid_months > 0 else 0
                    f.write(f"{category:<15} {avg_percentage:<8.1f}% {' | '.join(monthly_percentages)}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("CATEGORY GROWTH ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                # Write growth analysis
                f.write("MONTH-OVER-MONTH GROWTH RATES:\n\n")
                
                for category in categories:
                    f.write(f"{category} Growth:\n")
                    monthly_data = self.category_monthly_battles[category]
                    
                    prev_count = None
                    for month in all_months:
                        count = monthly_data.get(month, 0)
                        
                        if prev_count is not None and prev_count > 0:
                            growth_rate = ((count - prev_count) / prev_count) * 100
                            direction = "📈" if growth_rate > 0 else "📉" if growth_rate < 0 else "➡️"
                            f.write(f"  {month}: {count:,} battles ({growth_rate:+.1f}%) {direction}\n")
                        else:
                            f.write(f"  {month}: {count:,} battles (baseline)\n")
                        
                        prev_count = count
                    f.write("\n")
                
                # Write summary statistics
                f.write("SUMMARY STATISTICS:\n\n")
                
                for category in categories:
                    monthly_data = self.category_monthly_battles[category]
                    monthly_counts = [monthly_data.get(month, 0) for month in all_months]
                    
                    if monthly_counts:
                        total = sum(monthly_counts)
                        avg = total / len(monthly_counts)
                        max_month = all_months[monthly_counts.index(max(monthly_counts))]
                        min_month = all_months[monthly_counts.index(min(monthly_counts))]
                        
                        f.write(f"{category}:\n")
                        f.write(f"  Total battles: {total:,}\n")
                        f.write(f"  Average per month: {avg:.0f}\n")
                        f.write(f"  Peak month: {max_month} ({max(monthly_counts):,} battles)\n")
                        f.write(f"  Lowest month: {min_month} ({min(monthly_counts):,} battles)\n")
                        
                        if len(monthly_counts) >= 2 and monthly_counts[0] > 0:
                            overall_growth = ((monthly_counts[-1] - monthly_counts[0]) / monthly_counts[0]) * 100
                            trend = "Growing" if overall_growth > 10 else "Declining" if overall_growth < -10 else "Stable"
                            f.write(f"  Overall trend: {trend} ({overall_growth:+.1f}%)\n")
                        f.write("\n")
                
                f.write("✅ Monthly battle trends analysis complete!\n")
                f.write(f"Output saved to: {output_file}\n")
            
            print(f"\n💾 Analysis output saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error saving output: {e}")

if __name__ == "__main__":
    analyzer = MonthlyBattleTrendAnalyzer()
    analyzer.run_analysis()
