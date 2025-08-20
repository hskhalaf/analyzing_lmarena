#!/usr/bin/env python3
"""
Step 5: Extract all prompts from the Arena dataset and categorize them using a generative model.
This will create a comprehensive taxonomy of prompt types and their distribution.
"""

import pickle
from pathlib import Path
from collections import defaultdict, Counter
import json
import random
from datetime import datetime

class PromptCategorizer:
    def __init__(self):
        # Fine-grained prompt categories based on common LLM evaluation tasks
        self.prompt_categories = {
            # Reasoning & Problem Solving
            'mathematical_reasoning': 'Math problems, calculations, proofs, quantitative analysis',
            'logical_reasoning': 'Logic puzzles, deductive reasoning, syllogisms, boolean logic',
            'creative_problem_solving': 'Open-ended problems, design challenges, innovation tasks',
            
            # Language & Communication
            'creative_writing': 'Stories, poems, essays, creative content generation',
            'technical_writing': 'Documentation, manuals, technical explanations, procedures',
            'translation': 'Language translation, localization, multilingual tasks',
            'summarization': 'Text summarization, key point extraction, condensation',
            
            # Knowledge & Analysis
            'factual_qa': 'Question answering, trivia, knowledge retrieval',
            'analysis': 'Data analysis, trend identification, pattern recognition',
            'comparison': 'Comparing concepts, products, approaches, methodologies',
            'evaluation': 'Assessing quality, ranking, critique, review',
            
            # Code & Technical
            'coding': 'Programming, debugging, code review, algorithm implementation',
            'system_design': 'Architecture, system planning, technical design',
            'troubleshooting': 'Problem diagnosis, error resolution, technical support',
            
            # Creative & Artistic
            'artistic_creation': 'Art descriptions, creative direction, aesthetic tasks',
            'roleplay': 'Character simulation, scenario enactment, interactive storytelling',
            'brainstorming': 'Idea generation, concept development, creative thinking',
            
            # Professional & Business
            'business_analysis': 'Market research, business strategy, financial analysis',
            'professional_advice': 'Career guidance, professional development, consulting',
            'planning': 'Project planning, scheduling, strategic planning',
            
            # Educational & Learning
            'educational_content': 'Teaching, explanation, learning materials, tutorials',
            'assessment': 'Testing, evaluation, skill assessment, knowledge verification',
            
            # Social & Cultural
            'social_interaction': 'Conversation, social scenarios, interpersonal skills',
            'cultural_analysis': 'Cultural understanding, social commentary, diversity topics',
            
            # Specialized Domains
            'scientific': 'Scientific concepts, research, technical explanations',
            'medical': 'Health-related topics, medical information, wellness advice',
            'legal': 'Legal concepts, policy analysis, regulatory compliance',
            'ethical': 'Moral reasoning, ethical dilemmas, value judgments'
        }
        
        # Data structures
        self.all_prompts = []
        self.prompt_samples = []
        self.category_distribution = defaultdict(int)
        self.prompt_length_stats = defaultdict(list)
        
    def extract_prompts_from_dataset(self):
        """Extract all prompts from the Arena dataset."""
        print("="*80)
        print("EXTRACTING PROMPTS FROM ARENA DATASET")
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
            
            print("\n🔍 Extracting prompts from conversations...")
            
            extracted_prompts = 0
            skipped_prompts = 0
            
            for i, ex in enumerate(train_data):
                if i % 20000 == 0:
                    print(f"Processing {i}/{len(train_data)} examples...")
                
                # Try to extract prompt from conversation structure
                prompt = self.extract_prompt_from_conversation(ex)
                
                if prompt:
                    # Clean and validate prompt
                    cleaned_prompt = self.clean_prompt(prompt)
                    if cleaned_prompt and len(cleaned_prompt.strip()) > 10:
                        self.all_prompts.append(cleaned_prompt)
                        extracted_prompts += 1
                        
                        # Track length statistics
                        word_count = len(cleaned_prompt.split())
                        self.prompt_length_stats['word_count'].append(word_count)
                        self.prompt_length_stats['char_count'].append(len(cleaned_prompt))
                        
                        # Sample prompts for detailed analysis
                        if random.random() < 0.01:  # 1% sample
                            self.prompt_samples.append({
                                'prompt': cleaned_prompt[:200] + "..." if len(cleaned_prompt) > 200 else cleaned_prompt,
                                'word_count': word_count,
                                'char_count': len(cleaned_prompt)
                            })
                    else:
                        skipped_prompts += 1
                else:
                    skipped_prompts += 1
            
            print(f"✅ Prompt extraction complete!")
            print(f"Extracted prompts: {extracted_prompts:,}")
            print(f"Skipped examples: {skipped_prompts:,}")
            print(f"Sample prompts collected: {len(self.prompt_samples):,}")
            
            # Analyze prompt characteristics
            self.analyze_prompt_characteristics()
            
            return True
            
        except Exception as e:
            print(f"❌ Error extracting prompts: {e}")
            return False
    
    def extract_prompt_from_conversation(self, example):
        """Extract the user prompt from conversation structure."""
        try:
            # Try different conversation formats
            if 'conversation_a' in example and example['conversation_a']:
                conv_a = example['conversation_a']
                if isinstance(conv_a, list) and len(conv_a) > 0:
                    first_message = conv_a[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                return content[0]['text']
            
            # Try conversation_b as fallback
            if 'conversation_b' in example and example['conversation_b']:
                conv_b = example['conversation_b']
                if isinstance(conv_b, list) and len(conv_b) > 0:
                    first_message = conv_b[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                return content[0]['text']
            
            # Try full_conversation
            if 'full_conversation' in example and example['full_conversation']:
                full_conv = example['full_conversation']
                if isinstance(full_conv, list) and len(full_conv) > 0:
                    user_msg = full_conv[0].get('user', {})
                    if 'content' in user_msg:
                        content = user_msg['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                return content[0]['text']
            
            return None
            
        except Exception as e:
            return None
    
    def clean_prompt(self, prompt):
        """Clean and normalize prompt text."""
        if not prompt or not isinstance(prompt, str):
            return None
        
        # Basic cleaning
        cleaned = prompt.strip()
        
        # Remove very short prompts (keep this filter)
        if len(cleaned) < 10:
            return None
        
        # Only reject actual URLs, not content with slashes and dots
        # Check if it's actually a URL (starts with http/https or is just a file path)
        if cleaned.startswith('http'):
            return None
        
        # Only reject if it looks like a pure file path (e.g., "/path/to/file.txt")
        # but allow content that happens to contain slashes and dots
        if (cleaned.count('/') > 2 and cleaned.count('.') > 0 and 
            len(cleaned.split()) < 5 and 
            not any(char.isalpha() for char in cleaned)):
            return None
        
        return cleaned
    
    def analyze_prompt_characteristics(self):
        """Analyze basic characteristics of extracted prompts."""
        print("\n" + "="*80)
        print("PROMPT CHARACTERISTICS ANALYSIS")
        print("="*80)
        
        if not self.all_prompts:
            print("No prompts to analyze.")
            return
        
        total_prompts = len(self.all_prompts)
        
        # Length statistics
        word_counts = self.prompt_length_stats['word_count']
        char_counts = self.prompt_length_stats['char_count']
        
        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            median_words = sorted(word_counts)[len(word_counts)//2]
            min_words = min(word_counts)
            max_words = max(word_counts)
            
            print(f"\n📊 WORD COUNT STATISTICS:")
            print(f"Total prompts: {total_prompts:,}")
            print(f"Average words: {avg_words:.1f}")
            print(f"Median words: {median_words}")
            print(f"Range: {min_words} - {max_words} words")
        
        if char_counts:
            avg_chars = sum(char_counts) / len(char_counts)
            median_chars = sorted(char_counts)[len(char_counts)//2]
            min_chars = min(char_counts)
            max_chars = max(char_counts)
            
            print(f"\n📊 CHARACTER COUNT STATISTICS:")
            print(f"Average characters: {avg_chars:.1f}")
            print(f"Median characters: {median_chars}")
            print(f"Range: {min_chars} - {max_chars} characters")
        
        # Show sample prompts
        print(f"\n📝 SAMPLE PROMPTS ({len(self.prompt_samples)} samples):")
        for i, sample in enumerate(self.prompt_samples[:10]):
            print(f"\n--- Sample {i+1} ({sample['word_count']} words) ---")
            print(f"{sample['prompt']}")
    
    def create_categorization_prompt(self, prompt_text):
        """Create a prompt for the generative model to categorize the text."""
        categories_text = "\n".join([f"- {cat}: {desc}" for cat, desc in self.prompt_categories.items()])
        
        categorization_prompt = f"""Please categorize the following prompt into the most relevant categories from the list below. A prompt can belong to multiple categories.

Available categories:
{categories_text}

Prompt to categorize:
"{prompt_text}"

Please respond with ONLY the category names (comma-separated) that apply to this prompt. If none fit well, respond with "other". Do not include explanations or additional text.

Categories:"""

        return categorization_prompt
    
    def save_prompts_for_analysis(self):
        """Save extracted prompts and analysis for external categorization."""
        output_dir = Path("prompt_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save all prompts
        prompts_file = output_dir / "all_prompts.txt"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            for i, prompt in enumerate(self.all_prompts):
                f.write(f"=== PROMPT {i+1} ===\n")
                f.write(f"{prompt}\n\n")
        
        # Save sample prompts with categorization prompts
        samples_file = output_dir / "sample_prompts_with_categorization.txt"
        with open(samples_file, 'w', encoding='utf-8') as f:
            f.write("SAMPLE PROMPTS WITH CATEGORIZATION PROMPTS\n")
            f.write("="*80 + "\n\n")
            
            for i, sample in enumerate(self.prompt_samples[:50]):  # First 50 samples
                f.write(f"=== SAMPLE {i+1} ===\n")
                f.write(f"Original prompt: {sample['prompt']}\n")
                f.write(f"Word count: {sample['word_count']}\n")
                f.write(f"Character count: {sample['char_count']}\n\n")
                
                categorization_prompt = self.create_categorization_prompt(sample['prompt'])
                f.write(f"CATEGORIZATION PROMPT:\n{categorization_prompt}\n")
                f.write("-" * 80 + "\n\n")
        
        # Save category definitions
        categories_file = output_dir / "category_definitions.json"
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(self.prompt_categories, f, indent=2)
        
        # Save prompt statistics
        stats_file = output_dir / "prompt_statistics.json"
        stats_data = {
            'total_prompts': len(self.all_prompts),
            'sample_count': len(self.prompt_samples),
            'length_statistics': {
                'word_count': {
                    'mean': sum(self.prompt_length_stats['word_count']) / len(self.prompt_length_stats['word_count']) if self.prompt_length_stats['word_count'] else 0,
                    'median': sorted(self.prompt_length_stats['word_count'])[len(self.prompt_length_stats['word_count'])//2] if self.prompt_length_stats['word_count'] else 0,
                    'min': min(self.prompt_length_stats['word_count']) if self.prompt_length_stats['word_count'] else 0,
                    'max': max(self.prompt_length_stats['word_count']) if self.prompt_length_stats['word_count'] else 0
                },
                'char_count': {
                    'mean': sum(self.prompt_length_stats['char_count']) / len(self.prompt_length_stats['char_count']) if self.prompt_length_stats['char_count'] else 0,
                    'median': sorted(self.prompt_length_stats['char_count'])[len(self.prompt_length_stats['char_count'])//2] if self.prompt_length_stats['char_count'] else 0,
                    'min': min(self.prompt_length_stats['char_count']) if self.prompt_length_stats['char_count'] else 0,
                    'max': max(self.prompt_length_stats['char_count']) if self.prompt_length_stats['char_count'] else 0
                }
            }
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"\n💾 Prompt analysis files saved to: {output_dir}/")
        print(f"  - all_prompts.txt: {len(self.all_prompts):,} prompts")
        print(f"  - sample_prompts_with_categorization.txt: {len(self.prompt_samples)} samples with categorization prompts")
        print(f"  - category_definitions.json: Category definitions")
        print(f"  - prompt_statistics.json: Statistical analysis")
        
        return output_dir
    
    def run_analysis(self):
        """Run the complete prompt extraction and analysis."""
        print("="*100)
        print("PROMPT EXTRACTION AND CATEGORIZATION ANALYSIS")
        print("Extracting prompts from Arena dataset and preparing for generative model categorization")
        print("="*100)
        
        # Extract prompts
        if not self.extract_prompts_from_dataset():
            print("Failed to extract prompts. Exiting.")
            return
        
        # Save prompts for external analysis
        output_dir = self.save_prompts_for_analysis()
        
        print(f"\n✅ Prompt extraction and analysis complete!")
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use the categorization prompts in {output_dir}/sample_prompts_with_categorization.txt")
        print(f"2. Feed them to Llama 3 or another generative model")
        print(f"3. Collect the category responses")
        print(f"4. Analyze the distribution and patterns")
        print(f"\n💡 RECOMMENDED APPROACH:")
        print(f"- Start with the 50 sample prompts to test categorization quality")
        print(f"- Use a consistent prompt format for reliable results")
        print(f"- Consider batching prompts for efficiency")
        print(f"- Validate categories against the definitions in category_definitions.json")

if __name__ == "__main__":
    categorizer = PromptCategorizer()
    categorizer.run_analysis()
