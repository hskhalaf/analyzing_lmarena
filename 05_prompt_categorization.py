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
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptCategorizer:
    def __init__(self):
        # CUDA setup
        self.device = self.setup_cuda()
        
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
        
        # Model components for CUDA
        self.tokenizer = None
        self.model = None
        
    def setup_cuda(self):
        """Setup CUDA device for GPU acceleration."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🚀 MPS (Apple Silicon) available!")
        else:
            device = torch.device("cpu")
            print("⚠️  No GPU acceleration available, using CPU")
        
        return device
    
    def load_model_for_categorization(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """Load Llama 3.2-3B-Instruct model for prompt categorization using CUDA."""
        try:
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Fix tokenizer padding token issue
            if self.tokenizer.pad_token is None:
                print("Setting padding token for tokenizer...")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loading model: {model_name}")
            
            # Use safer loading approach with safetensors and no device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True,  # Force use of safetensors format
                device_map=None,  # Don't use device_map to avoid accelerate dependency
                low_cpu_mem_usage=True,  # Reduce memory usage during loading
                trust_remote_code=True  # Required for Llama models
            )
            
            # Manually move model to device
            if self.device.type == "cuda":
                print(f"Moving model to CUDA device...")
                self.model = self.model.to(self.device)
                print(f"Model successfully moved to {self.device}")
            
            print(f"✅ Llama model loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            print(f"Trying alternative loading method...")
            
            # Fallback: try loading without specific dtype
            try:
                print(f"Loading model with fallback method...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_safetensors=True,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                
                # Fix tokenizer padding token issue in fallback too
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                if self.device.type == "cuda":
                    self.model = self.model.to(self.device)
                
                print(f"✅ Llama model loaded successfully with fallback method on {self.device}!")
                return True
                
            except Exception as e2:
                print(f"❌ Fallback loading also failed: {e2}")
                print("⚠️  Llama model requires special access. You may need to:")
                print("1. Request access from Meta: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
                print("2. Use a different open model like 'microsoft/DialoGPT-medium'")
                return False
    
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
        
        # Use chat template format for Llama models
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": f"You are an expert at categorizing text prompts. Your task is to assign the most relevant categories from this list:\n\n{categories_text}\n\nIMPORTANT: Respond with ONLY the category names separated by commas. Do not add explanations, reasoning, or any other text. If the prompt doesn't fit any category well, respond with 'other'.\n\nExample responses:\n- mathematical_reasoning, problem_solving\n- creative_writing\n- other"},
                {"role": "user", "content": f"Categorize this prompt: {prompt_text}"}
            ]
            categorization_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to regular prompt format
            categorization_prompt = f"""You are an expert at categorizing text prompts. Available categories:

{categories_text}

IMPORTANT: Respond with ONLY the category names separated by commas. Do not add explanations or reasoning.

Example responses:
- mathematical_reasoning, problem_solving
- creative_writing
- other

Prompt to categorize: "{prompt_text}"

Categories:"""
        
        return categorization_prompt
    
    def categorize_prompt_with_cuda(self, prompt_text, max_length=512):
        """Categorize a prompt using CUDA-accelerated model inference."""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded. Call load_model_for_categorization() first.")
            return None
        
        try:
            # Create categorization prompt
            full_prompt = self.create_categorization_prompt(prompt_text)
            
            # Tokenize input - optimized for Llama chat template
            inputs = self.tokenizer(
                full_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=False  # No padding needed with chat template
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with CUDA - optimized for Llama
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=128,
                    temperature=0.3,  # Lower temperature for more focused responses
                    do_sample=True,
                    top_p=0.9,  # Add nucleus sampling for better quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1  # Prevent repetitive outputs
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove input prompt)
            generated_text = response[len(full_prompt):].strip()
            
            # Debug: Print the response for troubleshooting
            print(f"    Raw response: '{generated_text}'")
            
            # Parse categories from response
            categories = self.parse_categorization_response(generated_text)
            
            # Debug: Print parsed categories
            print(f"    Parsed categories: {categories}")
            
            return categories
            
        except Exception as e:
            print(f"❌ Error in CUDA categorization: {e}")
            return None
    
    def parse_categorization_response(self, response_text):
        """Parse the model response to extract category names."""
        if not response_text:
            return []
        
        # Clean response and extract categories
        response_clean = response_text.strip().lower()
        
        # Check if response looks like a valid category list
        if not any(separator in response_clean for separator in [',', ';', '\n', 'and', '&']):
            # If no separators, check if it's a single valid category
            if response_clean in [cat.lower() for cat in self.prompt_categories.keys()]:
                return [response_clean]
            else:
                # Invalid response, return empty
                return []
        
        # Split by common separators
        categories = []
        for separator in [',', ';', '\n', 'and', '&']:
            if separator in response_clean:
                parts = response_clean.split(separator)
                categories = [part.strip() for part in parts if part.strip()]
                break
        
        # Filter to only include valid categories from our predefined list
        valid_categories = []
        valid_category_keys = [cat.lower() for cat in self.prompt_categories.keys()]
        
        for cat in categories:
            # Clean the category name
            cat_clean = cat.replace('category:', '').replace('categories:', '').strip()
            
            # Check if it's a valid predefined category
            if cat_clean in valid_category_keys:
                valid_categories.append(cat_clean)
            elif cat_clean == 'other':
                valid_categories.append('other')
        
        return valid_categories[:5]  # Limit to 5 categories max
    
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
        
        # Load model for CUDA categorization
        print(f"\n🚀 LOADING MODEL FOR CUDA CATEGORIZATION")
        if self.load_model_for_categorization():
            print("✅ Model loaded successfully!")
            
            # Run CUDA categorization on sample prompts
            print(f"\n🤖 RUNNING CUDA CATEGORIZATION ON SAMPLE PROMPTS")
            self.run_cuda_categorization(sample_size=10)  # Start with 10 samples for debugging
        else:
            print("⚠️  Model loading failed, continuing with external analysis only")
        
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
    
    def run_cuda_categorization(self, sample_size=50):
        """Run CUDA-accelerated categorization on sample prompts."""
        if not self.prompt_samples:
            print("❌ No prompt samples available for categorization")
            return
        
        print(f"🔍 Running CUDA categorization on {min(sample_size, len(self.prompt_samples))} sample prompts...")
        
        categorized_prompts = []
        successful_categorizations = 0
        
        for i, sample in enumerate(self.prompt_samples[:sample_size]):
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{min(sample_size, len(self.prompt_samples))}...")
            
            prompt_text = sample['prompt'].replace("...", "").strip()
            
            # Run CUDA categorization
            categories = self.categorize_prompt_with_cuda(prompt_text)
            
            if categories:
                successful_categorizations += 1
                categorized_prompts.append({
                    'prompt': prompt_text,
                    'categories': categories,
                    'word_count': sample['word_count'],
                    'char_count': sample['char_count']
                })
                
                # Update category distribution
                for cat in categories:
                    self.category_distribution[cat] += 1
        
        print(f"✅ CUDA categorization complete!")
        print(f"  Successful categorizations: {successful_categorizations}/{min(sample_size, len(self.prompt_samples))}")
        print(f"  Categories discovered: {len(self.category_distribution)}")
        
        # Show top categories
        if self.category_distribution:
            print(f"\n📊 TOP CATEGORIES DISCOVERED:")
            top_categories = sorted(self.category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            for cat, count in top_categories:
                print(f"  {cat}: {count} prompts")
        
        # Save CUDA categorization results
        self.save_cuda_categorization_results(categorized_prompts)
    
    def save_cuda_categorization_results(self, categorized_prompts):
        """Save the CUDA categorization results."""
        output_dir = Path("prompt_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save CUDA categorization results
        cuda_results_file = output_dir / "cuda_categorization_results.json"
        with open(cuda_results_file, 'w', encoding='utf-8') as f:
            json.dump(categorized_prompts, f, indent=2)
        
        # Save category distribution
        category_dist_file = output_dir / "cuda_category_distribution.json"
        with open(category_dist_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.category_distribution), f, indent=2)
        
        print(f"\n💾 CUDA categorization results saved:")
        print(f"  - cuda_categorization_results.json: {len(categorized_prompts)} categorized prompts")
        print(f"  - cuda_category_distribution.json: Category distribution")

if __name__ == "__main__":
    categorizer = PromptCategorizer()
    categorizer.run_analysis()
