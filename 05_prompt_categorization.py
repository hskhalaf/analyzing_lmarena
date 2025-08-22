#!/usr/bin/env python3
"""
Step 5: Extract prompts from Arena dataset and categorize them.
"""

import pickle
from pathlib import Path
from collections import defaultdict
import json
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptCategorizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.prompts = []
        self.category_distribution = defaultdict(int)
        
        # Detailed prompt categories with explanations
        self.categories = {
            'mathematical_reasoning': 'Math problems, calculations, proofs, quantitative analysis, equations, mathematical concepts',
            'logical_reasoning': 'Logic puzzles, deductive reasoning, syllogisms, boolean logic, critical thinking',
            'creative_writing': 'Stories, poems, essays, creative content generation, imaginative writing',
            'technical_writing': 'Documentation, manuals, technical explanations, procedures, how-to guides',
            'translation': 'Language translation, localization, multilingual tasks, language conversion',
            'summarization': 'Text summarization, key point extraction, condensation, executive summaries',
            'factual_qa': 'Question answering, trivia, knowledge retrieval, information seeking',
            'analysis': 'Data analysis, trend identification, pattern recognition, analytical thinking',
            'comparison': 'Comparing concepts, products, approaches, methodologies, evaluation of alternatives',
            'evaluation': 'Assessing quality, ranking, critique, review, performance evaluation',
            'coding': 'Programming, debugging, code review, algorithm implementation, software development',
            'system_design': 'Architecture, system planning, technical design, infrastructure planning',
            'brainstorming': 'Idea generation, concept development, creative thinking, innovation',
            'business_analysis': 'Market research, business strategy, financial analysis, competitive analysis',
            'educational_content': 'Teaching, learning materials, tutorials, explanations, educational resources',
            'assessment': 'Testing, evaluation, skill assessment, knowledge verification, performance testing',
            'scientific': 'Scientific concepts, research, technical explanations, scientific methodology',
            'medical': 'Health topics, medical information, wellness advice, healthcare questions',
            'other': 'Other or unclear categories, miscellaneous topics'
        }
    
    def load_model(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """Load model with minimal configuration."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True,
                # device_map=None,
                trust_remote_code=True
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
            
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def load_prompts(self):
        """Load prompts from cached dataset."""
        cache_file = Path("data/arena_dataset.pkl")
        if not cache_file.exists():
            print("❌ No cached dataset found. Run 01_explore_data.py first.")
            return False
        
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"🔍 Dataset structure: {list(dataset.keys())}")
        print(f"📊 Train examples: {len(dataset['train']):,}")
        
        # Debug: Check first few examples to understand structure
        if dataset['train']:
            first_example = dataset['train'][0]
            print(f"🔍 First example keys: {list(first_example.keys())}")
            
            # Debug conversation_a structure
            if 'conversation_a' in first_example:
                conv_a = first_example['conversation_a']
                print(f"🔍 conversation_a type: {type(conv_a)}")
                if isinstance(conv_a, list) and len(conv_a) > 0:
                    print(f"🔍 conversation_a[0]: {conv_a[0]}")
                    if isinstance(conv_a[0], dict):
                        print(f"🔍 conversation_a[0] keys: {list(conv_a[0].keys())}")
        
        prompts = []
        for i, example in enumerate(dataset['train']):
            if i % 10000 == 0:
                print(f"Processing example {i:,}/{len(dataset['train']):,}")
            
            # Try to extract prompt from conversation_a (first conversation)
            prompt = None
            
            if 'conversation_a' in example and example['conversation_a']:
                conv_a = example['conversation_a']
                if isinstance(conv_a, list) and len(conv_a) > 0:
                    # Look for the first user message
                    for msg in conv_a:
                        if isinstance(msg, dict):
                            # Check if this is a user message
                            if msg.get('role') == 'user' or 'user' in msg:
                                if 'content' in msg:
                                    content = msg['content']
                                    if isinstance(content, str):
                                        prompt = content
                                        break
                                    elif isinstance(content, list) and len(content) > 0:
                                        # Handle content as list of blocks
                                        for block in content:
                                            if isinstance(block, dict) and 'text' in block:
                                                prompt = block['text']
                                                break
                                        if prompt:
                                            break
                            # Also try to find the first message if role is not specified
                            elif 'content' in msg and not prompt:
                                content = msg['content']
                                if isinstance(content, str):
                                    prompt = content
                                    break
                                elif isinstance(content, list) and len(content) > 0:
                                    for block in content:
                                        if isinstance(block, dict) and 'text' in block:
                                            prompt = block['text']
                                            break
                                    if prompt:
                                        break
            
            # If no prompt found in conversation_a, try conversation_b
            if not prompt and 'conversation_b' in example and example['conversation_b']:
                conv_b = example['conversation_b']
                if isinstance(conv_b, list) and len(conv_b) > 0:
                    for msg in conv_b:
                        if isinstance(msg, dict):
                            if msg.get('role') == 'user' or 'user' in msg:
                                if 'content' in msg:
                                    content = msg['content']
                                    if isinstance(content, str):
                                        prompt = content
                                        break
                                    elif isinstance(content, list) and len(content) > 0:
                                        for block in content:
                                            if isinstance(block, dict) and 'text' in block:
                                                prompt = block['text']
                                                break
                                        if prompt:
                                            break
            
            # Clean and validate prompt
            if prompt and isinstance(prompt, str):
                prompt = prompt.strip()
                if len(prompt) > 10:  # Minimum length filter
                    prompts.append(prompt)
        
        self.prompts = prompts
        print(f"✅ Loaded {len(prompts):,} prompts")
        
        if len(prompts) == 0:
            print("⚠️  No prompts found! Debugging dataset structure...")
            if dataset['train']:
                sample_example = dataset['train'][0]
                print(f"Sample example keys: {list(sample_example.keys())}")
                if 'conversation_a' in sample_example:
                    conv_a = sample_example['conversation_a']
                    print(f"conversation_a: {conv_a}")
                if 'conversation_b' in sample_example:
                    conv_b = sample_example['conversation_b']
                    print(f"conversation_b: {conv_b}")
        
        return True
    
    def create_categorization_prompt(self, prompt_text):
        """Create categorization prompt with detailed category explanations."""
        categories_text = "\n".join([f"- {cat}: {desc}" for cat, desc in self.categories.items()])
        
        return f"""Categorize this prompt into 1-3 most relevant categories from the list below.

Available categories:
{categories_text}

IMPORTANT RULES:
1. Choose EXACTLY 1-3 categories (no more, no less)
2. Each category can only appear ONCE
3. Respond with ONLY category names separated by commas
4. No explanations, quotes, or extra text
5. Use 'other' if none fit well
6. Be precise - don't over-categorize

EXAMPLES:
Prompt: 'Solve this equation: 2x + 5 = 13'
Response: mathematical_reasoning

Prompt: 'Write a story about a magical forest'
Response: creative_writing

Prompt: 'What is the capital of France?'
Response: factual_qa

Prompt: 'Compare iPhone vs Android features'
Response: comparison

Text to categorize: {prompt_text}

Categories:"""
    
    def categorize_batch(self, prompt_batch, max_length=512):
        """Process a batch of prompts efficiently."""
        if not self.model or not self.tokenizer:
            return [None] * len(prompt_batch)
        
        try:
            # Create prompts for batch
            all_prompts = [self.create_categorization_prompt(p) for p in prompt_batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                all_prompts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding='longest',
                return_attention_mask=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=64,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Process outputs
            results = []
            for i, prompt_text in enumerate(prompt_batch):
                prompt_length = inputs["input_ids"][i].shape[0]
                generated_tokens = outputs[i][prompt_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                categories = self.parse_categories(generated_text)
                results.append(categories)
                
                # Update distribution
                for cat in categories:
                    self.category_distribution[cat] += 1
            
            return results
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return [None] * len(prompt_batch)
    
    def parse_categories(self, response_text):
        """Parse categories from model response."""
        if not response_text:
            return []
        
        # Clean response
        response_clean = response_text.strip().lower()
        if "response:" in response_clean:
            response_clean = response_clean.split("response:")[-1].strip()
        
        # Split by separators
        for separator in [',', ';', '\n', 'and', '&']:
            if separator in response_clean:
                parts = [p.strip() for p in response_clean.split(separator) if p.strip()]
                break
        else:
            parts = [response_clean]
        
        # Validate categories
        valid_categories = []
        valid_keys = set(self.categories.keys())
        
        for part in parts:
            part_clean = part.replace('category:', '').replace('categories:', '').strip()
            if part_clean in valid_keys:
                valid_categories.append(part_clean)
            elif part_clean == 'other':
                valid_categories.append('other')
        
        return valid_categories[:3]  # Max 3 categories
    
    def run_categorization(self, max_prompts=20000, batch_size=400):
        """Main categorization pipeline."""
        if not self.load_prompts():
            return
        
        if not self.load_model():
            return
        
        # Random sample if needed
        if max_prompts and max_prompts < len(self.prompts):
            self.prompts = random.sample(self.prompts, max_prompts)
            print(f"🎲 Randomly selected {max_prompts:,} prompts")
        
        total_prompts = len(self.prompts)
        total_batches = (total_prompts + batch_size - 1) // batch_size
        
        print(f"🚀 Processing {total_prompts:,} prompts in {total_batches} batches (size: {batch_size})")
        
        # Process batches
        start_time = time.time()
        all_results = []
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_prompts)
            batch_prompts = self.prompts[start_idx:end_idx]
            
            print(f"Batch {batch_num + 1}/{total_batches}: {start_idx + 1:,}-{end_idx:,}")
            
            batch_results = self.categorize_batch(batch_prompts)
            
            # Store results - THIS IS NOW INSIDE THE LOOP!
            for prompt, categories in zip(batch_prompts, batch_results):
                all_results.append({
                    'prompt': prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    'categories': categories or [],
                    'word_count': len(prompt.split()),
                    'char_count': len(prompt)
                })
        
        # Final results
        end_time = time.time()
        total_time = end_time - start_time
        successful = sum(1 for r in all_results if r['categories'])
        
        print(f"\n🎉 Complete! {successful:,}/{total_prompts:,} prompts categorized")
        print(f"⚡ Performance: {total_time:.1f}s total, {total_prompts/total_time:.1f} prompts/second")
        
        # Show top categories
        if self.category_distribution:
            print(f"\n🏆 Top categories:")
            top_categories = sorted(self.category_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            for cat, count in top_categories:
                percentage = (count / total_prompts) * 100
                print(f"  {cat}: {count:,} ({percentage:.1f}%)")
        
        # Save results
        self.save_results(all_results)
    
    def save_results(self, results):
        """Save results in simple JSON format."""
        output_dir = Path("prompt_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save categorized prompts
        results_file = output_dir / "categorized_prompts.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save category distribution
        dist_file = output_dir / "category_distribution.json"
        with open(dist_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.category_distribution), f, indent=2)
        
        # Save category definitions for reference
        categories_file = output_dir / "category_definitions.json"
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, indent=2)
        
        print(f"\n💾 Results saved to prompt_analysis/")
        print(f"  - categorized_prompts.json: {len(results):,} prompts")
        print(f"  - category_distribution.json: Category counts")
        print(f"  - category_definitions.json: Category explanations")

if __name__ == "__main__":
    categorizer = PromptCategorizer()
    categorizer.run_categorization(max_prompts=20000, batch_size=400)
