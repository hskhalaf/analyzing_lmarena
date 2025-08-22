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
        """Process a batch of prompts and return raw model outputs."""
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
            
            # Extract raw text responses
            results = []
            for i, prompt_text in enumerate(prompt_batch):
                prompt_length = inputs["input_ids"][i].shape[0]
                generated_tokens = outputs[i][prompt_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                results.append(generated_text)
            
            return results
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return [None] * len(prompt_batch)
    
    def run_categorization(self, max_prompts=20000, batch_size=400):
        """Main categorization pipeline - saves raw model outputs."""
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
            
            batch_responses = self.categorize_batch(batch_prompts)
            
            # Store raw results
            for prompt, response in zip(batch_prompts, batch_responses):
                all_results.append({
                    'prompt': prompt,
                    'raw_response': response,
                    'word_count': len(prompt.split())
                })
        
        # Final results
        end_time = time.time()
        total_time = end_time - start_time
        successful = sum(1 for r in all_results if r['raw_response'])
        
        print(f"\n🎉 Complete! {successful:,}/{total_prompts:,} prompts processed")
        print(f"⚡ Performance: {total_time:.1f}s total, {total_prompts/total_time:.1f} prompts/second")
        
        # Save raw results
        self.save_raw_results(all_results)
    
    def save_raw_results(self, results):
        """Save raw model outputs for later parsing."""
        output_dir = Path("prompt_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save raw model outputs
        raw_results_file = output_dir / "raw_model_outputs.json"
        with open(raw_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Raw results saved to prompt_analysis/")
        print(f"  - raw_model_outputs.json: {len(results):,} prompts with raw responses")
        print(f"\n📝 Next step: Run parsing script to categorize the raw outputs")

if __name__ == "__main__":
    categorizer = PromptCategorizer()
    categorizer.run_categorization(max_prompts=20000, batch_size=400)
