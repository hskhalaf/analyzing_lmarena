#!/usr/bin/env python3
"""
Step 6: Discover emergent categories using Llama 3.2-3B-Instruct.
Instead of predefined categories, let the model discover what categories actually exist in the data.
"""

import pickle
from pathlib import Path
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict, Counter
import re

class EmergentCategoryDiscoverer:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = None
        self.model = None
        
        # Data structures
        self.all_prompts = []
        self.sampled_prompts = []
        self.discovered_categories = set()
        self.prompt_to_categories = {}
        
        # Sophisticated prompt engineering
        self.discovery_prompt_template = """You are an expert at analyzing and categorizing diverse text prompts. Your task is to identify the most meaningful, fine-grained categories that emerge from the actual content.

IMPORTANT: Do NOT use generic categories like "coding", "math", "personal", "creative", etc. Instead, look for:
- Specific domains, techniques, or methodologies
- Distinct cognitive tasks or reasoning patterns
- Unique content types or formats
- Specialized knowledge areas
- Specific interaction patterns or scenarios

Analyze the following 10 prompts and suggest 15-25 highly specific, informative categories that capture the true diversity and nuance of the content.

For each prompt, identify 2-4 relevant categories from your suggested taxonomy.

PROMPTS TO ANALYZE:
{prompts}

INSTRUCTIONS:
1. First, suggest 15-25 specific, fine-grained categories (one per line)
2. Then, for each prompt, list the relevant categories (comma-separated)
3. Be specific and avoid generic terms
4. Focus on what makes each prompt unique and challenging

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

CATEGORIES:
- category_name_1: brief_description
- category_name_2: brief_description
- category_name_3: brief_description
...

CLASSIFICATIONS:
Prompt 1: category1, category2, category3
Prompt 2: category1, category4
Prompt 3: category2, category5
...

Now analyze the prompts and provide your taxonomy:"""

    def load_model(self):
        """Load the Llama 3.2-3B-Instruct model."""
        print("="*80)
        print("LOADING LLAMA 3.2-3B-INSTRUCT MODEL")
        print("="*80)
        
        # Check MPS availability
        if torch.backends.mps.is_available():
            print("🚀 MPS (Apple Silicon) acceleration available!")
            device = "mps"
        else:
            print("⚠️  MPS not available, using CPU")
            device = "cpu"
        
        try:
            print(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print(f"Loading model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully!")
            print(f"📱 Device: {device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("This model requires special access. You may need to:")
            print("1. Request access from Meta: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            print("2. Use a different open model like 'microsoft/DialoGPT-medium'")
            print("3. Use an API-based model like OpenAI's GPT-4")
            return False
    
    def load_prompts(self):
        """Load the extracted prompts from the previous step."""
        print("\n" + "="*80)
        print("LOADING EXTRACTED PROMPTS")
        print("="*80)
        
        # Try to load from the prompt analysis directory
        prompts_file = Path("prompt_analysis/all_prompts.txt")
        
        if not prompts_file.exists():
            print("❌ No prompts file found. Please run 05_prompt_categorization.py first.")
            return False
        
        try:
            print("Loading prompts from file...")
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the prompts file
            prompt_blocks = content.split("=== PROMPT ")[1:]  # Skip the first empty split
            
            for block in prompt_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 2:
                    prompt_text = '\n'.join(lines[1:]).strip()
                    if prompt_text:
                        self.all_prompts.append(prompt_text)
            
            print(f"✅ Loaded {len(self.all_prompts):,} prompts")
            return True
            
        except Exception as e:
            print(f"❌ Error loading prompts: {e}")
            return False
    
    def sample_prompts(self, sample_size=1000):
        """Sample random prompts for category discovery."""
        print(f"\n🔍 SAMPLING {sample_size:,} RANDOM PROMPTS")
        
        if sample_size > len(self.all_prompts):
            sample_size = len(self.all_prompts)
            print(f"⚠️  Sample size reduced to {sample_size:,} (total available)")
        
        # Sample randomly
        self.sampled_prompts = random.sample(self.all_prompts, sample_size)
        
        print(f"✅ Sampled {len(self.sampled_prompts):,} prompts")
        
        # Show a few examples
        print(f"\n📝 SAMPLE OF SAMPLED PROMPTS:")
        for i, prompt in enumerate(self.sampled_prompts[:5]):
            print(f"\n--- Sample {i+1} ---")
            print(f"{prompt[:200]}{'...' if len(prompt) > 200 else ''}")
    
    def create_discovery_prompt(self, batch_size=10):
        """Create the discovery prompt for a batch of prompts."""
        # Take the first batch_size prompts
        batch_prompts = self.sampled_prompts[:batch_size]
        
        # Format them nicely
        formatted_prompts = ""
        for i, prompt in enumerate(batch_prompts, 1):
            # Truncate very long prompts
            truncated_prompt = prompt[:300] + "..." if len(prompt) > 300 else prompt
            formatted_prompts += f"Prompt {i}: {truncated_prompt}\n\n"
        
        # Create the full prompt
        full_prompt = self.discovery_prompt_template.format(prompts=formatted_prompts)
        
        return full_prompt, batch_prompts
    
    def generate_categories_with_llama(self, prompt_text):
        """Generate categories using the loaded Llama model."""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
            
            # Move to MPS if available
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove input)
            generated_text = response[len(prompt_text):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Error generating with Llama: {e}")
            return None
    
    def parse_llama_response(self, response_text, original_prompts):
        """Parse the Llama response to extract categories and classifications."""
        print(f"\n🔍 PARSING LLAMA RESPONSE")
        
        # Extract categories section
        categories_section = None
        classifications_section = None
        
        lines = response_text.split('\n')
        in_categories = False
        in_classifications = False
        
        categories = []
        classifications = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('CATEGORIES:'):
                in_categories = True
                in_classifications = False
                continue
            elif line.startswith('CLASSIFICATIONS:'):
                in_categories = False
                in_classifications = True
                continue
            
            if in_categories and line.startswith('-'):
                # Parse category line: "- category_name: brief_description"
                if ':' in line:
                    category_part = line[1:].strip()
                    if ':' in category_part:
                        cat_name, description = category_part.split(':', 1)
                        categories.append(cat_name.strip())
            
            elif in_classifications and line.startswith('Prompt'):
                # Parse classification line: "Prompt 1: category1, category2, category3"
                if ':' in line:
                    prompt_part, cats_part = line.split(':', 1)
                    prompt_num = int(prompt_part.split()[1])
                    if prompt_num <= len(original_prompts):
                        cat_list = [cat.strip() for cat in cats_part.split(',')]
                        classifications[prompt_num - 1] = cat_list
        
        print(f"✅ Parsed {len(categories)} categories")
        print(f"✅ Parsed {len(classifications)} prompt classifications")
        
        return categories, classifications
    
    def run_category_discovery(self, sample_size=1000, batch_size=10):
        """Run the complete category discovery process."""
        print("="*100)
        print("EMERGENT CATEGORY DISCOVERY WITH LLAMA 3.2-3B-INSTRUCT")
        print("Discovering fine-grained, data-driven categories from actual prompts")
        print("="*100)
        
        # Load prompts
        if not self.load_prompts():
            return False
        
        # Sample prompts
        self.sample_prompts(sample_size)
        
        # Load model
        if not self.load_model():
            print("⚠️  Model loading failed. Using fallback approach...")
            return self.run_fallback_discovery()
        
        print(f"\n🚀 STARTING CATEGORY DISCOVERY")
        print(f"Using batch size: {batch_size}")
        
        # Process in batches
        total_batches = (len(self.sampled_prompts) + batch_size - 1) // batch_size
        
        for batch_num in range(min(3, total_batches)):  # Process first 3 batches for now
            print(f"\n{'='*60}")
            print(f"PROCESSING BATCH {batch_num + 1}/{min(3, total_batches)}")
            print(f"{'='*60}")
            
            # Create discovery prompt for this batch
            discovery_prompt, batch_prompts = self.create_discovery_prompt(batch_size)
            
            print(f"📝 DISCOVERY PROMPT:")
            print(f"{discovery_prompt[:500]}...")
            
            # Generate categories with Llama
            print(f"\n🤖 GENERATING CATEGORIES WITH LLAMA...")
            response = self.generate_categories_with_llama(discovery_prompt)
            
            if response:
                print(f"✅ LLAMA RESPONSE RECEIVED")
                print(f"Response length: {len(response)} characters")
                
                # Parse the response
                categories, classifications = self.parse_llama_response(response, batch_prompts)
                
                # Store results
                self.discovered_categories.update(categories)
                
                # Map prompts to their categories
                for prompt_idx, prompt_cats in classifications.items():
                    if prompt_idx < len(batch_prompts):
                        prompt_text = batch_prompts[prompt_idx]
                        self.prompt_to_categories[prompt_text] = prompt_cats
                
                print(f"📊 BATCH RESULTS:")
                print(f"  New categories discovered: {len(categories)}")
                print(f"  Prompts classified: {len(classifications)}")
                print(f"  Total categories so far: {len(self.discovered_categories)}")
                
                # Show some examples
                print(f"\n🔍 SAMPLE CLASSIFICATIONS:")
                for i, (prompt_idx, prompt_cats) in enumerate(list(classifications.items())[:3]):
                    if prompt_idx < len(batch_prompts):
                        prompt = batch_prompts[prompt_idx]
                        print(f"\nPrompt {prompt_idx + 1}:")
                        print(f"  Text: {prompt[:100]}...")
                        print(f"  Categories: {', '.join(prompt_cats)}")
            else:
                print(f"❌ Failed to generate response for batch {batch_num + 1}")
            
            # Move to next batch
            self.sampled_prompts = self.sampled_prompts[batch_size:]
        
        # Save results
        self.save_discovery_results()
        
        return True
    
    def run_fallback_discovery(self):
        """Fallback method when model loading fails."""
        print(f"\n🔄 RUNNING FALLBACK CATEGORY DISCOVERY")
        print(f"Using manual analysis of prompt patterns")
        
        # Analyze prompt patterns manually
        prompt_patterns = self.analyze_prompt_patterns()
        
        # Create a basic taxonomy based on patterns
        basic_categories = self.create_basic_taxonomy(prompt_patterns)
        
        print(f"✅ Fallback discovery complete")
        print(f"Discovered {len(basic_categories)} basic categories")
        
        return True
    
    def analyze_prompt_patterns(self):
        """Analyze patterns in prompts to suggest categories."""
        patterns = defaultdict(int)
        
        for prompt in self.sampled_prompts[:100]:  # Analyze first 100
            # Look for common patterns
            if any(word in prompt.lower() for word in ['solve', 'calculate', 'compute', 'find']):
                patterns['problem_solving'] += 1
            if any(word in prompt.lower() for word in ['write', 'create', 'generate', 'compose']):
                patterns['content_creation'] += 1
            if any(word in prompt.lower() for word in ['compare', 'analyze', 'evaluate', 'assess']):
                patterns['analysis_evaluation'] += 1
            if any(word in prompt.lower() for word in ['translate', 'convert', 'transform']):
                patterns['transformation'] += 1
            if any(word in prompt.lower() for word in ['explain', 'describe', 'define', 'what is']):
                patterns['explanation'] += 1
        
        return patterns
    
    def create_basic_taxonomy(self, patterns):
        """Create a basic taxonomy based on pattern analysis."""
        categories = []
        
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 5:  # Only include patterns that appear frequently
                categories.append(pattern)
        
        return categories
    
    def save_discovery_results(self):
        """Save the discovered categories and classifications."""
        output_dir = Path("emergent_categories")
        output_dir.mkdir(exist_ok=True)
        
        # Save discovered categories
        categories_file = output_dir / "discovered_categories.json"
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.discovered_categories), f, indent=2)
        
        # Save prompt classifications
        classifications_file = output_dir / "prompt_classifications.json"
        classifications_data = {}
        for prompt, cats in self.prompt_to_categories.items():
            # Truncate very long prompts for storage
            prompt_key = prompt[:100] + "..." if len(prompt) > 100 else prompt
            classifications_data[prompt_key] = cats
        
        with open(classifications_file, 'w', encoding='utf-8') as f:
            json.dump(classifications_data, f, indent=2)
        
        # Save summary
        summary_file = output_dir / "discovery_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("EMERGENT CATEGORY DISCOVERY SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total categories discovered: {len(self.discovered_categories)}\n")
            f.write(f"Total prompts classified: {len(self.prompt_to_categories)}\n")
            f.write(f"Sample size used: {len(self.sampled_prompts)}\n\n")
            
            f.write("DISCOVERED CATEGORIES:\n")
            for i, category in enumerate(sorted(self.discovered_categories), 1):
                f.write(f"{i}. {category}\n")
        
        print(f"\n💾 Discovery results saved to: {output_dir}/")
        print(f"  - discovered_categories.json: {len(self.discovered_categories)} categories")
        print(f"  - prompt_classifications.json: {len(self.prompt_to_categories)} classifications")
        print(f"  - discovery_summary.txt: Summary report")

if __name__ == "__main__":
    discoverer = EmergentCategoryDiscoverer()
    discoverer.run_category_discovery(sample_size=1000, batch_size=10)
