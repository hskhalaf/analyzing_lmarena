#!/usr/bin/env python3
"""
Step 7: Analyze coding prompts with different persona perspectives.
Compare how PhD students vs software engineers would evaluate coding prompts,
and analyze how this differs from actual human preferences in the dataset.
"""

import pickle
from pathlib import Path
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PersonaCodingAnalyzer:
    def __init__(self):
        # CUDA setup
        self.device = self.setup_cuda()
        
        self.dataset = None
        self.coding_data = []
        self.persona_prompts = {}
        self.persona_preferences = {}
        self.actual_preferences = {}
        
        # Model components for CUDA
        self.tokenizer = None
        self.model = None
        
        # Define persona prompts
        self.setup_personas()
    
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
    
    def load_model_for_evaluation(self, model_name="microsoft/DialoGPT-medium"):
        """Load a model for persona evaluation using CUDA."""
        try:
            print(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loading model: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=self.device if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def setup_personas(self):
        """Setup different persona evaluation prompts."""
        print("🎭 SETTING UP PERSONA EVALUATION PROMPTS")
        
        # PhD Student Persona - Academic, research-focused
        self.persona_prompts['phd_student'] = """You are a PhD student in Computer Science with expertise in algorithms, data structures, and theoretical computer science. You value:

1. **Correctness & Efficiency**: Optimal time/space complexity, elegant algorithms
2. **Academic Rigor**: Mathematical proofs, formal analysis, edge case handling
3. **Research Value**: Novel approaches, interesting theoretical insights
4. **Code Quality**: Clean, readable, well-documented code
5. **Scalability**: How well the solution generalizes to larger problems

When evaluating coding solutions, prioritize:
- Algorithmic efficiency and correctness
- Mathematical soundness
- Academic contribution and novelty
- Theoretical analysis and proofs
- Research implications

Rate solutions from 1-10 based on these criteria."""

        # Software Engineer Persona - Industry, practical-focused
        self.persona_prompts['software_engineer'] = """You are a senior software engineer with 10+ years of industry experience. You value:

1. **Practicality**: Working solutions that solve real problems
2. **Maintainability**: Code that's easy to read, debug, and modify
3. **Performance**: Efficient execution in production environments
4. **Robustness**: Error handling, edge cases, production readiness
5. **Team Collaboration**: Code that other developers can understand and work with

When evaluating coding solutions, prioritize:
- Does it work reliably in production?
- Is it maintainable and readable?
- Does it handle edge cases gracefully?
- Is it performant enough for real-world use?
- Can the team easily work with this code?

Rate solutions from 1-10 based on these criteria."""

        # Add more personas
        self.persona_prompts['code_reviewer'] = """You are a senior code reviewer at a major tech company. You value:

1. **Code Standards**: Following best practices and company guidelines
2. **Security**: Avoiding vulnerabilities and security issues
3. **Testing**: Proper test coverage and validation
4. **Documentation**: Clear comments and documentation
5. **Performance**: Efficient algorithms and data structures

When evaluating coding solutions, prioritize:
- Adherence to coding standards
- Security and vulnerability assessment
- Testability and validation
- Documentation quality
- Performance characteristics

Rate solutions from 1-10 based on these criteria."""

        self.persona_prompts['startup_founder'] = """You are a startup founder who needs to ship quickly. You value:

1. **Speed**: Getting working solutions out fast
2. **Scalability**: Can this handle growth?
3. **User Experience**: Does it solve the user's problem?
4. **Cost**: Efficient resource usage
5. **Flexibility**: Easy to modify as requirements change

When evaluating coding solutions, prioritize:
- Time to market
- Scalability potential
- User problem solving
- Resource efficiency
- Adaptability

Rate solutions from 1-10 based on these criteria."""

        print(f"✅ Setup {len(self.persona_prompts)} personas:")
        for persona in self.persona_prompts.keys():
            print(f"  - {persona.replace('_', ' ').title()}")
    
    def load_dataset(self):
        """Load the Arena dataset."""
        print("\n" + "="*80)
        print("LOADING ARENA DATASET")
        print("="*80)
        
        # Try to load from cache first
        cache_file = Path("data/arena_dataset.pkl")
        
        if cache_file.exists():
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            print("❌ No cached dataset found. Please run 01_explore_data.py first.")
            return False
        
        print(f"✅ Dataset loaded successfully!")
        print(f"  - Train size: {len(self.dataset['train']):,}")
        print(f"  - Test size: {len(self.dataset['test']):,}")
        
        return True
    
    def extract_coding_data(self):
        """Extract all rows where is_code is True."""
        print("\n" + "="*80)
        print("EXTRACTING CODING DATA")
        print("="*80)
        
        if not self.dataset:
            print("❌ Dataset not loaded")
            return False
        
        train_data = self.dataset['train']
        
        # Extract coding prompts
        coding_count = 0
        total_count = 0
        
        for example in train_data:
            total_count += 1
            
            # Check if this is a coding prompt
            if example.get('is_code', False):
                coding_count += 1
                
                # Extract relevant information
                coding_example = {
                    'id': example.get('id', f'coding_{coding_count}'),
                    'prompt': example.get('prompt', ''),
                    'response_a': example.get('response_a', ''),
                    'response_b': example.get('response_b', ''),
                    'preferred': example.get('preferred', ''),
                    'model_a': example.get('model_a', ''),
                    'model_b': example.get('model_b', ''),
                    'category': example.get('category', ''),
                    'is_code': example.get('is_code', False),
                    'difficulty': example.get('difficulty', ''),
                    'language': example.get('language', ''),
                    'tags': example.get('tags', [])
                }
                
                self.coding_data.append(coding_example)
        
        print(f"✅ Extracted {coding_count:,} coding prompts out of {total_count:,} total")
        
        # Show some statistics
        if self.coding_data:
            print(f"\n📊 CODING DATA STATISTICS:")
            
            # Difficulty distribution
            difficulties = Counter(item['difficulty'] for item in self.coding_data if item['difficulty'])
            print(f"  Difficulty levels: {dict(difficulties)}")
            
            # Language distribution
            languages = Counter(item['language'] for item in self.coding_data if item['language'])
            print(f"  Programming languages: {dict(languages)}")
            
            # Category distribution
            categories = Counter(item['category'] for item in self.coding_data if item['category'])
            print(f"  Categories: {dict(categories)}")
            
            # Model distribution
            models_a = Counter(item['model_a'] for item in self.coding_data if item['model_a'])
            models_b = Counter(item['model_b'] for item in self.coding_data if item['model_b'])
            print(f"  Model A distribution: {dict(models_a.most_common(5))}")
            print(f"  Model B distribution: {dict(models_b.most_common(5))}")
        
        return True
    
    def analyze_actual_preferences(self):
        """Analyze actual human preferences in the coding data."""
        print("\n" + "="*80)
        print("ANALYZING ACTUAL HUMAN PREFERENCES")
        print("="*80)
        
        if not self.coding_data:
            print("❌ No coding data available")
            return False
        
        # Analyze preference patterns
        preferences = Counter()
        model_vs_model = defaultdict(lambda: defaultdict(int))
        difficulty_preferences = defaultdict(lambda: defaultdict(int))
        
        for item in self.coding_data:
            preferred = item.get('preferred', '')
            model_a = item.get('model_a', '')
            model_b = item.get('model_b', '')
            difficulty = item.get('difficulty', '')
            
            if preferred:
                preferences[preferred] += 1
                
                # Track model vs model preferences
                if preferred == 'A':
                    model_vs_model[model_a][model_b] += 1
                elif preferred == 'B':
                    model_vs_model[model_b][model_a] += 1
                
                # Track difficulty-based preferences
                if difficulty:
                    if preferred == 'A':
                        difficulty_preferences[difficulty]['A'] += 1
                    elif preferred == 'B':
                        difficulty_preferences[difficulty]['B'] += 1
        
        self.actual_preferences = {
            'overall': dict(preferences),
            'model_vs_model': dict(model_vs_model),
            'difficulty_based': dict(difficulty_preferences)
        }
        
        print(f"✅ Analyzed preferences for {len(self.coding_data)} coding prompts")
        print(f"\n📊 PREFERENCE ANALYSIS:")
        print(f"  Overall preferences: {dict(preferences)}")
        
        # Show some interesting patterns
        if preferences:
            total = sum(preferences.values())
            print(f"  Preference distribution:")
            for pref, count in preferences.items():
                percentage = (count / total) * 100
                print(f"    {pref}: {count} ({percentage:.1f}%)")
        
        return True
    
    def simulate_persona_preferences(self):
        """Simulate how different personas would evaluate coding prompts."""
        print("\n" + "="*80)
        print("SIMULATING PERSONA PREFERENCES")
        print("="*80)
        
        if not self.coding_data:
            print("❌ No coding data available")
            return False
        
        # Load model for CUDA evaluation if available
        print("🚀 LOADING MODEL FOR CUDA EVALUATION")
        if self.load_model_for_evaluation():
            print("✅ Model loaded successfully!")
            print("🤖 RUNNING CUDA-ACCELERATED PERSONA EVALUATIONS...")
            
            for persona_name, persona_prompt in self.persona_prompts.items():
                print(f"\n🎭 Running {persona_name.replace('_', ' ').title()} evaluation with CUDA...")
                
                # Run CUDA-accelerated evaluation
                persona_prefs = self.run_cuda_persona_evaluation(persona_name, self.coding_data[:50])  # Sample first 50
                
                self.persona_preferences[persona_name] = persona_prefs
                
                print(f"  ✅ {persona_name}: {len(persona_prefs)} evaluations completed with CUDA")
        else:
            print("⚠️  Model loading failed, using fallback simulation...")
            
            for persona_name, persona_prompt in self.persona_prompts.items():
                print(f"\n🎭 Simulating {persona_name.replace('_', ' ').title()} preferences...")
                
                # Simulate preferences based on persona characteristics
                persona_prefs = self.simulate_persona_evaluation(persona_name, self.coding_data[:100])  # Sample first 100
                
                self.persona_preferences[persona_name] = persona_prefs
                
                print(f"  ✅ {persona_name}: {len(persona_prefs)} evaluations simulated")
        
        return True
    
    def run_cuda_persona_evaluation(self, persona_name, coding_items):
        """Run CUDA-accelerated persona evaluation using the loaded model."""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded for CUDA evaluation")
            return self.simulate_persona_evaluation(persona_name, coding_items)
        
        persona_prefs = []
        persona_prompt = self.persona_prompts[persona_name]
        
        print(f"  🔍 Evaluating {len(coding_items)} coding items with CUDA...")
        
        for i, item in enumerate(coding_items):
            if i % 10 == 0:
                print(f"    Processing item {i+1}/{len(coding_items)}...")
            
            # Create evaluation prompt for this persona
            evaluation_prompt = self.create_evaluation_prompt(persona_name, persona_prompt, item)
            
            # Run CUDA evaluation
            evaluation_result = self.evaluate_with_cuda(evaluation_prompt)
            
            if evaluation_result:
                # Parse the evaluation result
                score_a, score_b = self.parse_evaluation_result(evaluation_result)
                
                # Determine preference
                if score_a > score_b:
                    persona_pref = 'A'
                elif score_b > score_a:
                    persona_pref = 'B'
                else:
                    persona_pref = 'tie'
                
                persona_prefs.append({
                    'id': item['id'],
                    'score_a': score_a,
                    'score_b': score_b,
                    'preference': persona_pref,
                    'actual_preference': item.get('preferred', ''),
                    'agreement': persona_pref == item.get('preferred', ''),
                    'evaluation_method': 'cuda'
                })
            else:
                # Fallback to simulation if CUDA evaluation fails
                score_a = self.simulate_persona_score(persona_name, item['response_a'])
                score_b = self.simulate_persona_score(persona_name, item['response_b'])
                
                if score_a > score_b:
                    persona_pref = 'A'
                elif score_b > score_a:
                    persona_pref = 'B'
                else:
                    persona_pref = 'tie'
                
                persona_prefs.append({
                    'id': item['id'],
                    'score_a': score_a,
                    'score_b': score_b,
                    'preference': persona_pref,
                    'actual_preference': item.get('preferred', ''),
                    'agreement': persona_pref == item.get('preferred', ''),
                    'evaluation_method': 'simulation_fallback'
                })
        
        return persona_prefs
    
    def create_evaluation_prompt(self, persona_name, persona_prompt, coding_item):
        """Create an evaluation prompt for the persona to evaluate coding responses."""
        prompt = f"""You are evaluating coding solutions from the perspective of a {persona_name.replace('_', ' ')}.

{persona_prompt}

Please evaluate the following two coding solutions and rate them from 1-10:

SOLUTION A:
{coding_item['response_a'][:500]}

SOLUTION B:
{coding_item['response_b'][:500]}

Rate each solution from 1-10 and explain your reasoning. Format your response as:
Score A: [1-10]
Score B: [1-10]
Reasoning: [brief explanation]"""
        
        return prompt
    
    def evaluate_with_cuda(self, prompt_text, max_length=1024):
        """Evaluate the prompt using CUDA-accelerated model inference."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with CUDA
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove input)
            generated_text = response[len(prompt_text):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"❌ Error in CUDA evaluation: {e}")
            return None
    
    def parse_evaluation_result(self, evaluation_text):
        """Parse the evaluation result to extract scores."""
        try:
            # Default scores
            score_a, score_b = 5, 5
            
            # Look for score patterns
            lines = evaluation_text.lower().split('\n')
            for line in lines:
                if 'score a:' in line:
                    try:
                        score_a = int(line.split(':')[1].strip().split()[0])
                        score_a = max(1, min(10, score_a))  # Clamp to 1-10
                    except:
                        pass
                elif 'score b:' in line:
                    try:
                        score_b = int(line.split(':')[1].strip().split()[0])
                        score_b = max(1, min(10, score_b))  # Clamp to 1-10
                    except:
                        pass
            
            return score_a, score_b
            
        except Exception as e:
            print(f"❌ Error parsing evaluation result: {e}")
            return 5, 5  # Default scores
    
    def simulate_persona_score(self, persona_name, response):
        """Simulate persona scoring as fallback method."""
        # This is the existing simulation logic
        if persona_name == 'phd_student':
            return self.simulate_phd_score(response)
        elif persona_name == 'software_engineer':
            return self.simulate_engineer_score(response)
        elif persona_name == 'code_reviewer':
            return self.simulate_reviewer_score(response)
        elif persona_name == 'startup_founder':
            return self.simulate_startup_score(response)
        else:
            return np.random.randint(1, 11)
    
    def simulate_persona_evaluation(self, persona_name, coding_items):
        """Simulate how a specific persona would evaluate coding items."""
        # This is a simplified simulation - in practice, you'd use an LLM
        
        persona_prefs = []
        
        for item in coding_items:
            # Simulate different evaluation criteria based on persona
            if persona_name == 'phd_student':
                # PhD students might prefer more academic, theoretical solutions
                score_a = self.simulate_phd_score(item['response_a'])
                score_b = self.simulate_phd_score(item['response_b'])
            elif persona_name == 'software_engineer':
                # Engineers might prefer practical, production-ready solutions
                score_a = self.simulate_engineer_score(item['response_a'])
                score_b = self.simulate_engineer_score(item['response_b'])
            elif persona_name == 'code_reviewer':
                # Code reviewers might focus on standards and security
                score_a = self.simulate_reviewer_score(item['response_a'])
                score_b = self.simulate_reviewer_score(item['response_b'])
            elif persona_name == 'startup_founder':
                # Startup founders might prioritize speed and scalability
                score_a = self.simulate_startup_score(item['response_a'])
                score_b = self.simulate_startup_score(item['response_b'])
            else:
                score_a = np.random.randint(1, 11)
                score_b = np.random.randint(1, 11)
            
            # Determine preference
            if score_a > score_b:
                persona_pref = 'A'
            elif score_b > score_a:
                persona_pref = 'B'
            else:
                persona_pref = 'tie'
            
            persona_prefs.append({
                'id': item['id'],
                'score_a': score_a,
                'score_b': score_b,
                'preference': persona_pref,
                'actual_preference': item.get('preferred', ''),
                'agreement': persona_pref == item.get('preferred', '')
            })
        
        return persona_prefs
    
    def simulate_phd_score(self, response):
        """Simulate PhD student scoring (academic focus)."""
        # PhD students value theoretical correctness, elegance, academic rigor
        base_score = np.random.randint(5, 9)
        
        # Boost for academic keywords
        academic_keywords = ['algorithm', 'complexity', 'theorem', 'proof', 'analysis', 'optimal']
        if any(keyword in response.lower() for keyword in academic_keywords):
            base_score += 1
        
        return min(10, base_score)
    
    def simulate_engineer_score(self, response):
        """Simulate software engineer scoring (practical focus)."""
        # Engineers value practicality, maintainability, production readiness
        base_score = np.random.randint(6, 9)
        
        # Boost for practical keywords
        practical_keywords = ['error', 'handle', 'test', 'production', 'maintain', 'robust']
        if any(keyword in response.lower() for keyword in practical_keywords):
            base_score += 1
        
        return min(10, base_score)
    
    def simulate_reviewer_score(self, response):
        """Simulate code reviewer scoring (standards focus)."""
        # Reviewers value standards, security, documentation
        base_score = np.random.randint(5, 8)
        
        # Boost for standards keywords
        standards_keywords = ['document', 'comment', 'secure', 'validate', 'standard', 'best practice']
        if any(keyword in response.lower() for keyword in standards_keywords):
            base_score += 1
        
        return min(10, base_score)
    
    def simulate_startup_score(self, response):
        """Simulate startup founder scoring (speed focus)."""
        # Startup founders value speed, scalability, user focus
        base_score = np.random.randint(4, 8)
        
        # Boost for startup keywords
        startup_keywords = ['quick', 'fast', 'scale', 'user', 'simple', 'efficient']
        if any(keyword in response.lower() for keyword in startup_keywords):
            base_score += 1
        
        return min(10, base_score)
    
    def compare_persona_vs_actual(self):
        """Compare persona preferences with actual human preferences."""
        print("\n" + "="*80)
        print("COMPARING PERSONA VS ACTUAL PREFERENCES")
        print("="*80)
        
        if not self.persona_preferences or not self.actual_preferences:
            print("❌ No preferences to compare")
            return False
        
        print("📊 PERSONA VS ACTUAL PREFERENCE ANALYSIS")
        
        for persona_name, persona_prefs in self.persona_preferences.items():
            print(f"\n🎭 {persona_name.replace('_', ' ').title()}:")
            
            # Calculate agreement rates
            total_evaluations = len(persona_prefs)
            agreements = sum(1 for pref in persona_prefs if pref['agreement'])
            agreement_rate = (agreements / total_evaluations) * 100 if total_evaluations > 0 else 0
            
            print(f"  Total evaluations: {total_evaluations}")
            print(f"  Agreements with humans: {agreements}")
            print(f"  Agreement rate: {agreement_rate:.1f}%")
            
            # Analyze preference distribution
            persona_dist = Counter(pref['preference'] for pref in persona_prefs)
            print(f"  Persona preference distribution: {dict(persona_dist)}")
            
            # Compare with actual distribution
            actual_dist = self.actual_preferences['overall']
            print(f"  Actual preference distribution: {actual_dist}")
            
            # Calculate correlation
            correlation = self.calculate_preference_correlation(persona_prefs)
            print(f"  Preference correlation: {correlation:.3f}")
    
    def calculate_preference_correlation(self, persona_prefs):
        """Calculate correlation between persona and actual preferences."""
        if not persona_prefs:
            return 0.0
        
        # Create preference vectors
        persona_vec = []
        actual_vec = []
        
        for pref in persona_prefs:
            if pref['preference'] == 'A':
                persona_vec.append(1)
            elif pref['preference'] == 'B':
                persona_vec.append(0)
            else:  # tie
                persona_vec.append(0.5)
            
            if pref['actual_preference'] == 'A':
                actual_vec.append(1)
            elif pref['actual_preference'] == 'B':
                actual_vec.append(0)
            else:  # no preference or tie
                actual_vec.append(0.5)
        
        # Calculate correlation
        if len(persona_vec) > 1:
            correlation = np.corrcoef(persona_vec, actual_vec)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def save_analysis_results(self):
        """Save the analysis results."""
        output_dir = Path("persona_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save persona preferences
        preferences_file = output_dir / "persona_preferences.json"
        with open(preferences_file, 'w', encoding='utf-8') as f:
            json.dump(self.persona_preferences, f, indent=2, default=str)
        
        # Save actual preferences
        actual_file = output_dir / "actual_preferences.json"
        with open(actual_file, 'w', encoding='utf-8') as f:
            json.dump(self.actual_preferences, f, indent=2, default=str)
        
        # Save coding data summary
        summary_file = output_dir / "coding_analysis_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PERSONA CODING ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total coding prompts analyzed: {len(self.coding_data)}\n\n")
            
            f.write("PERSONAS ANALYZED:\n")
            for persona in self.persona_prompts.keys():
                f.write(f"- {persona.replace('_', ' ').title()}\n")
            
            f.write(f"\nACTUAL PREFERENCE DISTRIBUTION:\n")
            if self.actual_preferences.get('overall'):
                for pref, count in self.actual_preferences['overall'].items():
                    f.write(f"- {pref}: {count}\n")
        
        print(f"\n💾 Analysis results saved to: {output_dir}/")
        print(f"  - persona_preferences.json: Persona evaluation results")
        print(f"  - actual_preferences.json: Actual human preferences")
        print(f"  - coding_analysis_summary.txt: Summary report")
    
    def run_analysis(self):
        """Run the complete persona coding analysis."""
        print("="*100)
        print("PERSONA CODING ANALYSIS")
        print("Comparing how different personas evaluate coding prompts")
        print("="*100)
        
        # Load dataset
        if not self.load_dataset():
            return False
        
        # Extract coding data
        if not self.extract_coding_data():
            return False
        
        # Analyze actual preferences
        if not self.analyze_actual_preferences():
            return False
        
        # Simulate persona preferences
        if not self.simulate_persona_preferences():
            return False
        
        # Compare preferences
        self.compare_persona_vs_actual()
        
        # Save results
        self.save_analysis_results()
        
        print(f"\n🎉 PERSONA CODING ANALYSIS COMPLETE!")
        print(f"Analyzed {len(self.coding_data)} coding prompts with {len(self.persona_prompts)} personas")

if __name__ == "__main__":
    analyzer = PersonaCodingAnalyzer()
    analyzer.run_analysis()
