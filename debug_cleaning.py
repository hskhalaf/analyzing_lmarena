#!/usr/bin/env python3
"""
Debug script to understand why prompts are being rejected during cleaning/filtering.
"""

import pickle
from pathlib import Path

def debug_prompt_cleaning():
    """Debug the prompt cleaning process to see what's being rejected."""
    print("="*80)
    print("DEBUGGING PROMPT CLEANING/FILTERING")
    print("="*80)
    
    # Load cached dataset
    cache_file = Path("data/arena_dataset.pkl")
    if not cache_file.exists():
        print("❌ No cached dataset found.")
        return
    
    try:
        print("Loading cached dataset...")
        with open(cache_file, 'rb') as f:
            dataset = pickle.load(f)
        
        train_data = dataset['train']
        print(f"✅ Dataset loaded: {len(train_data):,} examples")
        
        print(f"\n🔍 DEBUGGING CLEANING LOGIC:")
        
        extracted_count = 0
        cleaned_count = 0
        rejected_count = 0
        too_short_count = 0
        url_like_count = 0
        other_reason_count = 0
        
        for i, ex in enumerate(train_data):
            if i % 20000 == 0:
                print(f"Processing {i}/{len(train_data)} examples...")
            
            # Extract prompt (same logic as main script)
            prompt = None
            
            if 'conversation_a' in ex and ex['conversation_a']:
                conv_a = ex['conversation_a']
                if isinstance(conv_a, list) and len(conv_a) > 0:
                    first_message = conv_a[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            if not prompt and 'conversation_b' in ex and ex['conversation_b']:
                conv_b = ex['conversation_b']
                if isinstance(conv_b, list) and len(conv_b) > 0:
                    first_message = conv_b[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            if not prompt and 'full_conversation' in ex and ex['full_conversation']:
                full_conv = ex['full_conversation']
                if isinstance(full_conv, list) and len(full_conv) > 0:
                    user_msg = full_conv[0].get('user', {})
                    if 'content' in user_msg:
                        content = user_msg['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            if prompt:
                extracted_count += 1
                
                # Now test the cleaning logic
                original_prompt = prompt
                
                # Check length
                if len(original_prompt.strip()) < 10:
                    too_short_count += 1
                    if too_short_count <= 3:
                        print(f"\n--- TOO SHORT PROMPT {too_short_count} ---")
                        print(f"Length: {len(original_prompt.strip())}")
                        print(f"Content: '{original_prompt}'")
                        print("-" * 50)
                    continue
                
                # Check if it's URL-like
                if original_prompt.startswith('http') or ('/' in original_prompt and '.' in original_prompt):
                    url_like_count += 1
                    if url_like_count <= 3:
                        print(f"\n--- URL-LIKE PROMPT {url_like_count} ---")
                        print(f"Content: '{original_prompt}'")
                        print("-" * 50)
                    continue
                
                # If we get here, it passed cleaning
                cleaned_count += 1
                
                # Show first few successful examples
                if cleaned_count <= 3:
                    print(f"\n--- SUCCESSFUL PROMPT {cleaned_count} ---")
                    print(f"Length: {len(original_prompt.strip())}")
                    print(f"Content: '{original_prompt[:100]}{'...' if len(original_prompt) > 100 else ''}'")
                    print("-" * 50)
            else:
                rejected_count += 1
                
                # Show first few rejected examples
                if rejected_count <= 3:
                    print(f"\n--- REJECTED EXAMPLE {rejected_count} ---")
                    print(f"Keys: {list(ex.keys())}")
                    if 'conversation_a' in ex:
                        conv = ex['conversation_a']
                        print(f"conv_a type: {type(conv)}")
                        if isinstance(conv, list) and len(conv) > 0:
                            first = conv[0]
                            print(f"First message: {first}")
                    print("-" * 50)
            
            # Stop after processing a reasonable sample
            if i >= 5000:
                break
        
        print(f"\n📊 CLEANING DEBUG SUMMARY:")
        print(f"Examples processed: {i+1}")
        print(f"Successfully extracted: {extracted_count}")
        print(f"Passed cleaning: {cleaned_count}")
        print(f"Rejected during extraction: {rejected_count}")
        print(f"Rejected during cleaning:")
        print(f"  - Too short (<10 chars): {too_short_count}")
        print(f"  - URL-like: {url_like_count}")
        print(f"  - Other reasons: {other_reason_count}")
        
        # Calculate rates
        extraction_rate = (extracted_count / (i+1)) * 100
        cleaning_rate = (cleaned_count / extracted_count) * 100 if extracted_count > 0 else 0
        overall_rate = (cleaned_count / (i+1)) * 100
        
        print(f"\n📈 SUCCESS RATES:")
        print(f"Extraction rate: {extraction_rate:.1f}%")
        print(f"Cleaning rate: {cleaning_rate:.1f}%")
        print(f"Overall success rate: {overall_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_prompt_cleaning()
