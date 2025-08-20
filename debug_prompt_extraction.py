#!/usr/bin/env python3
"""
Debug script to understand why some prompts are being skipped during extraction.
"""

import pickle
from pathlib import Path

def debug_prompt_extraction():
    """Debug the prompt extraction process to see what's being skipped."""
    print("="*80)
    print("DEBUGGING PROMPT EXTRACTION")
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
        
        # Sample some examples to see what's happening
        print(f"\n🔍 SAMPLING EXAMPLES TO DEBUG EXTRACTION:")
        
        extracted_count = 0
        skipped_count = 0
        no_conversation_count = 0
        empty_content_count = 0
        structure_issue_count = 0
        
        for i, ex in enumerate(train_data):
            if i % 20000 == 0:
                print(f"Processing {i}/{len(train_data)} examples...")
            
            # Check what fields exist
            has_conv_a = 'conversation_a' in ex and ex['conversation_a']
            has_conv_b = 'conversation_b' in ex and ex['conversation_b']
            has_full_conv = 'full_conversation' in ex and ex['full_conversation']
            
            # Try to extract prompt
            prompt = None
            
            # Method 1: conversation_a
            if has_conv_a:
                conv_a = ex['conversation_a']
                if isinstance(conv_a, list) and len(conv_a) > 0:
                    first_message = conv_a[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            # Method 2: conversation_b
            if not prompt and has_conv_b:
                conv_b = ex['conversation_b']
                if isinstance(conv_b, list) and len(conv_b) > 0:
                    first_message = conv_b[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        content = first_message['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            # Method 3: full_conversation
            if not prompt and has_full_conv:
                full_conv = ex['full_conversation']
                if isinstance(full_conv, list) and len(full_conv) > 0:
                    user_msg = full_conv[0].get('user', {})
                    if 'content' in user_msg:
                        content = user_msg['content']
                        if isinstance(content, list) and len(content) > 0:
                            if 'text' in content[0]:
                                prompt = content[0]['text']
            
            # Count what we found
            if prompt:
                extracted_count += 1
            else:
                skipped_count += 1
                
                # Categorize why it was skipped
                if not (has_conv_a or has_conv_b or has_full_conv):
                    no_conversation_count += 1
                elif has_conv_a or has_conv_b or has_full_conv:
                    # Check structure issues
                    if has_conv_a and ex['conversation_a']:
                        conv = ex['conversation_a']
                        if isinstance(conv, list) and len(conv) > 0:
                            first_msg = conv[0]
                            if isinstance(first_msg, dict):
                                if 'content' not in first_msg:
                                    structure_issue_count += 1
                                elif 'content' in first_msg:
                                    content = first_msg['content']
                                    if not content or (isinstance(content, list) and len(content) == 0):
                                        empty_content_count += 1
                
                # Show first few skipped examples
                if skipped_count <= 5:
                    print(f"\n--- SKIPPED EXAMPLE {skipped_count} ---")
                    print(f"Keys: {list(ex.keys())}")
                    print(f"Has conv_a: {has_conv_a}")
                    print(f"Has conv_b: {has_conv_b}")
                    print(f"Has full_conv: {has_full_conv}")
                    
                    if has_conv_a and ex['conversation_a']:
                        conv = ex['conversation_a']
                        print(f"conv_a type: {type(conv)}")
                        if isinstance(conv, list):
                            print(f"conv_a length: {len(conv)}")
                            if len(conv) > 0:
                                first = conv[0]
                                print(f"First message type: {type(first)}")
                                if isinstance(first, dict):
                                    print(f"First message keys: {list(first.keys())}")
                                    if 'content' in first:
                                        content = first['content']
                                        print(f"Content type: {type(content)}")
                                        if isinstance(content, list):
                                            print(f"Content length: {len(content)}")
                                            if len(content) > 0:
                                                print(f"First content item: {content[0]}")
                    
                    print("-" * 50)
            
            # Stop after processing a reasonable sample
            if i >= 1000:
                break
        
        print(f"\n📊 EXTRACTION DEBUG SUMMARY:")
        print(f"Examples processed: {i+1}")
        print(f"Successfully extracted: {extracted_count}")
        print(f"Skipped: {skipped_count}")
        print(f"No conversation fields: {no_conversation_count}")
        print(f"Empty content: {empty_content_count}")
        print(f"Structure issues: {structure_issue_count}")
        
        # Show success rate
        success_rate = (extracted_count / (i+1)) * 100
        print(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_prompt_extraction()
