#!/usr/bin/env python3
"""
Script to inspect conversion quality from checkpoint or final dataset.
"""
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk, load_dataset
from config import settings


def inspect_quality(dataset_path: str, num_samples: int = 5):
    """
    Inspect conversion quality by comparing original and converted responses.
    
    Args:
        dataset_path: Path to dataset (checkpoint or final)
        num_samples: Number of random samples to inspect
    """
    print("\n" + "="*70)
    print("Conversion Quality Inspection")
    print("="*70 + "\n")
    
    # Load dataset
    try:
        ds = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Check for tracking fields
    has_tracking = '_conversion_success' in ds.features
    
    print(f"Dataset: {dataset_path}")
    print(f"Total samples: {len(ds)}")
    print(f"Has tracking fields: {has_tracking}")
    print()
    
    if has_tracking:
        success_count = sum(ex['_conversion_success'] for ex in ds)
        fail_count = len(ds) - success_count
        print(f"Successful conversions: {success_count} ({success_count/len(ds)*100:.1f}%)")
        print(f"Failed conversions: {fail_count} ({fail_count/len(ds)*100:.1f}%)")
        print()
        
        # Show failed indices
        if fail_count > 0:
            failed_indices = [i for i, ex in enumerate(ds) if not ex['_conversion_success']]
            print(f"Failed indices (first 20): {failed_indices[:20]}")
            print()
    
    # Random sampling
    print(f"Random sample inspection ({num_samples} samples):")
    print("="*70)
    
    indices = random.sample(range(len(ds)), min(num_samples, len(ds)))
    
    for i, idx in enumerate(indices, 1):
        sample = ds[idx]
        
        print(f"\n【Sample {i} - Index {idx}】")
        print("-"*70)
        print(f"Description: {sample['Description']}")
        print(f"Status: {sample['Status']}")
        
        if has_tracking:
            print(f"Conversion Success: {sample['_conversion_success']}")
        
        print()
        
        # Show original (if available)
        if has_tracking and '_original_doctor' in ds.features:
            orig_doc = sample['_original_doctor']
            print("Original Doctor Response:")
            if orig_doc and str(orig_doc) != 'null' and str(orig_doc) != 'None':
                print(orig_doc[:200] + ('...' if len(str(orig_doc)) > 200 else ''))
            else:
                print("[NULL/EMPTY in original dataset]")
            print()
        
        # Show converted
        conv_doc = sample['Doctor']
        print("Hierarchical Doctor Response:")
        if conv_doc and str(conv_doc) != 'None':
            print(conv_doc[:400] + ('...' if len(str(conv_doc)) > 400 else ''))
        else:
            print("[NULL/FAILED CONVERSION]")
        
        # Quality checks
        if has_tracking and sample['_conversion_success']:
            print()
            print("Quality Indicators:")
            # Check hierarchical structure
            has_main_level = '\n-' in str(conv_doc)
            has_sub_level = '\n  -' in str(conv_doc)
            has_third_level = '\n    -' in str(conv_doc)
            main_count = str(conv_doc).count('\n-')
            
            print(f"  ✓ Level 1 (main points): {'Yes' if has_main_level else 'No'} ({main_count} points)")
            print(f"  ✓ Level 2 (sub-points): {'Yes' if has_sub_level else 'No'}")
            print(f"  ✓ Level 3 (details): {'Yes' if has_third_level else 'No'}")
            
            # Check MECE criteria (basic)
            if main_count > 5:
                print(f"  ⚠ Warning: {main_count} main points (>5, violates MECE principle)")
        
        print()
    
    print("="*70)
    print("Inspection complete!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect conversion quality')
    parser.add_argument('--checkpoint', action='store_true', 
                       help='Inspect checkpoint instead of final dataset')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to inspect (default: 5)')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        dataset_path = os.path.join(settings.PROCESSED_DATA_DIR, "hierarchical_dataset.checkpoint")
    else:
        dataset_path = os.path.join(settings.PROCESSED_DATA_DIR, "hierarchical_dataset")
    
    inspect_quality(dataset_path, num_samples=args.samples)

