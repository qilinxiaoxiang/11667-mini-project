#!/usr/bin/env python3
"""
Script to retry only failed conversions from a previous run.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk, load_dataset, Dataset
from src.converter import HierarchicalConverter
from tqdm import tqdm
from config import settings


def retry_failed_conversions(checkpoint_path: str, output_path: str):
    """
    Retry only the failed conversions from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint with failed conversions
        output_path: Path to save updated dataset
    """
    print("\n" + "="*60)
    print("Retry Failed Conversions")
    print("="*60 + "\n")
    
    # Load checkpoint
    try:
        checkpoint_ds = load_from_disk(checkpoint_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Make sure the checkpoint exists at:", checkpoint_path)
        return
    
    # Check for tracking fields
    if '_conversion_success' not in checkpoint_ds.features:
        print("❌ Error: Checkpoint missing '_conversion_success' field")
        print("This checkpoint was created with an older version.")
        print("Please run full conversion with the updated code.")
        return
    
    # Find failed conversions
    failed_indices = [i for i, ex in enumerate(checkpoint_ds) 
                     if not ex['_conversion_success']]
    
    print(f"Checkpoint analysis:")
    print(f"  Total samples: {len(checkpoint_ds)}")
    print(f"  Successful: {len(checkpoint_ds) - len(failed_indices)}")
    print(f"  Failed: {len(failed_indices)}")
    print()
    
    if not failed_indices:
        print("✓ No failed conversions to retry!")
        return
    
    print(f"Retrying {len(failed_indices)} failed conversions...")
    print()
    
    # Initialize converter
    converter = HierarchicalConverter()
    
    # Retry failed ones
    retry_count = 0
    for idx in tqdm(failed_indices, desc="Retrying"):
        sample = checkpoint_ds[idx]
        original_doctor = sample['_original_doctor']
        
        # Retry conversion
        hierarchical_doctor = converter.convert(original_doctor)
        
        if hierarchical_doctor:
            # Update in place (creates new dataset)
            checkpoint_ds[idx]['Doctor'] = hierarchical_doctor
            checkpoint_ds[idx]['_conversion_success'] = True
            retry_count += 1
    
    # Save updated dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint_ds.save_to_disk(output_path)
    
    print(f"\n✓ Retry complete!")
    print(f"  Newly successful: {retry_count}/{len(failed_indices)}")
    print(f"  Still failed: {len(failed_indices) - retry_count}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    checkpoint_path = os.path.join(
        settings.PROCESSED_DATA_DIR, 
        "hierarchical_dataset.checkpoint"
    )
    output_path = os.path.join(
        settings.PROCESSED_DATA_DIR,
        "hierarchical_dataset"
    )
    
    retry_failed_conversions(checkpoint_path, output_path)

