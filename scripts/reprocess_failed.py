#!/usr/bin/env python3
"""
Script to reprocess the 81 failed samples with the bug fix.
Only processes samples that failed in the previous run.
Uses multiprocessing for parallel execution.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk, load_dataset, Dataset
from src.converter import HierarchicalConverter
from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple, Dict
from config import settings


def reprocess_single_sample(args: Tuple[int, Dict, str]) -> Tuple[int, Dict]:
    """
    Reprocess a single failed sample.
    
    Args:
        args: Tuple of (index, original_example, original_doctor_text)
        
    Returns:
        Tuple of (index, updated_data)
    """
    idx, original_example, original_doctor_text = args
    
    # Create converter in worker process
    converter = HierarchicalConverter()
    
    # Convert
    hierarchical_doctor = converter.convert(original_doctor_text)
    
    # Return updated data
    updated = {
        'hierarchical_doctor': hierarchical_doctor,
        'original_doctor': original_doctor_text,
        'success': hierarchical_doctor is not None
    }
    
    return idx, updated


def reprocess_failed_samples():
    """
    Reprocess the 81 samples that failed due to the .get() bug.
    """
    print("\n" + "="*60)
    print("Reprocessing Failed Samples (Bug Fix)")
    print("="*60 + "\n")
    
    # Load current dataset
    dataset_path = os.path.join(settings.PROCESSED_DATA_DIR, "hierarchical_dataset")
    current_ds = load_from_disk(dataset_path)
    
    # Load original dataset
    original_ds = load_dataset(
        settings.DATASET_NAME,
        split=settings.DATASET_SPLIT,
        cache_dir=settings.RAW_DATA_DIR
    )
    
    # Find failed indices
    failed_indices = [i for i, ex in enumerate(current_ds) if not ex['_conversion_success']]
    
    print(f"Found {len(failed_indices)} failed samples to reprocess")
    print()
    
    if not failed_indices:
        print("✓ No failed samples to reprocess!")
        return
    
    # Convert current dataset to list for modification
    data_list = [{
        'Description': ex['Description'],
        'Doctor': ex['Doctor'],
        'Patient': ex['Patient'],
        'Status': ex['Status'],
        '_original_doctor': ex['_original_doctor'],
        '_conversion_success': ex['_conversion_success']
    } for ex in current_ds]
    
    # Prepare arguments for multiprocessing
    args_list = [
        (idx, original_ds[idx], original_ds[idx]['Doctor'])
        for idx in failed_indices
    ]
    
    print(f"Using {settings.MAX_WORKERS} workers for parallel reprocessing...\n")
    
    # Reprocess with multiprocessing
    success_count = 0
    with Pool(processes=settings.MAX_WORKERS) as pool:
        with tqdm(total=len(failed_indices), desc="Reprocessing", ncols=80) as pbar:
            for idx, updated in pool.imap_unordered(reprocess_single_sample, args_list):
                # Update in list
                if updated['success']:
                    data_list[idx]['Doctor'] = updated['hierarchical_doctor']
                    data_list[idx]['_original_doctor'] = updated['original_doctor']
                    data_list[idx]['_conversion_success'] = True
                    success_count += 1
                else:
                    # Keep original if still fails
                    data_list[idx]['Doctor'] = updated['original_doctor'] if updated['original_doctor'] else None
                    data_list[idx]['_original_doctor'] = updated['original_doctor']
                    data_list[idx]['_conversion_success'] = False
                
                pbar.update(1)
                pbar.set_postfix({'success': f'{success_count}/{pbar.n}'})
    
    # Create new dataset
    new_ds = Dataset.from_dict({
        'Description': [d['Description'] for d in data_list],
        'Doctor': [d['Doctor'] for d in data_list],
        'Patient': [d['Patient'] for d in data_list],
        'Status': [d['Status'] for d in data_list],
        '_original_doctor': [d['_original_doctor'] for d in data_list],
        '_conversion_success': [d['_conversion_success'] for d in data_list],
    })
    
    # Save updated dataset
    new_ds.save_to_disk(dataset_path)
    
    # Print statistics
    still_failed = len(failed_indices) - success_count
    print(f"\n{'='*60}")
    print(f"✓ Reprocessing complete!")
    print(f"{'='*60}")
    print(f"Newly successful:     {success_count}/{len(failed_indices)} ({success_count/len(failed_indices)*100:.1f}%)")
    print(f"Still failed:         {still_failed}")
    print(f"Total success rate:   {sum(ex['_conversion_success'] for ex in new_ds)}/{len(new_ds)} ({sum(ex['_conversion_success'] for ex in new_ds)/len(new_ds)*100:.2f}%)")
    print(f"Saved to:             {dataset_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Check API key
    if not settings.DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY not found!")
        sys.exit(1)
    
    reprocess_failed_samples()

