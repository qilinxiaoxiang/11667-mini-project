"""
Dataset processor for multi-process conversion.
"""
import os
from multiprocessing import Pool
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from typing import List, Dict, Tuple
from src.converter import HierarchicalConverter
from config import settings


def process_single_example(args: Tuple[int, Dict]) -> Tuple[int, Dict]:
    """
    Process a single example from the dataset (executed in worker process).
    
    Args:
        args: Tuple of (index, example)
        
    Returns:
        Tuple of (index, processed_example)
    """
    idx, example = args
    
    # Create converter in worker process
    converter = HierarchicalConverter()
    
    # Get original doctor response
    original_doctor = example.get('Doctor')
    
    # Convert to hierarchical format
    hierarchical_doctor = converter.convert(original_doctor)
    
    # Prepare processed example (keep same structure as original)
    processed = {
        'Description': example.get('Description'),
        'Doctor': hierarchical_doctor if hierarchical_doctor else original_doctor,  # Use original if conversion fails
        'Patient': example.get('Patient'),
        'Status': example.get('Status'),
        '_original_doctor': original_doctor,  # Keep original for comparison
        '_conversion_success': hierarchical_doctor is not None
    }
    
    return idx, processed


class DatasetProcessor:
    """Handles multi-process conversion of the entire dataset."""
    
    def __init__(self, max_workers=None):
        """
        Initialize the processor.
        
        Args:
            max_workers: Number of parallel workers (default from config)
        """
        self.max_workers = max_workers or settings.MAX_WORKERS
    
    def process(self, dataset, output_path: str, test_mode: bool = None) -> Dataset:
        """
        Process entire dataset using multiprocessing.
        
        Args:
            dataset: Input dataset from HuggingFace
            output_path: Path to save processed dataset
            test_mode: Whether to process only a subset (default from config)
            
        Returns:
            Processed dataset
        """
        test_mode = test_mode if test_mode is not None else settings.TEST_MODE
        
        # Use subset in test mode
        if test_mode:
            dataset = dataset.select(range(min(settings.TEST_SIZE, len(dataset))))
            print(f"\n🧪 TEST MODE: Processing only {len(dataset)} examples")
        
        # Check for existing checkpoint and resume
        checkpoint_path = output_path + ".checkpoint"
        start_idx = 0
        existing_results = []
        
        if settings.RESUME_FROM_CHECKPOINT and os.path.exists(checkpoint_path):
            try:
                checkpoint_ds = load_from_disk(checkpoint_path)
                if '_conversion_success' in checkpoint_ds.features:
                    start_idx = len(checkpoint_ds)
                    # Convert checkpoint back to results format
                    existing_results = [{
                        'Description': ex['Description'],
                        'Doctor': ex['Doctor'],
                        'Patient': ex['Patient'],
                        'Status': ex['Status'],
                        '_original_doctor': ex['_original_doctor'],
                        '_conversion_success': ex['_conversion_success']
                    } for ex in checkpoint_ds]
                    print(f"\n📂 Resuming from checkpoint: {start_idx}/{len(dataset)} already processed")
                    print(f"   Success rate so far: {sum(r['_conversion_success'] for r in existing_results)/len(existing_results)*100:.1f}%")
            except Exception as e:
                print(f"\n⚠ Could not load checkpoint: {e}")
                print("   Starting from beginning...")
        
        if start_idx >= len(dataset):
            print("\n✓ Dataset already fully processed!")
            return load_from_disk(checkpoint_path)
        
        total_examples = len(dataset)
        remaining = total_examples - start_idx
        
        print(f"\n{'='*60}")
        print(f"Processing {total_examples} examples with {self.max_workers} workers")
        if start_idx > 0:
            print(f"Resuming from index {start_idx} ({remaining} remaining)")
        print(f"{'='*60}\n")
        
        # Prepare arguments for multiprocessing (only unprocessed samples)
        args_list = [(idx, example) for idx, example in enumerate(dataset) if idx >= start_idx]
        
        # Process with multiprocessing
        results = []
        
        with Pool(processes=self.max_workers) as pool:
            # Set initial progress from checkpoint
            with tqdm(total=remaining, desc="Converting", ncols=80, initial=0) as pbar:
                for idx, result in pool.imap_unordered(process_single_example, args_list):
                    results.append((idx, result))  # Store idx with result
                    pbar.update(1)
                    
                    # Combine with existing results for stats
                    all_results_so_far = existing_results + [r[1] for r in results]
                    success_count = sum(r['_conversion_success'] for r in all_results_so_far)
                    success_rate = success_count / len(all_results_so_far) * 100
                    pbar.set_postfix({
                        'success': f'{success_rate:.1f}%'
                    })
                    
                    # Save checkpoint (combine with existing)
                    if len(all_results_so_far) % settings.SAVE_INTERVAL == 0:
                        self._save_checkpoint(all_results_so_far, output_path)
        
        # Sort new results by index
        results.sort(key=lambda x: x[0])
        # Extract just the result dicts
        new_results = [r[1] for r in results]
        
        # Combine with existing results
        all_results = existing_results + new_results
        
        # Create HuggingFace dataset
        processed_dataset = self._create_dataset(all_results)
        
        # Save final dataset
        self._save_dataset(processed_dataset, output_path)
        
        # Print statistics
        self._print_statistics(all_results, total_examples, output_path)
        
        return processed_dataset
    
    def _create_dataset(self, results: List[Dict]) -> Dataset:
        """Create HuggingFace dataset from results."""
        # Keep original structure + metadata for tracking
        data = {
            'Description': [r['Description'] for r in results],
            'Doctor': [r['Doctor'] for r in results],
            'Patient': [r['Patient'] for r in results],
            'Status': [r['Status'] for r in results],
            '_original_doctor': [r['_original_doctor'] for r in results],
            '_conversion_success': [r['_conversion_success'] for r in results],
        }
        return Dataset.from_dict(data)
    
    def _save_checkpoint(self, results: List[Dict], output_path: str):
        """Save intermediate checkpoint."""
        checkpoint_data = self._create_dataset(results)
        checkpoint_path = output_path + ".checkpoint"
        checkpoint_data.save_to_disk(checkpoint_path)
    
    def _save_dataset(self, dataset: Dataset, output_path: str):
        """Save final dataset to disk."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save_to_disk(output_path)
    
    def _print_statistics(self, results: List[Dict], total: int, output_path: str):
        """Print conversion statistics."""
        success_count = sum(r['_conversion_success'] for r in results)
        fail_count = total - success_count
        
        # Get failed indices
        failed_indices = [i for i, r in enumerate(results) if not r['_conversion_success']]
        
        print(f"\n{'='*60}")
        print(f"✓ Conversion Complete!")
        print(f"{'='*60}")
        print(f"Total processed:      {total}")
        print(f"Successful:           {success_count} ({success_count/total*100:.2f}%)")
        print(f"Failed (kept orig):   {fail_count} ({fail_count/total*100:.2f}%)")
        
        if failed_indices:
            print(f"\nFailed sample indices (first 50):")
            print(f"  {failed_indices[:50]}")
            if len(failed_indices) > 50:
                print(f"  ... and {len(failed_indices) - 50} more")
        
        print(f"\nOutput saved to:      {output_path}")
        print(f"Metadata fields preserved: _conversion_success, _original_doctor")
        print(f"{'='*60}\n")

