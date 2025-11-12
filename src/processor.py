"""
Dataset processor for multi-process conversion.
"""
import os
from multiprocessing import Pool
from tqdm import tqdm
from datasets import Dataset
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
        
        total_examples = len(dataset)
        print(f"\n{'='*60}")
        print(f"Processing {total_examples} examples with {self.max_workers} workers")
        print(f"{'='*60}\n")
        
        # Prepare arguments for multiprocessing
        args_list = [(idx, example) for idx, example in enumerate(dataset)]
        
        # Process with multiprocessing
        results = []
        
        with Pool(processes=self.max_workers) as pool:
            with tqdm(total=total_examples, desc="Converting", ncols=80) as pbar:
                for idx, result in pool.imap_unordered(process_single_example, args_list):
                    results.append((idx, result))  # Store idx with result
                    pbar.update(1)
                    
                    # Update progress stats
                    success_count = sum(r[1]['_conversion_success'] for r in results)
                    success_rate = success_count / len(results) * 100
                    pbar.set_postfix({
                        'success': f'{success_rate:.1f}%'
                    })
                    
                    # Save checkpoint
                    if len(results) % settings.SAVE_INTERVAL == 0:
                        # Extract just the result dicts for checkpoint
                        checkpoint_results = [r[1] for r in results]
                        self._save_checkpoint(checkpoint_results, output_path)
        
        # Sort results by index to maintain original order
        results.sort(key=lambda x: x[0])
        # Extract just the result dicts
        results = [r[1] for r in results]
        
        # Create HuggingFace dataset (keep only successful conversions)
        processed_dataset = self._create_dataset(results)
        
        # Save final dataset
        self._save_dataset(processed_dataset, output_path)
        
        # Print statistics
        self._print_statistics(results, total_examples, output_path)
        
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

