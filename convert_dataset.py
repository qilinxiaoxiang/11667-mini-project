"""
Multi-process dataset conversion script.
Converts doctor responses from Patient-Doctor-Conversation dataset into hierarchical format.
"""
import os
import json
import time
from multiprocessing import Pool, Manager, Lock
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import config


class HierarchicalConverter:
    """Handles conversion of medical responses to hierarchical format using DeepSeek API."""
    
    def __init__(self):
        """Initialize the converter with API client."""
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )
    
    def convert_to_hierarchical(self, text: str, max_retries: int = config.MAX_RETRIES) -> Optional[str]:
        """
        Convert a single doctor response to hierarchical format.
        
        Args:
            text: Original doctor response text
            max_retries: Maximum number of retry attempts
            
        Returns:
            Hierarchical formatted text or None if conversion fails
        """
        if not text or text == "null" or pd.isna(text):
            return None
            
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": config.SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    timeout=config.TIMEOUT,
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to convert after {max_retries} attempts: {str(e)[:100]}")
                    return None
        return None


def process_single_example(args):
    """
    Process a single example from the dataset.
    
    Args:
        args: Tuple of (index, example, converter)
        
    Returns:
        Tuple of (index, processed_example)
    """
    idx, example, _ = args
    converter = HierarchicalConverter()  # Create converter in worker process
    
    original_doctor = example.get('Doctor')
    hierarchical_doctor = converter.convert_to_hierarchical(original_doctor)
    
    processed = {
        'index': idx,
        'Description': example.get('Description'),
        'Patient': example.get('Patient'),
        'Doctor_Original': original_doctor,
        'Doctor_Hierarchical': hierarchical_doctor,
        'Status': example.get('Status'),
        'conversion_success': hierarchical_doctor is not None
    }
    
    return idx, processed


def save_checkpoint(results: List[Dict], output_path: str):
    """Save intermediate results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def convert_dataset(
    dataset_name: str = config.DATASET_NAME,
    split: str = config.DATASET_SPLIT,
    output_dir: str = config.OUTPUT_DIR,
    max_workers: int = config.MAX_WORKERS,
    batch_size: int = config.BATCH_SIZE
):
    """
    Convert entire dataset using multiprocessing.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use
        output_dir: Directory to save processed data
        max_workers: Number of parallel workers
        batch_size: Save checkpoint every N samples
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    total_examples = len(dataset)
    print(f"Total examples: {total_examples}")
    
    # Prepare arguments for multiprocessing
    converter = HierarchicalConverter()  # Dummy for argument preparation
    args_list = [(idx, example, converter) for idx, example in enumerate(dataset)]
    
    # Process with multiprocessing
    results = []
    output_path = os.path.join(output_dir, config.OUTPUT_FILENAME)
    
    print(f"Starting conversion with {max_workers} workers...")
    with Pool(processes=max_workers) as pool:
        with tqdm(total=total_examples, desc="Converting") as pbar:
            for idx, result in pool.imap_unordered(process_single_example, args_list):
                results.append(result)
                pbar.update(1)
                
                # Save checkpoint
                if len(results) % batch_size == 0:
                    save_checkpoint(results, output_path)
                    pbar.set_postfix({
                        'saved': len(results),
                        'success_rate': f"{sum(r['conversion_success'] for r in results) / len(results) * 100:.1f}%"
                    })
    
    # Sort results by index to maintain original order
    results.sort(key=lambda x: x['index'])
    
    # Final save
    save_checkpoint(results, output_path)
    
    # Print statistics
    success_count = sum(r['conversion_success'] for r in results)
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Total processed: {len(results)}")
    print(f"Successful conversions: {success_count} ({success_count/len(results)*100:.2f}%)")
    print(f"Failed conversions: {len(results) - success_count}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*50}")
    
    return results


def convert_to_huggingface_format(results: List[Dict], output_dir: str):
    """
    Convert processed results to HuggingFace datasets format for fine-tuning.
    
    Args:
        results: List of processed examples
        output_dir: Directory to save the dataset
    """
    from datasets import Dataset
    
    # Filter successful conversions
    successful_results = [r for r in results if r['conversion_success']]
    
    # Prepare data for HuggingFace format
    hf_data = {
        'Description': [r['Description'] for r in successful_results],
        'Patient': [r['Patient'] for r in successful_results],
        'Doctor_Original': [r['Doctor_Original'] for r in successful_results],
        'Doctor_Hierarchical': [r['Doctor_Hierarchical'] for r in successful_results],
        'Status': [r['Status'] for r in successful_results],
    }
    
    # Create dataset
    dataset = Dataset.from_dict(hf_data)
    
    # Save in HuggingFace format
    hf_output_dir = os.path.join(output_dir, "hf_dataset")
    dataset.save_to_disk(hf_output_dir)
    
    print(f"\nHuggingFace dataset saved to: {hf_output_dir}")
    print(f"Total examples in HF dataset: {len(dataset)}")
    
    return dataset


if __name__ == "__main__":
    import pandas as pd
    
    # Check API key
    if not config.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables!")
    
    print("Starting hierarchical dataset conversion...")
    print(f"Configuration:")
    print(f"  - Workers: {config.MAX_WORKERS}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Output: {config.OUTPUT_DIR}")
    print()
    
    # Convert dataset
    results = convert_dataset()
    
    # Convert to HuggingFace format for fine-tuning
    convert_to_huggingface_format(results, config.OUTPUT_DIR)
    
    print("\nAll done! ✅")

