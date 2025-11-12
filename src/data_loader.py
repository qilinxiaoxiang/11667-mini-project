"""
Data loader for downloading and loading the Patient-Doctor-Conversation dataset.
"""
import os
from datasets import load_dataset
from config import settings


class DatasetLoader:
    """Handles loading and caching of the medical conversation dataset."""
    
    def __init__(self, cache_dir=None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded dataset
        """
        self.cache_dir = cache_dir or settings.RAW_DATA_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load(self, dataset_name=None, split=None, streaming=False):
        """
        Load the dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset (default from config)
            split: Dataset split to load (default from config)
            streaming: Whether to stream the dataset instead of downloading
            
        Returns:
            Loaded dataset object
        """
        dataset_name = dataset_name or settings.DATASET_NAME
        split = split or settings.DATASET_SPLIT
        
        print(f"Loading dataset: {dataset_name} (split: {split})")
        print(f"Cache directory: {self.cache_dir}")
        
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=self.cache_dir,
            streaming=streaming
        )
        
        if not streaming:
            print(f"✓ Loaded {len(dataset)} examples")
            
        return dataset
    
    def get_subset(self, dataset, n=16):
        """
        Get a subset of the dataset for testing.
        
        Args:
            dataset: Full dataset
            n: Number of examples to return
            
        Returns:
            Subset of the dataset
        """
        return dataset.select(range(min(n, len(dataset))))

