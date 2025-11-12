#!/usr/bin/env python3
"""
Script to download the dataset without processing.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DatasetLoader
from config import settings


def main():
    """Download and cache the dataset."""
    print("\n" + "="*60)
    print("Downloading Patient-Doctor-Conversation Dataset")
    print("="*60 + "\n")
    
    loader = DatasetLoader()
    dataset = loader.load()
    
    print("\n✓ Dataset downloaded and cached successfully!")
    print(f"\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")
    print(f"  Cache location: {settings.RAW_DATA_DIR}")
    
    # Show first example
    print(f"\nFirst example:")
    print("="*60)
    example = dataset[0]
    for key, value in example.items():
        if value and str(value) != "null":
            display_value = str(value)[:100] + "..." if len(str(value)) > 100 else value
            print(f"{key}: {display_value}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

