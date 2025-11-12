#!/usr/bin/env python3
"""
Main script to download and convert the Patient-Doctor-Conversation dataset.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DatasetLoader
from src.processor import DatasetProcessor
from config import settings


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Hierarchical Medical Response Conversion")
    print("="*60 + "\n")
    
    # Check API key
    if not settings.DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY not found in environment variables!")
        print("Please set it in ~/.zshrc or ~/.bashrc:")
        print("  export DEEPSEEK_API_KEY='your_api_key_here'")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Dataset:      {settings.DATASET_NAME}")
    print(f"  Split:        {settings.DATASET_SPLIT}")
    print(f"  Workers:      {settings.MAX_WORKERS}")
    print(f"  Test mode:    {settings.TEST_MODE}")
    if settings.TEST_MODE:
        print(f"  Test size:    {settings.TEST_SIZE}")
    print()
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    loader = DatasetLoader()
    dataset = loader.load()
    
    # Step 2: Process dataset
    print("\nStep 2: Converting to hierarchical format...")
    processor = DatasetProcessor()
    output_path = os.path.join(settings.PROCESSED_DATA_DIR, "hierarchical_dataset")
    
    processed_dataset = processor.process(
        dataset=dataset,
        output_path=output_path,
        test_mode=settings.TEST_MODE
    )
    
    # Step 3: Show sample results
    print("Step 3: Sample results\n")
    print("="*60)
    sample = processed_dataset[0]
    print(f"Description: {sample['Description']}")
    print(f"\nPatient: {sample['Patient'][:100]}...")
    print(f"\nDoctor (Hierarchical):\n{sample['Doctor'][:500]}...")
    print("="*60)
    
    print("\n✅ All done!\n")


if __name__ == "__main__":
    main()

