Hierarchical Medical Conversation Dataset
==========================================

Format: HuggingFace Datasets (Apache Arrow)
Total Samples: 3,311
Conversion Success Rate: 99.73%

How to Load:
-----------
from datasets import load_from_disk

dataset = load_from_disk('data/processed/hierarchical_dataset_clean')

# Access samples
sample = dataset[0]
print(sample['Doctor'])  # Hierarchical response

# Batch access
batch = dataset[:10]

Fields:
-------
- Description: Medical question/issue
- Patient: Patient's detailed description
- Doctor: Hierarchical medical response (markdown format, 3+ levels)
- Status: Severity level (low/medium/high)
- _original_doctor: Original response for comparison
- _conversion_success: All True (failed samples removed)

Data Organization:
-----------------
Arrow format handles all data separation automatically.
Each sample is an independent record - no manual separators needed.
Direct compatibility with HuggingFace Trainer.

Quality:
--------
✓ Hierarchical structure: ≥3 levels
✓ Breadth control: ≤5 points per level
✓ Pyramid Principle: Top-down reasoning
✓ MECE structure: Mutually exclusive, collectively exhaustive

Generated using DeepSeek API with custom hierarchical prompts.
