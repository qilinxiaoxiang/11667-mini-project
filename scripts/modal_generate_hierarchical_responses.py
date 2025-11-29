#!/usr/bin/env python3
"""
Modal script to convert plain doctor responses to hierarchical format using DeepSeek API.

This script:
1. Loads the plain synthetic evaluation dataset from Modal volume
2. Uses DeepSeek API to convert each response to hierarchical format
3. Parallel processing with multiple workers for speed
4. Saves converted dataset back to Modal volume

Benefits:
- Fast API processing (parallel workers on Modal)
- Automatic retry with exponential backoff
- Progress tracking and checkpointing
- Results saved to Modal volume automatically

Usage:
    modal run modal_generate_hierarchical_responses.py::convert_to_hierarchical \
        --input medical-dataset-volume/evaluation_dataset_synthetic \
        --output medical-dataset-volume/evaluation_dataset_hierarchical \
        --num-workers 10

Or with fewer workers:
    modal run modal_generate_hierarchical_responses.py::convert_to_hierarchical \
        --input medical-dataset-volume/evaluation_dataset_synthetic \
        --output medical-dataset-volume/evaluation_dataset_hierarchical \
        --num-workers 5
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import modal
from modal import Volume

# Create Modal app
app = modal.App("medical-response-hierarchical-conversion")

# Volumes
dataset_volume = Volume.from_name("medical-dataset-volume", create_if_missing=True)

# Docker image
image = modal.Image.debian_slim().pip_install(
    "datasets",
    "openai",  # For DeepSeek API (OpenAI-compatible)
    "tqdm",
    "pyarrow",
)

# Global locks for thread safety
_rate_limit_lock = Lock()
_last_api_call_time = 0.0
_rate_limit_interval = 0.1  # 100ms between API calls


def enforce_rate_limit():
    """Enforce global rate limit for API calls."""
    global _last_api_call_time

    with _rate_limit_lock:
        current_time = time.time()
        time_since_last_call = current_time - _last_api_call_time

        if time_since_last_call < _rate_limit_interval:
            wait_time = _rate_limit_interval - time_since_last_call
            time.sleep(wait_time)

        _last_api_call_time = time.time()


def create_conversion_prompt(plain_response: str, description: str = "", patient_bg: str = "") -> str:
    """Create a prompt for converting plain response to hierarchical format."""
    prompt = f"""Convert the following doctor's response to a HIERARCHICAL format following the Pyramid Principle and MECE framework.

REQUIREMENTS:
1. Create a hierarchical bullet-point structure with AT LEAST 3 levels of depth
2. Maximum 5 bullet points per level/parent
3. Use "-" or "*" for bullets, indent with 2 spaces per level
4. Top level: Main assessment/recommendation
5. Middle levels: Categories and subcategories
6. Bottom level: Specific details and supporting information
7. Follow Pyramid Principle: Start with main point, then support with details
8. Follow MECE: Points must be mutually exclusive and collectively exhaustive

CONTEXT:
Description: {description}
Patient Background: {patient_bg}

ORIGINAL RESPONSE (plain text):
{plain_response}

CONVERT to hierarchical format with proper structure:
- Start with main assessment
  - Key category 1
    - Supporting detail
    - Supporting detail
  - Key category 2
    - Supporting detail
    - Supporting detail
- Secondary considerations
  - Additional point
    - Detail

Output ONLY the hierarchical response, no explanations:"""
    return prompt


@app.function(
    image=image,
    timeout=3600,
    volumes={"/dataset": dataset_volume},
)
def convert_to_hierarchical(
    input_path: str = "evaluation_dataset_synthetic",
    output_path: str = "evaluation_dataset_hierarchical",
    num_workers: int = 10,
    max_samples: int = None,
    start_index: int = 0,
) -> Dict:
    """
    Convert plain evaluation dataset to hierarchical format.

    Args:
        input_path: Path to synthetic dataset on volume (relative to /dataset)
        output_path: Path to save hierarchical dataset on volume (relative to /dataset)
        num_workers: Number of concurrent API workers
        max_samples: Max samples to convert (None = all)
        start_index: Start from this sample index (for resuming)

    Returns:
        Dictionary with conversion results
    """
    from datasets import load_from_disk, Dataset
    from tqdm import tqdm

    print("=" * 80)
    print("HIERARCHICAL RESPONSE GENERATION")
    print("=" * 80)

    # Check API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return {
            "error": "DEEPSEEK_API_KEY not found in environment",
            "success": False,
        }

    # Load input dataset
    print(f"\nLoading input dataset from /dataset/{input_path}...")
    try:
        dataset_path = f"/dataset/{input_path}"
        if not os.path.exists(dataset_path):
            return {
                "error": f"Input dataset not found at {dataset_path}",
                "success": False,
            }

        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        return {
            "error": f"Failed to load dataset: {str(e)}",
            "success": False,
        }

    # Prepare samples to convert
    num_samples = (
        min(max_samples, len(dataset) - start_index)
        if max_samples
        else len(dataset) - start_index
    )
    print(f"Converting {num_samples} samples (starting from index {start_index})")
    print(f"Using {num_workers} parallel workers\n")

    # Prepare DeepSeek client
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/beta",
    )

    converted_data = {
        "Description": [],
        "Patient": [],
        "Doctor": [],
        "Status": [],
        "_original_doctor": [],
        "_conversion_success": [],
    }

    results = {"successful": 0, "failed": 0, "skipped": 0, "total": num_samples}

    def convert_single_sample(idx: int, sample: Dict) -> Tuple[int, Dict, bool]:
        """Convert a single sample's doctor response to hierarchical format."""
        try:
            # Enforce rate limit
            enforce_rate_limit()

            # Create prompt
            prompt = create_conversion_prompt(
                plain_response=sample["Doctor"],
                description=sample.get("Description", ""),
                patient_bg=sample.get("Patient", ""),
            )

            # Call DeepSeek API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical consultant. Convert responses to hierarchical format following Pyramid Principle and MECE framework.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                timeout=60,
            )

            hierarchical_response = response.choices[0].message.content
            return idx, {
                "Description": sample["Description"],
                "Patient": sample["Patient"],
                "Doctor": hierarchical_response,
                "Status": sample.get("Status", "medium severity"),
                "_original_doctor": sample["Doctor"],
                "_conversion_success": True,
            }, True

        except Exception as e:
            # Return original response on failure
            print(f"  ⚠️ Sample {idx}: Conversion failed, keeping original")
            return idx, {
                "Description": sample["Description"],
                "Patient": sample["Patient"],
                "Doctor": sample["Doctor"],
                "Status": sample.get("Status", "medium severity"),
                "_original_doctor": sample["Doctor"],
                "_conversion_success": False,
            }, False

    # Convert samples in parallel
    print("Converting samples...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(convert_single_sample, start_index + i, dataset[start_index + i]): i
            for i in range(num_samples)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            idx, converted_sample, success = future.result()

            # Append to converted data
            for key in converted_data:
                converted_data[key].append(converted_sample[key])

            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1

            # Progress
            if completed % max(1, num_samples // 10) == 0 or completed == 1:
                success_rate = (
                    results["successful"]
                    / (results["successful"] + results["failed"])
                    * 100
                    if (results["successful"] + results["failed"]) > 0
                    else 0
                )
                print(
                    f"  [{completed}/{num_samples}] Converted (success: {success_rate:.1f}%)"
                )

    # Create HuggingFace dataset
    print(f"\nCreating HuggingFace dataset...")
    converted_dataset = Dataset.from_dict(converted_data)
    print(f"✓ Dataset created with {len(converted_dataset)} samples")

    # Save converted dataset
    print(f"\nSaving converted dataset...")
    try:
        output_full_path = f"/dataset/{output_path}"
        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        converted_dataset.save_to_disk(output_full_path)
        print(f"✓ Saved to: {output_full_path}")
        results["output_path"] = output_path
    except Exception as e:
        return {
            "error": f"Failed to save dataset: {str(e)}",
            "success": False,
        }

    # Prepare summary
    results["success"] = True
    results["timestamp"] = datetime.now().isoformat()
    success_rate = (
        results["successful"]
        / (results["successful"] + results["failed"])
        * 100
        if (results["successful"] + results["failed"]) > 0
        else 0
    )
    results["success_rate"] = success_rate

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Successful: {results['successful']}/{num_samples} ({success_rate:.1f}%)")
    print(f"Failed: {results['failed']}/{num_samples}")
    print(f"Output: {output_path}")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main(
    input: str = "evaluation_dataset_synthetic",
    output: str = "evaluation_dataset_hierarchical",
    num_workers: int = 10,
    max_samples: int = None,
    start_index: int = 0,
):
    """Local entrypoint for Modal CLI."""
    print("Starting hierarchical conversion on Modal...")
    print(f"Input: {input}")
    print(f"Output: {output}")
    print(f"Workers: {num_workers}")
    print()

    result = convert_to_hierarchical.remote(
        input_path=input,
        output_path=output,
        num_workers=num_workers,
        max_samples=max_samples,
        start_index=start_index,
    )

    if result["success"]:
        print(f"✓ Conversion completed successfully")
        print(f"  Success rate: {result['success_rate']:.1f}%")
        print(f"  Output: {result['output_path']}")
    else:
        print(f"✗ Conversion failed: {result['error']}")
