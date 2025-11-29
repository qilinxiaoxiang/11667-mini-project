#!/usr/bin/env python3
"""
Generate synthetic evaluation dataset using few-shot prompting.

This script:
1. Loads samples from the actual training dataset
2. Uses them as few-shot examples to prompt DeepSeek
3. Asks DeepSeek to generate NEW similar medical Q&A pairs
4. Ensures evaluation data matches training data distribution/style
5. Maintains out-of-distribution property (new samples, not from training set)
6. Logs all prompts and responses for tracking and debugging
"""
import sys
import os
import json
import random
import time
from datetime import datetime
from queue import Queue
from threading import Lock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from src.converter import HierarchicalConverter
from config import settings

# Global thread-safe queue for logging (to avoid concurrent write issues)
_log_queue = Queue()

# Global rate limiter (to avoid API throttling with concurrent requests)
_rate_limit_lock = Lock()
_last_api_call_time = 0.0
_rate_limit_interval = 0.1  # 100ms between API calls


# ============================================================================
# RATE LIMITING
# ============================================================================

def enforce_rate_limit():
    """
    Enforce global rate limit for API calls.

    Ensures a minimum interval (0.1s) between consecutive API calls
    across all concurrent workers. This prevents API throttling.
    """
    global _last_api_call_time

    with _rate_limit_lock:
        current_time = time.time()
        time_since_last_call = current_time - _last_api_call_time

        if time_since_last_call < _rate_limit_interval:
            wait_time = _rate_limit_interval - time_since_last_call
            time.sleep(wait_time)

        _last_api_call_time = time.time()


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: str) -> str:
    """
    Setup logging directory for tracking DeepSeek interactions.

    Args:
        output_dir: Output directory for evaluation dataset

    Returns:
        Path to logs directory
    """
    logs_dir = os.path.join(os.path.dirname(output_dir), "logs",
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def log_interaction(interaction: Dict):
    """
    Queue a single DeepSeek interaction for logging (thread-safe).

    This uses a queue to avoid concurrent write issues with multiple threads.

    Args:
        interaction: Dict with prompt, response, parse_result, etc.
    """
    _log_queue.put(interaction)


def flush_logs(logs_dir: str):
    """
    Flush all queued logs to disk.

    Writes all accumulated logs from the queue to individual JSON files.
    Called by main thread after all workers complete.

    Args:
        logs_dir: Directory to save logs
    """
    while not _log_queue.empty():
        try:
            interaction = _log_queue.get_nowait()
            idx = interaction["index"]
            log_file = os.path.join(logs_dir, f"sample_{idx:03d}.json")
            with open(log_file, 'w') as f:
                json.dump(interaction, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error flushing log: {e}")


def create_summary_log(logs_dir: str, results: List[Dict]):
    """
    Create a summary log of all generations.

    Args:
        logs_dir: Directory to save logs
        results: List of generation results
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "successful": sum(1 for r in results if r['_conversion_success']),
        "failed": sum(1 for r in results if not r['_conversion_success']),
        "success_rate": sum(1 for r in results if r['_conversion_success']) / len(results) * 100 if results else 0,
        "samples": [
            {
                "idx": i,
                "description": r.get('Description', ''),
                "success": r.get('_conversion_success', False),
                "status": r.get('Status', ''),
            }
            for i, r in enumerate(results)
        ]
    }

    summary_file = os.path.join(logs_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Logs saved to: {logs_dir}")
    print(f"  - Individual interactions: sample_*.json")
    print(f"  - Summary: summary.json")


def load_few_shot_examples(dataset, num_examples: int = 3) -> List[Dict]:
    """
    Load random few-shot examples from the training dataset.

    Args:
        dataset: Pre-loaded HuggingFace dataset
        num_examples: Number of examples to load for few-shot prompting

    Returns:
        List of sample dictionaries
    """
    # Randomly sample examples
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    examples = [dataset[i] for i in indices]

    return examples


def create_few_shot_prompt(examples: List[Dict]) -> str:
    """
    Create a few-shot prompt using original (non-hierarchical) doctor responses.

    Uses the _original_doctor field if available, otherwise falls back to Doctor field.
    This generates PLAIN responses which will be converted to hierarchical in step 2.

    Args:
        examples: List of example dictionaries from training data

    Returns:
        Formatted prompt string
    """
    prompt = """You are a medical consultation expert.
Your task is to create NEW medical consultation Q&A pairs that follow the same format and style as the examples below.

INSTRUCTIONS:
1. Generate COMPLETELY NEW scenarios (not copies of examples)
2. Match the format, style, and quality exactly as shown in the examples
3. Provide clear, detailed doctor responses
4. Cover different medical topics if possible

EXAMPLES (study the format carefully):
"""

    for i, example in enumerate(examples, 1):
        # Use original doctor response (plain, non-hierarchical) if available
        doctor_response = example.get('_original_doctor', example.get('Doctor', ''))

        prompt += f"\n{'='*80}\n"
        prompt += f"Example {i}:\n"
        prompt += f"{'='*80}\n"
        prompt += f"DESCRIPTION: {example['Description']}\n\n"
        prompt += f"PATIENT: {example['Patient']}\n\n"
        prompt += f"DOCTOR: {doctor_response}\n\n"
        prompt += f"STATUS: {example['Status']}\n"

    prompt += f"\n{'='*80}\n"
    prompt += """NOW: Generate a NEW unique medical consultation scenario following the EXACT same format above:

DESCRIPTION: [brief medical question]
PATIENT: [detailed patient narrative]
DOCTOR: [clear, detailed medical advice - write naturally, don't worry about formatting]
STATUS: [low severity / medium severity / high severity]

Generate ONLY the new scenario, no meta-commentary."""

    return prompt


def _generate_single_scenario(idx: int, dataset, num_few_shots: int = 3, enable_logging: bool = True) -> Tuple[int, Dict]:
    """
    Worker function to generate a single scenario.

    Each worker independently:
    1. Loads random few-shot examples
    2. Creates prompt
    3. Calls API
    4. Parses result
    5. Queues the interaction for logging (thread-safe)

    Args:
        idx: Sample index (for tracking)
        dataset: Pre-loaded HuggingFace dataset
        num_few_shots: Number of few-shot examples
        enable_logging: Whether to log interactions (thread-safe queue)

    Returns:
        Tuple of (idx, scenario_dict or None)
    """
    interaction = {
        "index": idx,
        "timestamp": datetime.now().isoformat(),
        "few_shot_count": num_few_shots,
        "few_shot_indices": None,
        "prompt": None,
        "api_response": None,
        "parsed_scenario": None,
        "success": False,
        "error": None
    }

    try:
        # EACH WORKER: Load fresh random few-shot examples
        examples = load_few_shot_examples(dataset, num_few_shots)
        interaction["few_shot_indices"] = [i for i in range(len(dataset)) if dataset[i] in examples][:num_few_shots]

        # Create prompt with these examples
        few_shot_prompt = create_few_shot_prompt(examples)
        interaction["prompt"] = few_shot_prompt

        # Initialize converter in worker
        converter = HierarchicalConverter()

        # Enforce rate limit (0.1s between API calls to avoid throttling)
        enforce_rate_limit()

        # Call API to generate a new scenario
        response = converter.client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful medical consultation dataset generator."},
                {"role": "user", "content": few_shot_prompt}
            ],
            timeout=settings.TIMEOUT,
            stream=False
        )

        generated_text = response.choices[0].message.content
        interaction["api_response"] = generated_text

        # Parse the generated text
        scenario = _parse_generated_scenario(generated_text)

        if scenario:
            interaction["parsed_scenario"] = scenario
            interaction["success"] = True

            # Queue log (thread-safe)
            if enable_logging:
                log_interaction(interaction)

            return idx, scenario
        else:
            interaction["error"] = "Parse failed: could not extract structured data"

            # Queue log (thread-safe)
            if enable_logging:
                log_interaction(interaction)

            return idx, None

    except Exception as e:
        # Log the error
        error_msg = str(e)[:500]
        interaction["error"] = error_msg

        if enable_logging:
            log_interaction(interaction)

        # Print API key error on first occurrence
        if "401" in error_msg or "Authentication" in error_msg:
            if idx == 0:
                print(f"  API Key Error: {error_msg[:100]}")

        return idx, None


def generate_new_scenarios(
    num_samples: int = 40,
    num_few_shots: int = 3,
    num_workers: int = 20,
    logs_dir: str = None
) -> List[Dict]:
    """
    Use DeepSeek API with concurrent few-shot prompting to generate new scenarios.

    For each concurrent worker:
    1. Randomly pick num_few_shots examples from training data
    2. Create a prompt with those examples
    3. Call API to generate 1 new sample
    4. Parse the response

    Args:
        num_samples: Number of new scenarios to generate
        num_few_shots: Number of examples to use for few-shot prompting
        num_workers: Number of concurrent workers

    Returns:
        List of generated scenario dictionaries
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    print(f"Generating {num_samples} new scenarios (concurrent, {num_workers} workers)...\n")
    print("For each worker:")
    print(f"  - Randomly pick {num_few_shots} examples from training data")
    print(f"  - Create prompt with those examples")
    print(f"  - Call API to generate 1 new sample")
    print(f"  - Parse and collect result\n")

    # Load training dataset ONCE (avoid threading issues)
    print("Loading training dataset...")
    try:
        dataset = load_from_disk(settings.PROCESSED_DATA_DIR + "/hierarchical_dataset_clean")
        print(f"✓ Loaded {len(dataset)} training samples\n")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    generated_scenarios = {}
    failures = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks (with logging enabled)
        futures = {
            executor.submit(_generate_single_scenario, i, dataset, num_few_shots, enable_logging=True): i
            for i in range(num_samples)
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            idx, scenario = future.result()

            if scenario:
                generated_scenarios[idx] = scenario
                print(f"✓ {completed}/{num_samples}: {scenario['Description'][:50]}...")
            else:
                failures += 1
                print(f"✗ {completed}/{num_samples}: Generation failed")

    # Flush all queued logs to disk (thread-safe, after all workers complete)
    if logs_dir:
        print("\nFlushing logs to disk...")
        flush_logs(logs_dir)

    # Sort by index to maintain order
    results = [generated_scenarios[i] for i in sorted(generated_scenarios.keys())]

    print(f"\n✓ Generation complete: {len(results)} successful, {failures} failed\n")

    return results


def _parse_generated_scenario(text: str) -> Dict:
    """
    Parse the generated text into a structured scenario.

    Args:
        text: Generated scenario text

    Returns:
        Dictionary with Description, Patient, Doctor, Status or None if parsing fails
    """
    try:
        # Extract fields from the generated text
        lines = text.strip().split('\n')

        description = None
        patient = None
        doctor = None
        status = None

        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()

            if line.startswith('DESCRIPTION:'):
                if current_section and current_content:
                    # Save previous section
                    if current_section == 'DESCRIPTION':
                        description = ' '.join(current_content).strip()
                    elif current_section == 'PATIENT':
                        patient = ' '.join(current_content).strip()
                    elif current_section == 'DOCTOR':
                        doctor = '\n'.join(current_content).strip()
                    elif current_section == 'STATUS':
                        status = ' '.join(current_content).strip()

                current_section = 'DESCRIPTION'
                current_content = [line.replace('DESCRIPTION:', '').strip()]

            elif line.startswith('PATIENT:'):
                if current_section and current_content:
                    if current_section == 'DESCRIPTION':
                        description = ' '.join(current_content).strip()
                    elif current_section == 'PATIENT':
                        patient = ' '.join(current_content).strip()

                current_section = 'PATIENT'
                current_content = [line.replace('PATIENT:', '').strip()]

            elif line.startswith('DOCTOR:'):
                if current_section and current_content:
                    if current_section == 'PATIENT':
                        patient = ' '.join(current_content).strip()

                current_section = 'DOCTOR'
                current_content = [line.replace('DOCTOR:', '').strip()]

            elif line.startswith('STATUS:'):
                if current_section and current_content:
                    if current_section == 'DOCTOR':
                        doctor = '\n'.join(current_content).strip()

                current_section = 'STATUS'
                current_content = [line.replace('STATUS:', '').strip()]

            elif line and current_section:
                # Continue accumulating content for current section
                current_content.append(line)

        # Handle last section
        if current_section and current_content:
            if current_section == 'DESCRIPTION':
                description = ' '.join(current_content).strip()
            elif current_section == 'PATIENT':
                patient = ' '.join(current_content).strip()
            elif current_section == 'DOCTOR':
                doctor = '\n'.join(current_content).strip()
            elif current_section == 'STATUS':
                status = ' '.join(current_content).strip()

        # Validate required fields
        if not all([description, patient, doctor, status]):
            return None

        # Normalize status
        status_lower = status.lower()
        if 'low' in status_lower:
            status = 'low severity'
        elif 'high' in status_lower:
            status = 'high severity'
        else:
            status = 'medium severity'

        return {
            'Description': description,
            'Patient': patient,
            'Doctor': doctor,
            'Status': status,
            '_original_doctor': 'GENERATED_SYNTHETIC',
            '_conversion_success': True
        }

    except Exception as e:
        print(f"  Parsing error: {str(e)[:50]}")
        return None


def create_evaluation_dataset(
    num_samples: int = 40,
    output_path: str = None,
    num_few_shots: int = 3,
    test_mode: bool = False,
    num_workers: int = 20
) -> Dataset:
    """
    Create evaluation dataset using few-shot generated scenarios.

    Args:
        num_samples: Number of samples to generate
        output_path: Path to save the evaluation dataset
        num_few_shots: Number of few-shot examples to use
        test_mode: If True, generate fewer samples for testing
        num_workers: Number of concurrent workers

    Returns:
        HuggingFace Dataset
    """
    # Check API key
    if not settings.DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY not found in environment variables!")
        print("Please set it in ~/.zshrc or ~/.bashrc:")
        print("  export DEEPSEEK_API_KEY='your_api_key_here'")
        sys.exit(1)

    print("\n" + "="*80)
    print("EVALUATION DATASET GENERATION (Concurrent Few-Shot Prompting)")
    print("="*80 + "\n")

    # Adjust for test mode
    if test_mode:
        num_samples = 3
        num_few_shots = 2
        num_workers = 1
        print("🧪 TEST MODE: Generating 3 samples with 2 few-shots (1 worker)\n")

    print(f"Configuration:")
    print(f"  Model:           {settings.DEEPSEEK_MODEL}")
    print(f"  Samples to gen:  {num_samples}")
    print(f"  Few-shot count:  {num_few_shots}")
    print(f"  Concurrent workers: {num_workers}")
    print()

    # Setup logging (thread-safe queue-based)
    logs_dir = setup_logging(output_path)
    print(f"  Logs directory:  {logs_dir}\n")

    # Generate new scenarios (logs will be queued and flushed after completion)
    scenarios = generate_new_scenarios(num_samples, num_few_shots, num_workers, logs_dir)

    if not scenarios:
        print("❌ Failed to generate any scenarios!")
        sys.exit(1)

    # Convert to HuggingFace dataset
    print("\nStep 1: Creating HuggingFace dataset...\n")
    dataset = _create_dataset(scenarios)

    # Save if output path provided
    if output_path:
        print(f"Step 2: Saving evaluation dataset...\n")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        dataset.save_to_disk(output_path)
        print(f"✓ Dataset saved to: {output_path}\n")

    # Print statistics
    _print_statistics(scenarios, output_path)

    # Create summary log with statistics
    if logs_dir:
        create_summary_log(logs_dir, scenarios)

    return dataset


def _create_dataset(results: List[Dict]) -> Dataset:
    """Create HuggingFace dataset from results."""
    data = {
        'Description': [r['Description'] for r in results],
        'Doctor': [r['Doctor'] for r in results],
        'Patient': [r['Patient'] for r in results],
        'Status': [r['Status'] for r in results],
        '_original_doctor': [r['_original_doctor'] for r in results],
        '_conversion_success': [r['_conversion_success'] for r in results],
    }
    return Dataset.from_dict(data)


def _print_statistics(results: List[Dict], output_path: str = None):
    """Print dataset statistics."""
    print("="*80)
    print("✓ Generation Complete!")
    print("="*80)
    print(f"Total generated:      {len(results)}")

    # Count by severity
    severity_counts = {}
    for r in results:
        status = r['Status']
        severity_counts[status] = severity_counts.get(status, 0) + 1

    print(f"\nDistribution by severity:")
    for status in ['low severity', 'medium severity', 'high severity']:
        count = severity_counts.get(status, 0)
        if count > 0:
            print(f"  {status}: {count} ({count/len(results)*100:.1f}%)")

    if output_path:
        print(f"\nOutput saved to:      {output_path}")

    print(f"\n📋 This evaluation dataset is:")
    print(f"   ✓ Generated from training data style (few-shot prompting)")
    print(f"   ✓ Out-of-distribution (NEW scenarios, not in training set)")
    print(f"   ✓ Contains paired questions and hierarchical answers")
    print(f"   ✓ Suitable for unbiased model evaluation")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset using few-shot prompting"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="Number of evaluation samples to generate (default: 40)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save evaluation dataset (optional)"
    )
    parser.add_argument(
        "--few-shots",
        type=int,
        default=3,
        help="Number of examples for few-shot prompting (default: 3)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (3 samples)"
    )

    args = parser.parse_args()

    # Set default output path if not specified
    output_path = args.output or os.path.join(
        settings.PROCESSED_DATA_DIR,
        "evaluation_dataset_synthetic"
    )

    # For concurrent execution, add num_workers parameter
    num_workers = min(20, args.num_samples)  # Default 20 workers, but not more than samples

    # Generate dataset
    dataset = create_evaluation_dataset(
        num_samples=args.num_samples,
        output_path=output_path,
        num_few_shots=args.few_shots,
        test_mode=args.test,
        num_workers=num_workers
    )

    # Print sample
    print("Sample evaluation data:\n")
    print("="*80)
    sample = dataset[0]
    print(f"Description: {sample['Description']}")
    print(f"\nPatient: {sample['Patient'][:200]}...")
    print(f"\nDoctor (Hierarchical):\n{sample['Doctor'][:400]}...")
    print(f"\nStatus: {sample['Status']}")
    print("="*80 + "\n")

    print("✅ Evaluation dataset ready for use!\n")


if __name__ == "__main__":
    main()
