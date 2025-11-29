#!/usr/bin/env python3
"""
Modal script to evaluate a SINGLE fine-tuned model on the evaluation dataset.

This script:
1. Loads one fine-tuned model from Modal volume
2. Loads the evaluation dataset from Modal volume
3. Generates responses for all samples
4. Computes evaluation metrics (text similarity, structure, etc.)
5. Saves results to Modal volume

Benefits of Modal:
- Fast GPU inference (A100 for 10-50x speedup vs local CPU)
- Parallel batch processing
- No need to download large models locally
- Automatic results upload to volume

Usage:
    modal run modal_evaluate_single_model.py::evaluate \
        --model qwen_full \
        --max-samples 40

Or specify model path directly:
    modal run modal_evaluate_single_model.py::evaluate \
        --model-path medical-models-volume/qwen_full_20251120_164212/final \
        --max-samples 40
"""

import os
import sys
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import modal
from modal import Volume

# Create Modal app
app = modal.App("medical-model-evaluation")

# Volumes for data and model access
dataset_volume = Volume.from_name("medical-dataset-volume", create_if_missing=True)
model_volume = Volume.from_name("medical-models-volume", create_if_missing=True)

# Docker image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "datasets",
    "peft",
    "numpy",
    "scikit-learn",
)


@app.function(
    image=image,
    gpu="A100",  # A100 for fast inference
    timeout=3600,  # 1 hour timeout
    volumes={
        "/dataset": dataset_volume,
        "/models": model_volume,
    },
)
def evaluate_single_model(
    model_key: str,
    model_path: str = None,
    max_samples: int = None,
    batch_size: int = 4,
) -> Dict:
    """
    Evaluate a single fine-tuned model on the evaluation dataset.

    Args:
        model_key: Model identifier (qwen_full, qwen_lora, deepseek_full, deepseek_lora)
        model_path: Optional custom model path on Modal volume (e.g., medical-models-volume/model_name/final)
        max_samples: Max samples to evaluate (None = all)
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation results and statistics
    """
    # This is needed because we're inside a Modal function
    sys.path.insert(0, "/root")

    print("=" * 80)
    print(f"MODAL EVALUATION: {model_key.upper()}")
    print("=" * 80)

    # Import evaluation modules locally (inside Modal function)
    try:
        # Try to import from standard location first
        from datasets import load_from_disk
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import AutoPeftModelForCausalLM
    except ImportError as e:
        print(f"Import error: {e}")
        return {"error": str(e), "success": False}

    # Step 1: Load evaluation dataset
    print("\nStep 1: Loading evaluation dataset...")
    try:
        dataset_path = "/dataset/evaluation_dataset_hierarchical"
        if not os.path.exists(dataset_path):
            return {
                "error": f"Dataset not found at {dataset_path}",
                "success": False,
            }
        dataset = load_from_disk(dataset_path)
        print(f"✓ Loaded {len(dataset)} evaluation samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return {"error": f"Dataset load failed: {str(e)}", "success": False}

    # Step 2: Load model
    print(f"\nStep 2: Loading model: {model_key}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        # Determine model path
        if model_path:
            # Custom path provided
            full_model_path = f"/models/{model_path}"
        else:
            # Use standard naming convention
            full_model_path = f"/models/{model_key}_*/final"
            # Find the actual directory (glob pattern)
            import glob
            matches = glob.glob(full_model_path)
            if not matches:
                return {
                    "error": f"No model found matching {full_model_path}",
                    "success": False,
                }
            full_model_path = matches[0]

        if not os.path.exists(full_model_path):
            return {
                "error": f"Model not found at {full_model_path}",
                "success": False,
            }

        print(f"Loading from: {full_model_path}")

        # Determine if LoRA or full
        is_lora = "lora" in model_key.lower()

        if is_lora:
            model = AutoPeftModelForCausalLM.from_pretrained(
                full_model_path, device_map="auto", torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                full_model_path, device_map="auto", torch_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        print(f"✓ Model loaded successfully")

        # Print memory info
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return {
            "error": f"Model load failed: {str(e)}",
            "success": False,
        }

    # Step 3: Generate responses and evaluate
    print(f"\nStep 3: Generating responses and evaluating...")

    # Import evaluation functions
    try:
        # Create a minimal evaluation module inline since imports might fail
        from collections import defaultdict
        import re

        # We'll implement eval_pair inline
        def string_similarity(pred: str, ref: str) -> float:
            """Token-level normalized Levenshtein similarity."""
            a = pred.lower().strip().split()
            b = ref.lower().strip().split()
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0

            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )

            dist = dp[m][n]
            max_len = max(m, n)
            return 1.0 - dist / max_len

        def rouge_l(pred: str, ref: str) -> float:
            """Token-level ROUGE-L F score."""
            hyp = pred.lower().strip().split()
            ref_tokens = ref.lower().strip().split()
            m, n = len(ref_tokens), len(hyp)
            if m == 0 and n == 0:
                return 1.0
            if m == 0 or n == 0:
                return 0.0

            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_tokens[i - 1] == hyp[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            lcs = dp[m][n]
            prec = lcs / n
            rec = lcs / m
            if prec == 0 or rec == 0:
                return 0.0
            return (1 + 1.0) * prec * rec / (rec + 1.0 * prec)

        def check_compliance(
            text: str, min_depth: int = 3, max_points_per_level: int = 5
        ) -> Dict:
            """Check structural compliance."""
            lines = text.strip().split("\n")
            parsed = []

            for original_line in lines:
                stripped_line = original_line.strip()
                if not stripped_line:
                    continue
                match = re.match(r"^(\s*)([-*])\s+(.+)$", original_line)
                if match:
                    leading_spaces = len(match.group(1))
                    level = (leading_spaces // 2) + 1
                    content = match.group(3).strip()
                    parsed.append((level, content))

            if not parsed:
                return {
                    "is_compliant": False,
                    "max_depth": 0,
                    "total_points": 0,
                    "has_structure": False,
                    "violations": ["No valid markdown structure"],
                    "node_child_counts": [],
                }

            # Build tree
            nodes = []
            stack = []

            for level, content in parsed:
                while stack and nodes[stack[-1]]["level"] >= level:
                    stack.pop()

                parent_idx = stack[-1] if stack else None
                node = {
                    "level": level,
                    "content": content,
                    "parent": parent_idx,
                    "child_count": 0,
                }
                idx = len(nodes)
                nodes.append(node)

                if parent_idx is not None:
                    nodes[parent_idx]["child_count"] += 1

                stack.append(idx)

            max_depth = max(n["level"] for n in nodes)
            total_points = len(nodes)

            violations = []
            if max_depth < min_depth:
                violations.append(f"Insufficient depth: {max_depth} < {min_depth}")

            for n in nodes:
                if n["child_count"] > max_points_per_level:
                    violations.append(
                        f"Node has {n['child_count']} children > {max_points_per_level}"
                    )

            is_compliant = max_depth >= min_depth and not violations

            return {
                "is_compliant": is_compliant,
                "max_depth": max_depth,
                "total_points": total_points,
                "has_structure": True,
                "violations": violations,
                "node_child_counts": [
                    {
                        "level": n["level"],
                        "content_preview": n["content"][:50],
                        "child_count": n["child_count"],
                    }
                    for n in nodes
                ],
            }

        def structure_metrics(
            pred: str, min_depth: int = 3, max_points_per_parent: int = 5
        ) -> Dict:
            """Compute structure metrics."""
            comp_pred = check_compliance(pred, min_depth, max_points_per_parent)

            max_depth = comp_pred["max_depth"]
            total_points = comp_pred["total_points"]

            # Depth score
            depth_ok = 1.0 if max_depth >= min_depth else 0.0
            depth_score = (
                min(1.0, max_depth / min_depth) if min_depth > 0 else 1.0
            )

            # Constraint & grouping scores
            parent_nodes = [
                n
                for n in comp_pred["node_child_counts"]
                if n["child_count"] > 0
            ]

            if len(parent_nodes) == 0:
                constraint_score = 0.0
                grouping_score = 0.0
                per_parent_constraint_ok = 1.0
            else:
                compliant_count = sum(
                    1
                    for n in parent_nodes
                    if n["child_count"] <= max_points_per_parent
                )
                constraint_score = compliant_count / len(parent_nodes)
                per_parent_constraint_ok = (
                    1.0 if constraint_score == 1.0 else 0.0
                )

                grouping_sum = 0.0
                for n in parent_nodes:
                    cc = n["child_count"]
                    if cc > max_points_per_parent:
                        score = 0.0
                    elif 3 <= cc <= max_points_per_parent:
                        score = 1.0
                    elif cc == 2:
                        score = 0.9
                    elif cc == 1:
                        score = 0.5
                    else:
                        score = 0.0
                    grouping_sum += score

                grouping_score = grouping_sum / len(parent_nodes)

            mece_compliant = (
                1.0 if (depth_ok == 1.0 and per_parent_constraint_ok == 1.0) else 0.0
            )
            mece_score = (
                (constraint_score * 0.4) + (grouping_score * 0.4) + (depth_score * 0.2)
            )

            return {
                "depth_ok": depth_ok,
                "max_depth": float(max_depth),
                "depth_score": depth_score,
                "per_parent_constraint_ok": per_parent_constraint_ok,
                "constraint_score": constraint_score,
                "grouping_score": grouping_score,
                "total_points": float(total_points),
                "mece_compliant": mece_compliant,
                "mece_score": mece_score,
            }

        def eval_pair(pred: str, ref: str, min_depth: int = 3, max_points_per_parent: int = 5) -> Dict:
            """Evaluate a single (prediction, reference) pair."""
            s_sim = string_similarity(pred, ref)
            r_l = rouge_l(pred, ref)
            s_metrics = structure_metrics(pred, min_depth, max_points_per_parent)

            metrics = {
                "string_similarity": s_sim,
                "rouge_l": r_l,
                **s_metrics,
            }
            return metrics

        print("✓ Evaluation functions initialized")

    except Exception as e:
        print(f"✗ Error initializing evaluation: {e}")
        return {
            "error": f"Evaluation init failed: {str(e)}",
            "success": False,
        }

    # Step 4: Evaluate samples
    num_samples = min(max_samples, len(dataset)) if max_samples else len(dataset)
    print(f"Evaluating on {num_samples} samples...\n")

    results = []
    total_generation_time = 0
    model.eval()

    for idx in range(num_samples):
        sample = dataset[idx]

        # Create prompt
        prompt = f"""Medical Question: {sample['Description']}

Patient Background: {sample['Patient']}

Doctor Response:"""

        try:
            # Generate response
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generation_time = time.time() - start_time
            total_generation_time += generation_time

            # Extract doctor response
            doctor_response = generated.split("Doctor Response:")[-1].strip()

            # Evaluate
            reference = sample["Doctor"]
            metrics = eval_pair(
                pred=doctor_response,
                ref=reference,
                min_depth=3,
                max_points_per_parent=5,
            )

            # Add metadata
            metrics.update(
                {
                    "index": idx,
                    "description": sample["Description"],
                    "status": sample["Status"],
                    "generation_time": generation_time,
                    "generated_length": len(doctor_response),
                    "reference_length": len(reference),
                    "success": True,
                }
            )

            results.append(metrics)

            # Progress
            if (idx + 1) % max(1, num_samples // 10) == 0 or idx == 0:
                avg_time = total_generation_time / (idx + 1)
                print(
                    f"  [{idx+1}/{num_samples}] Generated in {generation_time:.2f}s "
                    f"(avg: {avg_time:.2f}s)"
                )

        except Exception as e:
            print(f"  ✗ Error on sample {idx}: {str(e)[:100]}")
            results.append(
                {
                    "index": idx,
                    "description": sample["Description"],
                    "status": sample["Status"],
                    "success": False,
                    "error": str(e),
                }
            )

    # Step 5: Compute statistics
    print(f"\nStep 4: Computing statistics...")

    successful_results = [r for r in results if r.get("success", False)]
    failed_count = len(results) - len(successful_results)

    stats = {
        "model": model_key,
        "total_samples": len(results),
        "successful_samples": len(successful_results),
        "failed_samples": failed_count,
        "success_rate": (
            len(successful_results) / len(results) * 100 if results else 0
        ),
        "detailed_results": results,
    }

    # Add metric statistics
    if successful_results:
        import numpy as np

        metric_names = [
            "string_similarity",
            "rouge_l",
            "mece_score",
            "mece_compliant",
            "depth_score",
            "constraint_score",
            "grouping_score",
            "max_depth",
            "total_points",
            "generation_time",
        ]

        for metric_name in metric_names:
            values = [
                r.get(metric_name, 0)
                for r in successful_results
                if metric_name in r
            ]
            if values:
                stats[f"{metric_name}_mean"] = float(np.mean(values))
                stats[f"{metric_name}_std"] = float(np.std(values))
                stats[f"{metric_name}_min"] = float(np.min(values))
                stats[f"{metric_name}_max"] = float(np.max(values))

        # MECE compliance rate
        compliant_count = sum(
            1 for r in successful_results if r.get("mece_compliant", 0) == 1.0
        )
        stats["mece_compliant_rate"] = (
            compliant_count / len(successful_results)
            if successful_results
            else 0
        )

        # By severity
        severity_stats = {}
        for severity in ["low severity", "medium severity", "high severity"]:
            severity_results = [
                r
                for r in successful_results
                if r.get("status", "").lower() == severity.lower()
            ]
            if severity_results:
                severity_stats[severity] = {
                    "count": len(severity_results),
                    "mece_score_mean": float(
                        np.mean([r.get("mece_score", 0) for r in severity_results])
                    ),
                    "rouge_l_mean": float(
                        np.mean([r.get("rouge_l", 0) for r in severity_results])
                    ),
                }
        stats["by_severity"] = severity_stats

    # Step 6: Save results
    print(f"\nStep 5: Saving results...")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"model_evaluation_{model_key}_{timestamp}.json"

        # Write to volume
        output_path = f"/models/evaluation_results/{results_file}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Results saved to: {output_path}")
        stats["results_file"] = results_file
        stats["success"] = True

    except Exception as e:
        print(f"✗ Error saving results: {e}")
        stats["error"] = f"Save failed: {str(e)}"
        stats["success"] = False

    return stats


@app.local_entrypoint()
def main(
    model: str,
    model_path: str = None,
    max_samples: int = None,
    batch_size: int = 4,
):
    """Local entrypoint for Modal CLI."""

    print("Starting Modal evaluation...")
    print(f"Model: {model}")
    print(f"Max samples: {max_samples}")
    print()

    # Call the Modal function
    result = evaluate_single_model.remote(
        model_key=model,
        model_path=model_path,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    if result.get("success"):
        print(f"Model: {result['model']}")
        print(f"Total samples: {result['total_samples']}")
        print(f"Successful: {result['successful_samples']}")
        print(f"Failed: {result['failed_samples']}")
        print(f"Success rate: {result['success_rate']:.1f}%")

        if result.get("successful_samples", 0) > 0:
            print(f"\nMetrics:")
            print(f"  MECE Score: {result.get('mece_score_mean', 0):.4f} ± {result.get('mece_score_std', 0):.4f}")
            print(f"  ROUGE-L: {result.get('rouge_l_mean', 0):.4f} ± {result.get('rouge_l_std', 0):.4f}")
            print(f"  Depth Score: {result.get('depth_score_mean', 0):.4f} ± {result.get('depth_score_std', 0):.4f}")
            print(f"  MECE Compliant Rate: {result.get('mece_compliant_rate', 0)*100:.1f}%")
            print(f"  Avg Generation Time: {result.get('generation_time_mean', 0):.2f}s")

        print(f"\nResults saved to: {result.get('results_file', 'unknown')}")
    else:
        print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")

    print("=" * 80)


if __name__ == "__main__":
    main()
