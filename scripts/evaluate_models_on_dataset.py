#!/usr/bin/env python3
"""
Evaluate all 4 fine-tuned models on the evaluation dataset.

Models:
- Qwen2.5-0.5B (Full fine-tuning)
- Qwen2.5-0.5B (LoRA)
- DeepSeek-R1-distill-qwen-1.5B (Full fine-tuning)
- DeepSeek-R1-distill-qwen-1.5B (LoRA)

Evaluation dataset: data/processed/evaluation_dataset_hierarchical

Metrics:
- String similarity, ROUGE-L, BLEU
- MECE compliance, hierarchy depth, structure quality
"""

import sys
import os
import json
import torch
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from src.evaluation import eval_pair, check_compliance
import numpy as np


class ModelEvaluator:
    """Evaluate fine-tuned models on evaluation dataset."""

    def __init__(self):
        self.device, self.dtype = self._get_device_and_dtype()
        self.models_config = {
            "qwen_full": {"model_type": "qwen", "training_type": "full"},
            "qwen_lora": {"model_type": "qwen", "training_type": "lora"},
            "deepseek_full": {"model_type": "deepseek", "training_type": "full"},
            "deepseek_lora": {"model_type": "deepseek", "training_type": "lora"},
        }

    @staticmethod
    def _get_device_and_dtype():
        """Determine device and dtype based on available hardware."""
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32
        return device, dtype

    def get_model_path(self, model_key: str) -> str:
        """Get path to fine-tuned model."""
        project_root = Path(__file__).parent.parent
        outputs_dir = project_root / "outputs"
        model_path = outputs_dir / f"{model_key}_final"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        return str(model_path)

    def load_model_and_tokenizer(self, model_key: str):
        """Load fine-tuned model and tokenizer."""
        model_path = self.get_model_path(model_key)
        training_type = self.models_config[model_key]["training_type"]

        if training_type == "lora":
            if self.device == "mps":
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path, dtype=self.dtype
                )
                model = model.to(self.device)
            else:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path, dtype=self.dtype, device_map="auto"
                )
        else:  # full
            if self.device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, dtype=self.dtype
                )
                model = model.to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, dtype=self.dtype, device_map="auto"
                )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from model."""
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        model.eval()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_model_on_dataset(
        self, model_key: str, dataset, max_samples: int = None
    ) -> Dict:
        """Evaluate a single model on the dataset."""
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_key.upper()}")
        print(f"{'='*80}")

        # Load model
        try:
            print(f"Loading model...")
            model, tokenizer = self.load_model_and_tokenizer(model_key)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None

        # Prepare dataset
        num_samples = min(max_samples, len(dataset)) if max_samples else len(dataset)
        print(f"Evaluating on {num_samples} samples...\n")

        results = []
        total_generation_time = 0

        for idx in range(num_samples):
            sample = dataset[idx]

            # Create prompt
            prompt = f"""Medical Question: {sample['Description']}

Patient Background: {sample['Patient']}

Doctor Response:"""

            # Generate response
            try:
                start_time = time.time()
                generated = self.generate_response(model, tokenizer, prompt)
                generation_time = time.time() - start_time
                total_generation_time += generation_time

                # Extract doctor response (remove prompt)
                doctor_response = generated.split("Doctor Response:")[-1].strip()

                # Evaluate
                reference = sample["Doctor"]
                metrics = eval_pair(
                    pred=doctor_response,
                    ref=reference,
                    min_depth=3,
                    max_points_per_parent=5,
                    verbose=False,
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

                # Progress indicator
                if (idx + 1) % max(1, num_samples // 10) == 0:
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

        # Compute statistics
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
            metric_names = [
                "string_similarity",
                "rouge_l",
                "bleu",
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
                values = [r.get(metric_name, 0) for r in successful_results if metric_name in r]
                if values:
                    stats[f"{metric_name}_mean"] = float(np.mean(values))
                    stats[f"{metric_name}_std"] = float(np.std(values))
                    stats[f"{metric_name}_min"] = float(np.min(values))
                    stats[f"{metric_name}_max"] = float(np.max(values))

            # MECE compliance rate
            compliant_count = sum(1 for r in successful_results if r.get("mece_compliant", 0) == 1.0)
            stats["mece_compliant_rate"] = (
                compliant_count / len(successful_results) if successful_results else 0
            )

            # By severity
            severity_stats = {}
            for severity in ["low severity", "medium severity", "high severity"]:
                severity_results = [r for r in successful_results if r.get("status", "").lower() == severity.lower()]
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

        return stats

    def print_summary(self, all_stats: Dict[str, Dict]):
        """Print summary comparison of all models."""
        print("\n" + "="*100)
        print("EVALUATION SUMMARY - ALL MODELS")
        print("="*100)

        # Table header
        print(f"\n{'Model':<20} {'Success':<12} {'MECE Score':<15} {'ROUGE-L':<12} {'BLEU':<12} {'Depth':<10}")
        print("-" * 100)

        # Table rows
        for model_key in sorted(all_stats.keys()):
            stats = all_stats[model_key]
            if stats is None:
                print(f"{model_key:<20} FAILED")
                continue

            success_rate = stats.get("success_rate", 0)
            mece_score = stats.get("mece_score_mean", 0)
            rouge_l = stats.get("rouge_l_mean", 0)
            bleu = stats.get("bleu_mean", 0)
            depth = stats.get("max_depth_mean", 0)

            print(
                f"{model_key:<20} {success_rate:>6.1f}%      "
                f"{mece_score:>7.4f}         {rouge_l:>7.4f}     {bleu:>7.4f}   {depth:>7.2f}"
            )

        # Detailed stats per model
        print(f"\n{'='*100}")
        print("DETAILED STATISTICS BY MODEL")
        print(f"{'='*100}")

        for model_key in sorted(all_stats.keys()):
            stats = all_stats[model_key]
            if stats is None:
                print(f"\n{model_key}: FAILED TO EVALUATE")
                continue

            print(f"\n{model_key.upper()}")
            print("-" * 100)
            print(f"  Total samples:           {stats['total_samples']}")
            print(f"  Successful:              {stats['successful_samples']}")
            print(f"  Failed:                  {stats['failed_samples']}")
            print(f"  Success rate:            {stats['success_rate']:.1f}%")

            if stats["successful_samples"] > 0:
                print(f"\n  Text Similarity Metrics:")
                print(f"    String Similarity:     {stats.get('string_similarity_mean', 0):.4f} ± {stats.get('string_similarity_std', 0):.4f}")
                print(f"    ROUGE-L:               {stats.get('rouge_l_mean', 0):.4f} ± {stats.get('rouge_l_std', 0):.4f}")
                print(f"    BLEU:                  {stats.get('bleu_mean', 0):.4f} ± {stats.get('bleu_std', 0):.4f}")

                print(f"\n  Structure Metrics:")
                print(f"    MECE Score:            {stats.get('mece_score_mean', 0):.4f} ± {stats.get('mece_score_std', 0):.4f}")
                print(f"    MECE Compliant Rate:   {stats.get('mece_compliant_rate', 0)*100:.1f}%")
                print(f"    Depth Score:           {stats.get('depth_score_mean', 0):.4f} ± {stats.get('depth_score_std', 0):.4f}")
                print(f"    Constraint Score:      {stats.get('constraint_score_mean', 0):.4f} ± {stats.get('constraint_score_std', 0):.4f}")

                print(f"\n  Hierarchy Depth:")
                print(f"    Average:               {stats.get('max_depth_mean', 0):.2f} ± {stats.get('max_depth_std', 0):.2f}")
                print(f"    Range:                 [{stats.get('max_depth_min', 0):.0f}, {stats.get('max_depth_max', 0):.0f}]")

                if stats.get("by_severity"):
                    print(f"\n  By Severity:")
                    for severity, sev_stats in stats["by_severity"].items():
                        print(f"    {severity}:")
                        print(f"      Count: {sev_stats['count']}")
                        print(f"      MECE Score: {sev_stats['mece_score_mean']:.4f}")
                        print(f"      ROUGE-L: {sev_stats['rouge_l_mean']:.4f}")

        print(f"\n{'='*100}\n")


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all fine-tuned models on evaluation dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate per model (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen_full", "qwen_lora", "deepseek_full", "deepseek_lora"],
        default=["qwen_full", "qwen_lora", "deepseek_full", "deepseek_lora"],
        help="Models to evaluate",
    )

    args = parser.parse_args()

    # Load evaluation dataset
    print("="*80)
    print("FINE-TUNED MODELS EVALUATION")
    print("="*80)
    print(f"\nLoading evaluation dataset...")

    dataset = load_from_disk("data/processed/evaluation_dataset_hierarchical")
    print(f"✓ Loaded {len(dataset)} evaluation samples\n")

    # Evaluate all models
    evaluator = ModelEvaluator()
    all_stats = {}

    for model_key in args.models:
        stats = evaluator.evaluate_model_on_dataset(model_key, dataset, args.max_samples)
        all_stats[model_key] = stats

    # Print summary
    evaluator.print_summary(all_stats)

    # Save results
    if args.output is None:
        args.output = "evaluation_results"

    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output, f"model_evaluation_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"✓ Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
