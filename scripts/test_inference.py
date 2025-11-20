"""
Unified inference test script for fine-tuned models (LoRA and Full).
Supports: Qwen2.5-0.5B and DeepSeek-R1-distill-qwen-1.5B
Usage:
  python test_inference.py qwen full
  python test_inference.py qwen lora
  python test_inference.py deepseek full
  python test_inference.py deepseek lora
"""

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM


def get_device_and_dtype():
    """Determine device and dtype based on available hardware"""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # M1 works better with float32
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


def get_model_path(model_type, training_type):
    """Infer model path based on model and training type"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    outputs_dir = project_root / "outputs"

    model_name = f"{model_type}_{training_type}"
    model_path = outputs_dir / f"{model_name}_final"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Expected path: outputs/{model_name}_final/\n"
            f"Please ensure the model is saved there."
        )

    return str(model_path)


def load_model_and_tokenizer(model_path, training_type):
    """Load fine-tuned model and tokenizer based on type"""
    device, dtype = get_device_and_dtype()

    if training_type == "lora":
        # Load LoRA adapter
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
        )
    else:  # full
        # Load full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device, dtype


def get_base_model_info(model_type):
    """Get base model ID"""
    if model_type == "qwen":
        return "Qwen/Qwen2.5-0.5B"
    elif model_type == "deepseek":
        return "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def verify_lora_weights(model, training_type):
    """Verify that LoRA weights are actually loaded (only for LoRA models)"""
    if training_type != "lora":
        return

    print("\n" + "="*60)
    print("VERIFYING LORA ADAPTER")
    print("="*60)

    # Check for LoRA layers in the model
    lora_layers = []
    for name, module in model.named_modules():
        if 'lora' in name.lower():
            lora_layers.append(name)

    print(f"\nLoRA layers found: {len(lora_layers)}")
    if lora_layers:
        print("Sample LoRA layers:")
        for name in lora_layers[:5]:
            print(f"  ✓ {name}")
        if len(lora_layers) > 5:
            print(f"  ... and {len(lora_layers) - 5} more")
    else:
        print("  ✗ NO LoRA layers found!")

    # Extract actual LoRA weight parameters
    lora_weights = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_weights[name] = param.data.clone().detach()

    if lora_weights:
        print(f"\nLoRA weights statistics: {len(lora_weights)} weight matrices")
        all_weights = torch.cat([w.flatten() for w in lora_weights.values()])
        print(f"  Mean value: {all_weights.mean():.6f}")
        print(f"  Std value: {all_weights.std():.6f}")
        print(f"  Min value: {all_weights.min():.6f}")
        print(f"  Max value: {all_weights.max():.6f}")
        print(f"  Non-zero parameters: {(all_weights != 0).sum().item():,} / {all_weights.numel():,}")

        if (all_weights != 0).sum() > 0:
            print("  ✓ LoRA weights are NON-ZERO (model was trained)")
        else:
            print("  ✗ LoRA weights are ALL ZEROS (model was NOT trained)")
    else:
        print("  ✗ No LoRA weight parameters found")


def generate_response(model, tokenizer, device, prompt, max_tokens=512):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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


def main():
    parser = argparse.ArgumentParser(
        description="Test inference for fine-tuned medical response models",
        usage="python test_inference.py <model> <type> [--max-tokens TOKENS]"
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["qwen", "deepseek"],
        help="Base model type (qwen or deepseek)",
    )
    parser.add_argument(
        "type",
        type=str,
        choices=["lora", "full"],
        help="Fine-tuning type (lora or full)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    device, dtype = get_device_and_dtype()
    print(f"Using device: {device} with dtype: {dtype}")

    # Infer model path
    print(f"\n{'='*60}")
    print(f"Loading {args.model.upper()} fine-tuned model ({args.type.upper()})")
    print(f"{'='*60}")

    try:
        model_path = get_model_path(args.model, args.type)
        print(f"Model path: {model_path}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # Load fine-tuned model
    try:
        model, tokenizer, device, dtype = load_model_and_tokenizer(
            model_path, args.type
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Verify LoRA weights if applicable
    verify_lora_weights(model, args.type)

    # Test prompt
    test_prompt = """Medical Question: What are the symptoms of diabetes?
Patient Background: I've been feeling tired lately
Doctor Response:"""

    print(f"\n{'='*60}")
    print("GENERATING RESPONSE")
    print(f"{'='*60}")
    print(f"\nInput prompt:")
    print(test_prompt)

    try:
        response = generate_response(model, tokenizer, device, test_prompt, args.max_tokens)
        print(f"\n{'='*60}")
        print(f"{args.model.upper()} ({args.type.upper()}) - FINE-TUNED MODEL:")
        print(f"{'='*60}")
        print(response)
    except Exception as e:
        print(f"✗ Error generating response: {e}")
        return

    # Load and compare with base model
    print(f"\n{'='*60}")
    print("LOADING BASE MODEL FOR COMPARISON")
    print(f"{'='*60}")

    base_model_id = get_base_model_info(args.model)
    print(f"Base model: {base_model_id}")

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=dtype,
            device_map="auto",
        )
        print("✓ Base model loaded successfully")

        base_response = generate_response(base_model, tokenizer, device, test_prompt, args.max_tokens)
        print(f"\n{'='*60}")
        print(f"{args.model.upper()} - BASE MODEL:")
        print(f"{'='*60}")
        print(base_response)

        # Analysis
        print(f"\n{'='*60}")
        print("COMPARISON ANALYSIS")
        print(f"{'='*60}")

        if response == base_response:
            print("\n⚠ Outputs are IDENTICAL")
            print("  This may indicate the fine-tuning has minimal effect on this prompt.")
        else:
            print("\n✓ Outputs are DIFFERENT")
            print("  Fine-tuned model produces different output than base model.")

            # Check for hierarchical structure (bulleted responses)
            finetuned_response = response.split("Doctor Response:")[-1] if "Doctor Response:" in response else response
            base_response_text = base_response.split("Doctor Response:")[-1] if "Doctor Response:" in base_response else base_response

            finetuned_has_bullets = "- " in finetuned_response or "• " in finetuned_response
            base_has_bullets = "- " in base_response_text or "• " in base_response_text

            print(f"\n  Fine-tuned response has bullets: {finetuned_has_bullets}")
            print(f"  Base response has bullets: {base_has_bullets}")

            if finetuned_has_bullets and not base_has_bullets:
                print("\n  ✓ EVIDENCE: Fine-tuning adds hierarchical structure")
            elif finetuned_has_bullets and base_has_bullets:
                print("\n  ~ Both have hierarchy (may be model default)")
            else:
                print("\n  ~ No clear structural difference detected")

    except Exception as e:
        print(f"✗ Error loading or comparing with base model: {e}")

    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
