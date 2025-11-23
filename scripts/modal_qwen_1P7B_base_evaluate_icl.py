"""
Modal script for evaluating ICL method on test set
- Loads test dataset from volume
- Uses local model loading (not Inference API)
- Saves results to volume
- Supports GPU (A100) or CPU

Usage:
  modal run scripts/modal_evaluate_icl.py::evaluate --model "Qwen/Qwen2.5-0.5B" --max-samples 10
  modal run scripts/modal_evaluate_icl.py::evaluate --model "Qwen/Qwen2.5-1.5B" --max-samples 50
"""

import os
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import modal
from modal import Volume


# ICL Template
ICL_TEMPLATE = """Task: Convert a medical/nutrition question into a MECE hierarchical bullet answer.

You MUST follow the exact output constraints:

=== Output Format (STRICT) ===

- **1. <Top-level section name>**

  - **1.1 <Subsection name>**

    - <bullet>

    - <bullet>

- **2. <Top-level section name>**

  - **2.1 <Subsection name>**

    - <bullet>

...

Rules:

A) Use 5–7 top-level sections if applicable.

B) Subsections must be MECE within each top-level section.

C) No diagnosis; only plausible causes + safe advice.

D) No extra text before or after the hierarchy.

=== Fully-worked Example ===

Input:

"I eat very little, feel weak, and want to gain weight safely. What should I do?"

Output:

- **1. Daily calorie & eating pattern**

  - **1.1 Total intake**

    - Current intake is likely below needs; aim to increase gradually.

    - Avoid extremely low-calorie patterns.

  - **1.2 Meal structure**

    - 3 meals + 1 snack.

    - Add a light meal after protein-rich meals.

  - **1.3 Gradual adjustment**

    - Increase calories slowly over weeks.

    - Appetite may rise with consistent activity.

- **2. Recommended foods**

  - **2.1 High-calorie add-ons**

    - Ghee 2–3 tsp/day if tolerated.

    - Cheese/paneer ~3× weekly.

  - **2.2 Protein sources**

    - Dals/whole pulses daily.

    - Protein supplement post-exercise if needed.

  - **2.3 Drinks & dairy**

    - Milkshakes or lassi for extra calories.

  - **2.4 Produce**

    - Include diverse fruits and vegetables daily.

- **3. Exercise guidance**

  - **3.1 Frequency**

    - Daily or near-daily.

  - **3.2 Duration**

    - 45–60 minutes.

  - **3.3 Type**

    - Brisk walking is sufficient to start.

- **4. Health expectations**

  - **4.1 Appetite**

    - Often improves with regular exercise.

  - **4.2 Weight**

    - Expect gradual gain, not rapid change.

  - **4.3 Immunity**

    - Adequate nutrition supports immunity, but avoid guarantees.

- **5. Psychological & lifestyle advice**

  - **5.1 Mental practices**

    - Meditation or stress reduction if anxiety affects eating.

  - **5.2 Mindset**

    - Focus on overall health, not only weight.

- **6. Follow-up**

  - **6.1 Timeframe**

    - Reassess after ~2 weeks.

=== Now do the same for: ===

Input:

{{medical_question}}

Output:"""


def create_prompt(medical_question: str) -> str:
    """创建 ICL prompt"""
    return ICL_TEMPLATE.replace("{{medical_question}}", medical_question)


def call_local_model(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Optional[str]:
    """
    使用本地加载的模型生成响应
    
    Args:
        model: 本地加载的模型
        tokenizer: 分词器
        device: 设备（cuda/cpu）
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: Nucleus sampling参数
        
    Returns:
        生成的响应文本或 None（如果失败）
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]  # 记录 prompt 的长度
        
        model.eval()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,  # 确保至少生成10个token
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # 避免重复的3-gram
            )
        
        # 只解码生成的部分（从 input_length 开始）
        generated_ids = outputs[0][input_length:]
        
        # 调试：检查生成的 token 数量
        if len(generated_ids) == 0:
            print(f"  ⚠ 生成了0个token（output_length={len(outputs[0])}, input_length={input_length}）")
            return None
        
        # 过滤掉 EOS 和 PAD token
        generated_ids_filtered = [
            token_id for token_id in generated_ids 
            if token_id not in [tokenizer.eos_token_id, tokenizer.pad_token_id]
        ]
        
        if len(generated_ids_filtered) == 0:
            print(f"  ⚠ 生成的所有token都是特殊token（EOS/PAD）")
            return None
        
        generated_text = tokenizer.decode(generated_ids_filtered, skip_special_tokens=True).strip()
        
        # 如果生成的内容为空，返回 None
        if not generated_text:
            # 尝试不解码生成的所有 token 看看
            all_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            print(f"  ⚠ 生成了空内容（原始解码: {all_text[:50]}...）")
            return None
        
        return generated_text
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"  ✗ 本地模型生成失败")
        print(f"     错误类型: {error_type}")
        print(f"     错误信息: {error_msg[:200]}...")  # 只显示前200个字符
        import traceback
        traceback.print_exc()
        return None


def load_local_model(model_name: str, device: str = "cuda"):
    """
    加载本地模型和分词器
    
    Args:
        model_name: 模型名称
        device: 设备（cuda/cpu）
        
    Returns:
        (model, tokenizer, device) 元组
    """
    print(f"  正在加载模型 {model_name} 到设备 {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 设置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确定 dtype
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,  # 使用 dtype 替代 torch_dtype（torch_dtype 已弃用）
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"  ✓ 模型加载成功")
    
    return model, tokenizer, device


def print_memory_info(device):
    """Print GPU memory information"""
    if device.type == "cuda":
        print("\n" + "=" * 60)
        print("GPU Memory Information")
        print("=" * 60)
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Allocated: {allocated:.2f}GB")
        print(f"Reserved: {reserved:.2f}GB")
        print(f"Total GPU: {total:.2f}GB")
        print(f"Available: {(total - reserved):.2f}GB")
        print("=" * 60 + "\n")


def evaluate_icl(
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_samples: Optional[int] = 10,  # 默认只跑10个样本（用于测试），None表示全部
    results_volume_obj=None,
):
    """
    在测试集上评估 ICL 方法
    
    Args:
        model_name: 模型名称
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: Nucleus sampling参数
        max_samples: 最大评估样本数（10为默认测试数量，None表示全部）
        results_volume_obj: Modal volume对象，用于保存结果
        
    Returns:
        评估结果摘要
    """
    # 在 Modal GPU 上总是使用 CUDA
    # 在 Modal 上，如果配置了 GPU，则使用 CUDA
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print("=" * 60)
    print("ICL 方法评估 - 测试集 (Modal)")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"最大样本数: {max_samples or '全部'}")
    
    # 加载测试集
    print("\n加载测试集...")
    dataset_path = "/dataset/hierarchical_dataset_clean"
    
    # 检查数据集是否存在
    import os
    if not os.path.exists(dataset_path):
        error_msg = f"""
✗ 错误: 数据集路径不存在: {dataset_path}

请先上传数据集到 Modal Volume。你可以：

1. 从本地数据集上传：
   modal volume put medical-dataset-volume \\
     ./data/processed/hierarchical_dataset_clean \\
     hierarchical_dataset_clean

2. 或者检查数据集是否在 volume 中的其他路径：
   modal volume ls medical-dataset-volume
"""
        print(error_msg)
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please upload it first.")
    
    # 从完整数据集加载
    print(f"从 {dataset_path} 加载数据集...")
    dataset = load_from_disk(dataset_path)
    print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本")
    
    # 使用相同的划分方式
    print("划分训练集和测试集（test_size=0.1, seed=42）...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = split_dataset["test"]
    
    print(f"✓ 测试集加载成功，共 {len(test_dataset)} 个样本")
    
    # 确定评估样本数
    if max_samples is None:
        num_samples = len(test_dataset)
    else:
        num_samples = min(max_samples, len(test_dataset))
    print(f"\n将评估 {num_samples} 个样本（共 {len(test_dataset)} 个）")
    
    # 加载模型
    print(f"\n{'='*60}")
    print("加载模型")
    print(f"{'='*60}")
    model, tokenizer, device_str = load_local_model(model_name, device_str)
    device = torch.device(device_str)
    
    print_memory_info(device)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/results/icl_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    failed_count = 0
    
    print(f"\n{'='*60}")
    print("开始评估")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        sample = test_dataset[i]
        
        # 构建问题（使用 Description 作为问题）
        question = sample['Description']
        
        print(f"\n[{i+1}/{num_samples}] 处理问题...")
        print(f"问题: {question[:100]}..." if len(question) > 100 else f"问题: {question}")
        
        # 使用 ICL 生成答案
        prompt = create_prompt(question)
        
        start_time = time.time()
        generated_answer = call_local_model(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed_time = time.time() - start_time
        
        if generated_answer:
            print(f"  ✓ 生成成功 (耗时: {elapsed_time:.2f}s)")
            print(f"  答案长度: {len(generated_answer)} 字符")
        else:
            print(f"  ✗ 生成失败")
            failed_count += 1
        
        # 保存结果
        result = {
            "index": i,
            "question": question,
            "patient_description": sample.get('Patient', ''),
            "reference_answer": sample.get('Doctor', ''),
            "generated_answer": generated_answer if generated_answer else "FAILED",
            "status": sample.get('Status', ''),
            "generation_time": elapsed_time,
            "success": generated_answer is not None,
        }
        results.append(result)
        
        # 每10个样本保存一次中间结果
        if (i + 1) % 10 == 0:
            checkpoint_file = os.path.join(output_dir, f"icl_results_checkpoint_{i+1}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  💾 已保存检查点: {checkpoint_file}")
            
            # 提交 volume 更新
            if results_volume_obj:
                results_volume_obj.commit()
    
    # 保存最终结果
    results_file = os.path.join(output_dir, f"icl_results_{timestamp}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("评估完成")
    print(f"{'='*60}")
    print(f"总样本数: {num_samples}")
    print(f"成功: {num_samples - failed_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {(num_samples - failed_count) / num_samples * 100:.1f}%")
    print(f"\n结果已保存到: {results_file}")
    
    # 生成摘要
    summary = {
        "model": model_name,
        "total_samples": num_samples,
        "successful": num_samples - failed_count,
        "failed": failed_count,
        "success_rate": (num_samples - failed_count) / num_samples * 100,
        "average_generation_time": sum(r['generation_time'] for r in results) / len(results) if results else 0,
        "results_file": results_file,
        "timestamp": timestamp,
        "device": device_str,
    }
    
    summary_file = os.path.join(output_dir, f"icl_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"摘要已保存到: {summary_file}")
    
    # 提交 volume 更新
    if results_volume_obj:
        results_volume_obj.commit()
    
    print_memory_info(device)
    
    return summary


# Create Modal app and volumes
app = modal.App("medical-icl-evaluation")

# Create volumes for dataset and results
dataset_volume = Volume.from_name("medical-dataset-volume", create_if_missing=True)
results_volume = Volume.from_name("medical-results-volume", create_if_missing=True)

# Docker image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "datasets",
    "accelerate",
)


@app.function(
    image=image,
    gpu="A10",  # 可以改为 None 使用 CPU，或 "A10G" 使用更便宜的 GPU，或 "T4" 使用更便宜的 GPU
    volumes={
        "/dataset": dataset_volume,
        "/results": results_volume,
    },
    timeout=86400,  # 24小时超时
)
def evaluate(
    model: str = "Qwen/Qwen3-1.7B-Base",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_samples: Optional[int] = 10,  # 默认只跑10个样本（用于测试），可改为 None 跑全部或指定数量
):
    """
    Entrypoint for modal run command
    
    Example:
        # 默认跑10个样本（测试用）
        modal run scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate --model "Qwen/Qwen2.5-0.5B"
        
        # 跑50个样本
        modal run scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate --model "Qwen/Qwen2.5-0.5B" --max-samples 50
        
        # 跑全部332个测试样本
        modal run scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate --model "Qwen/Qwen2.5-0.5B" --max-samples 332
    """
    # 在函数内部访问 volume
    return evaluate_icl(
        model_name=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_samples=max_samples,
        results_volume_obj=results_volume,
    )

