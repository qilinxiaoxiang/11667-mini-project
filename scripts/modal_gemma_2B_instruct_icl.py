"""
Modal script for evaluating ICL method with Instruct-style template on test set
- Loads dataset from volume
- Uses local model loading (not Inference API)
- Saves results to volume
- Supports GPU (A10) or CPU
- 默认在测试集上进行全量评估
- 使用指令式模板（Instruct template）
- 默认使用 Phi-2 (2.7B) 模型（开放访问）

Usage:
  # 使用默认模型 (Phi-2 2.7B) - 评估整个测试集
  modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate
  
  # 使用其他模型 - 评估整个测试集
  modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --model "stabilityai/stablelm-2-1_6b-instruct"
  
  # 评估指定数量的测试集样本
  modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --max-samples 50
  modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --max-samples 200
  
  可用的 2B 级别模型（开放访问，无需 gated）:
  - microsoft/phi-2 (推荐，2.7B 参数，开放访问)
  - stabilityai/stablelm-2-1_6b-instruct (1.6B 参数，开放访问)
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
from modal import Volume, Secret


# ICL Template - Instruct Style
ICL_TEMPLATE = """You are a clinical assistant specializing in structured medical reasoning.

Given a patient's question or description, produce a MECE-style hierarchical analysis.

Requirements:
1. Use numbered sections and subsections (1, 1.1, 1.2 …).
2. Each section must follow MECE (mutually exclusive, collectively exhaustive).
3. No diagnosis or prescriptions; list only plausible causes + safe, general advice.
4. Keep bullets concise, factual, and non-overlapping.
5. Output **only** the structured hierarchy. No explanations or extra text.

=== FEW-SHOT EXAMPLES ===

Q: hi doctor my mom aged 45 years is having a fever for almost two months the temperature used to vary from 100 degrees celsius in the night times and gradually it comes down she was weighing 105 keg and now she is keg now she is having bone pain and light fever in the night when there is a pain in the bone she has been advised with calcium and iron tablets she is also getting her regular periods menopause has not reached yet kindly advise
A:
- **1. Clinical Presentation**
  - **1.1 Fever Pattern**
    - Low-grade nocturnal fever
    - Duration around two months
  - **1.2 Systemic Features**
    - Bone pain
    - Notable weight loss

- **2. Plausible Underlying Causes**
  - **2.1 Infectious Etiologies**
    - Chronic infections such as tuberculosis
  - **2.2 Non-Infectious Etiologies**
    - Metabolic, inflammatory, or autoimmune processes

- **3. Recommended Evaluations**
  - **3.1 Baseline Tests**
    - CBC, ESR/CRP
  - **3.2 TB-Specific Tests**
    - Mantoux or IGRA screening

- **4. Supportive Measures**
  - **4.1 Hydration and Diet**
    - Maintain fluid intake
    - Ensure adequate nutrition

- **5. Follow-up Plan**
  - **5.1 Next Steps**
    - Complete evaluations
    - Reassess based on results


Q: hi doctor I am having a problem with sinusitis my doctor ordered me to take co altria 10 mg at bedtime I have taken it in the morning and currently I am experiencing dryness of mouth and headache what to do is this serious I am also taking cefixime 200 mg twice daily sinupret thrice daily and nasoflo spray
A:
- **1. Symptom Explanation**
  - **1.1 Medication Effects**
    - Anticholinergic action may reduce saliva
    - Headache can occur from timing mismatch
  - **1.2 Expected Duration**
    - Often transient
    - Usually improves as body adapts

- **2. Possible Contributing Factors**
  - **2.1 Morning Dose Error**
    - Taking sedating meds early may worsen side effects
  - **2.2 Combination Therapy**
    - Multiple agents can amplify dryness

- **3. Safe Management**
  - **3.1 Immediate Steps**
    - Maintain hydration
    - Monitor symptom progression
  - **3.2 When to Seek Help**
    - Persistent headache
    - Worsening dryness impacting eating or speech


Q: hello doctor I have erosion in my stomach severe burning sensation and sometimes vomiting and bloating I have done endoscopy colonoscopy angiography and hospitalized for about four times recently again I have done endoscopy found astral erosion please advice
A:
- **1. Symptom Profile**
  - **1.1 Upper GI Symptoms**
    - Burning epigastric pain
    - Bloating and nausea
  - **1.2 Course of Illness**
    - Recurrent episodes despite evaluation

- **2. Potential Contributing Factors**
  - **2.1 Gastric Irritation**
    - Acid-related mucosal injury
  - **2.2 Lifestyle Irritants**
    - Spicy food, irregular meals

- **3. Non-Prescriptive Management**
  - **3.1 Behavioral Measures**
    - Smaller frequent meals
    - Avoid irritants such as caffeine/spices
  - **3.2 Monitoring**
    - Track symptom triggers
    - Watch for red-flag symptoms (vomiting blood, weight loss)

- **4. Information Needed**
  - **4.1 Prior Reports**
    - Endoscopy findings trend
  - **4.2 Infection Status**
    - Whether H. pylori testing was done


=== NOW ANSWER IN THE SAME MECE STYLE ===

Q: {{question}}

A:
"""


def create_prompt(patient_description: str) -> str:
    """创建 ICL prompt，使用患者描述（Patient字段）作为输入"""
    # 注意：ICL模板中使用 {{question}} 作为占位符，但实际传入的是 Patient 字段（患者详细描述）
    return ICL_TEMPLATE.replace("{{question}}", patient_description)


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


def load_local_model(model_name: str, device: str = "cuda", hf_token: Optional[str] = None):
    """
    加载本地模型和分词器
    
    Args:
        model_name: 模型名称
        device: 设备（cuda/cpu）
        hf_token: Hugging Face token（用于访问 gated 模型）
        
    Returns:
        (model, tokenizer, device) 元组
    """
    print(f"  正在加载模型 {model_name} 到设备 {device}...")
    
    # 获取 HF token（用于 gated 模型如 Llama/Gemma）
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # 准备 tokenizer 参数
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
        print(f"  ✓ 使用 Hugging Face token 访问模型")
    
    # Gemma 模型可能需要 token，尝试加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            **tokenizer_kwargs
        )
    except Exception as e1:
        # 如果失败，尝试不使用 trust_remote_code
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_kwargs
            )
        except Exception as e2:
            print(f"  ✗ Tokenizer 加载失败")
            print(f"     错误1: {str(e1)[:200]}")
            print(f"     错误2: {str(e2)[:200]}")
            raise
    
    # 设置 pad token（Gemma 模型通常需要这个设置）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 确定 dtype
    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # 准备 model 参数
    model_kwargs = {
        "dtype": dtype,
        "device_map": "auto" if device == "cuda" else None,
    }
    if hf_token:
        model_kwargs["token"] = hf_token
    
    # Gemma 模型加载
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs
        )
    except Exception:
        # 如果失败，尝试不使用 trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
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
    model_name: str = "microsoft/phi-2",  # 2.7B 参数，Phi-2 模型（开放访问，无需 gated）
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_samples: Optional[int] = None,  # 默认评估整个测试集，可指定其他数量
    hf_token: Optional[str] = None,  # Hugging Face token（用于访问 gated 模型如 Gemma）
    results_volume_obj=None,
):
    """
    在测试集上评估 ICL 方法（默认评估全部样本）- 使用指令式模板
    
    Args:
        model_name: 模型名称（默认 Phi-2 2.7B）
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: Nucleus sampling参数
        max_samples: 评估样本数（默认None表示全部测试集样本，可指定其他数量）
        results_volume_obj: Modal volume对象，用于保存结果
        
    Returns:
        评估结果摘要
    """
    # 在 Modal GPU 上总是使用 CUDA
    # 在 Modal 上，如果配置了 GPU，则使用 CUDA
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print("=" * 60)
    print("ICL 方法评估 - 测试集 (Modal) - Phi-2")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"设备: {device}")
    print(f"评估样本数: {max_samples or '全部'}")
    
    # 加载数据集
    print("\n加载数据集...")
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
    print(f"\n将从测试集中评估 {num_samples} 个样本（测试集共 {len(test_dataset)} 个）")
    
    # 加载模型
    print(f"\n{'='*60}")
    print("加载模型")
    print(f"{'='*60}")
    model, tokenizer, device_str = load_local_model(model_name, device_str, hf_token=hf_token)
    device = torch.device(device_str)
    
    print_memory_info(device)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/results/icl_eval_test_phi2_instruct_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    failed_count = 0
    
    # 每20个样本保存一次中间结果
    checkpoint_interval = 20
    
    print(f"\n{'='*60}")
    print("开始评估")
    print(f"{'='*60}\n")
    
    for i in range(num_samples):
        sample = test_dataset[i]
        
        # 构建问题（使用 Patient 作为输入，包含患者的详细描述）
        patient_input = sample.get('Patient', '')
        if not patient_input:
            # 如果 Patient 为空，回退到 Description
            patient_input = sample.get('Description', '')
        
        print(f"\n[{i+1}/{num_samples}] 处理问题...")
        print(f"患者描述: {patient_input[:100]}..." if len(patient_input) > 100 else f"患者描述: {patient_input}")
        
        # 使用 ICL 生成答案（传入患者描述）
        prompt = create_prompt(patient_input)
        
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
            "description": sample.get('Description', ''),  # 简短的问题描述
            "patient_description": patient_input,  # 患者详细描述（用于ICL输入）
            "reference_answer": sample.get('Doctor', ''),
            "generated_answer": generated_answer if generated_answer else "FAILED",
            "status": sample.get('Status', ''),
            "generation_time": elapsed_time,
            "success": generated_answer is not None,
        }
        results.append(result)
        
        # 每20个样本保存一次中间结果
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_file = os.path.join(output_dir, f"icl_results_checkpoint_{i+1}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  💾 已保存检查点 ({i+1}/{num_samples}): {checkpoint_file}")
            
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
        "template_type": "instruct",  # 标识这是指令式模板
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
app = modal.App("medical-icl-evaluation-gemma2b-instruct")

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
    # secrets=[Secret.from_name("huggingface-secret")],  # Phi-2 不需要 gated access，可以注释掉
    timeout=86400,  # 24小时超时
)
def evaluate(
    model: str = "microsoft/phi-2",  # 2.7B 参数，Phi-2 模型（开放访问，无需 gated）
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_samples: Optional[int] = None,  # 默认评估整个测试集，可指定其他数量
):
    """
    Entrypoint for modal run command - 默认评估整个测试集（使用指令式模板 + Phi-2）
    
    Example:
        # 评估整个测试集 - 默认行为（使用 Phi-2 2.7B）
        modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate
        
        # 使用其他模型：
        modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --model "stabilityai/stablelm-2-1_6b-instruct"
        
        可用的 2B 级别模型选项（开放访问）：
        - microsoft/phi-2 (2.7B, 推荐，无需 gated access)
        - stabilityai/stablelm-2-1_6b-instruct (1.6B, 无需 gated access)
        
        # 评估指定数量的测试集样本
        modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --max-samples 50
        modal run scripts/modal_gemma_2B_instruct_icl.py::evaluate --max-samples 200
    """
    # Phi-2 不需要 gated access，但如果使用其他 gated 模型，需要从环境变量获取 HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # 在函数内部访问 volume
    return evaluate_icl(
        model_name=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        max_samples=max_samples,
        hf_token=hf_token,
        results_volume_obj=results_volume,
    )


