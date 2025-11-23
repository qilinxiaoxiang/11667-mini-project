# Modal ICL 评估脚本运行指南

## 运行步骤

### 1. 确认数据集已上传到 Modal Volume

评估脚本需要从 Modal Volume 加载数据集。如果你的数据集还没有上传，需要先上传：

```bash
# 检查 volume 中是否有数据集
modal volume ls medical-dataset-volume

# 如果没有，需要先上传数据集到 volume
# 可以使用训练脚本中的方式，或者手动上传
```

### 2. 运行评估（默认10个样本）

```bash
cd /Users/xinshiwang/Documents/CMU\ Courses/11667/mini-project/11667-mini-project

# 基本运行（默认10个样本，模型 Qwen/Qwen3-1.7B-Base）
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate

# 或者指定参数
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate \
  --model "Qwen/Qwen3-1.7B-Base" \
  --max-samples 10
```

### 3. 查看运行状态

```bash
# 查看所有运行中的应用
modal apps list

# 查看日志（替换 APP_ID 为实际的 app ID）
modal logs APP_ID

# 或者实时跟踪日志
modal logs APP_ID --follow
```

### 4. 运行全部测试集（332个样本）

```bash
# 跑全部332个测试样本
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate \
  --model "Qwen/Qwen3-1.7B-Base" \
  --max-samples 332
```

### 5. 下载结果

评估完成后，结果会保存到 `medical-results-volume` volume 中：

```bash
# 查看结果文件
modal volume ls medical-results-volume

# 下载结果（替换 TIMESTAMP 为实际的时间戳）
modal volume get medical-results-volume icl_eval_TIMESTAMP ./evaluation/icl_results/
```

## 参数说明

- `--model`: 模型名称（默认: "Qwen/Qwen3-1.7B-Base"）
- `--max-samples`: 评估样本数（默认: 10，None 或 332 表示全部）
- `--max-new-tokens`: 最大生成token数（默认: 1024）
- `--temperature`: 采样温度（默认: 0.7）
- `--top-p`: Nucleus sampling参数（默认: 0.9）

## 示例命令

```bash
# 测试运行（10个样本）
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate

# 跑50个样本
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate \
  --max-samples 50

# 跑全部332个测试样本
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate \
  --max-samples 332

# 使用其他模型
modal run --detach scripts/modal_qwen_1.78B_base_evaluate_icl.py::evaluate \
  --model "Qwen/Qwen2.5-0.5B" \
  --max-samples 10
```

## 注意事项

1. **数据集路径**: 脚本会从 `/dataset/hierarchical_dataset_clean` 加载数据集
2. **结果路径**: 结果会保存到 `/results/icl_eval_TIMESTAMP/`
3. **GPU使用**: 默认使用 A10 GPU，可以在脚本中修改为其他 GPU 类型或 CPU
4. **超时设置**: 默认24小时超时，对于332个样本应该足够

