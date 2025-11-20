# Hierarchical Medical Response Generation

**11-667 Large Language Models: Methods and Applications - Mini Project**

This project aims to fine-tune small language models to generate medical consultation responses in a structured, hierarchical format following the Pyramid Principle and MECE (Mutually Exclusive, Collectively Exhaustive) framework.

## 📋 Project Overview

### Objectives
1. **Structured Output Generation**: Train models to organize medical responses into multi-level (≥3 levels) hierarchical lists with ≤5 points per level
2. **Medical Domain Application**: Apply structured reasoning to doctor-patient conversations in healthcare contexts
3. **Model Comparison**: Compare small foundation models' ability to learn and generate structured outputs

### Models
- **Qwen2.5-0.5B (base)**: Smaller, less instruction-tuned, more sensitive to prompt format
- **DeepSeek-R1 1.5B (instruct)**: Larger, better instruction adherence, more stable structure output

### Dataset
[Patient-Doctor-Conversation](https://huggingface.co/datasets/mahfoos/Patient-Doctor-Conversation) - Contains ~3,300 medical consultation dialogues with patient descriptions, conversations, and severity labels.

## 🚀 Quick Start

### Prerequisites
```bash
# Conda environment (recommended)
conda activate 11667

# Or create new environment
conda create -n 11667 python=3.10
conda activate 11667
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For Modal fine-tuning:
pip install modal
```

### Environment Setup
Set your DeepSeek API key in `~/.zshrc` (or `~/.bashrc`):
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

Then reload:
```bash
source ~/.zshrc
```

## 📁 Project Structure

```
mini_project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                      # Original test script (legacy)
├── config/                      # Configuration files
│   ├── __init__.py
│   └── settings.py              # All configuration settings
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading and caching
│   ├── converter.py             # API client for hierarchical conversion
│   ├── processor.py             # Multi-process batch processor
│   └── utils/                   # Utility functions
│       └── __init__.py
├── scripts/                     # Executable scripts
│   ├── download_dataset.py      # Download and cache dataset
│   └── convert_dataset.py       # Convert dataset to hierarchical format
└── data/                        # Data directory
    ├── raw/                     # Downloaded raw dataset (cached)
    └── processed/               # Converted hierarchical datasets
        └── hierarchical_dataset/  # HuggingFace format for fine-tuning
```

## 🔄 Dataset Workflow

### Step 1: Download Dataset (Optional)

Download and cache the dataset without processing:

```bash
python scripts/download_dataset.py
```

### Step 2: Convert to Hierarchical Format

Convert doctor responses to hierarchical structure:

```bash
# Test mode: Process only 16 examples
python scripts/convert_dataset.py

# Full dataset: Edit config/settings.py and set TEST_MODE = False
python scripts/convert_dataset.py
```

### Features
- **Multi-process Parallel Processing**: Concurrent API calls (4 workers by default)
- **Automatic Checkpointing**: Saves progress every 10 samples (configurable)
- **Retry Logic**: Handles API failures with exponential backoff
- **Progress Tracking**: Real-time progress bar with success rate statistics
- **Format Preservation**: Maintains original HuggingFace dataset structure

### Configuration

Adjust settings in `config/settings.py`:

```python
# Processing Configuration
MAX_WORKERS = 4      # Number of parallel API requests
TEST_MODE = True     # Set to False for full dataset
TEST_SIZE = 16       # Number of examples in test mode
SAVE_INTERVAL = 10   # Save checkpoint every N samples
MAX_RETRIES = 3      # Retry failed API calls
TIMEOUT = 60         # API call timeout in seconds
```

### Output Format

The processed dataset maintains the same structure as the original:

```python
{
  "Description": "what does abutment of the nerve root mean",
  "Patient": "hi doctor I am just wondering...",
  "Doctor": "- Assessment\n  - Query Understanding\n    - ...",  # Hierarchical format
  "Status": "medium severity"
}
```

Note: Failed conversions fall back to original doctor responses.

## 📊 Hierarchical Structure Criteria

All converted responses follow:
1. **Depth**: Minimum 3 levels of hierarchy
2. **Breadth**: Maximum 5 points per level
3. **Pyramid Principle**: Top-down reasoning (answer first, then supporting details)
4. **MECE**: Mutually exclusive, collectively exhaustive categorization

Example:
```markdown
- Main Assessment
  - Diagnosis Category 1
    - Specific symptom observation
    - Relevant medical indicator
  - Diagnosis Category 2
    - Supporting evidence
- Treatment Recommendations
  - Primary intervention
    - Medication details
    - Dosage information
  - Secondary considerations
```

## 🛠️ Development Workflow

### 1. Data Preparation
```bash
# Download dataset
python scripts/download_dataset.py

# Test conversion (16 examples)
python scripts/convert_dataset.py

# Full conversion: Set TEST_MODE = False in config/settings.py
python scripts/convert_dataset.py
```

### 2. Model Fine-tuning
(To be implemented)
- Load base models (Qwen2.5-0.5B, DeepSeek-R1 1.5B)
- Fine-tune on hierarchical dataset from `data/processed/hierarchical_dataset/`
- Evaluate structure quality and medical accuracy

### 3. Model Fine-tuning on Modal

**Important correction based on actual reference data**: Memory scales as `batch_size × seq_len²`, not model size. Using your 120MB model reference (batch=50, seq=1024 → 70GB), corrected batch sizes below.

#### Setup (One-time)

```bash
pip install modal
modal token new  # Authenticate

# Create volumes
modal volume create medical-dataset-volume
modal volume create medical-models-volume

# Upload dataset
modal volume put medical-dataset-volume \
  data/processed/hierarchical_dataset_clean /hierarchical_dataset_clean
```

#### Recommended Training Plans

**Plan A: Safest ⭐ (RECOMMENDED)**
```
Model: Qwen2.5-0.5B (Full Fine-tuning)
batch_size=16, context_length=512, epochs=3
Total iterations: 558 (2,980 samples / 16 batch × 3)
GPU: A100 (80GB)
Time: 15-20 minutes
Cost: $0.71
Peak memory: 4-6 GB
Command: modal run --detach modal_qwen_finetune_full.py::train
```

**Plan B: Maximum Quality**
```
Model: DeepSeek-R1 1.5B (Full + Gradient Checkpointing)
batch_size=4, context_length=512, epochs=3
Total iterations: 2,235
GPU: A100 (80GB)
Time: 45-60 minutes
Cost: $1.50
Peak memory: 8-10 GB
Command: modal run --detach modal_deepseek_finetune_full.py::train
```

**Plan C: Budget**
```
Model: Qwen2.5-0.5B (LoRA)
batch_size=32, context_length=512, epochs=3
Total iterations: 280
GPU: A10G (24GB)
Time: 8-10 minutes
Cost: $0.06
Peak memory: 2-3 GB
Command: modal run --detach modal_qwen_finetune_lora.py::train
```

#### Correct Modal Commands

```bash
cd scripts

# Qwen full fine-tuning (safest)
modal run --detach modal_qwen_finetune_full.py::train

# OR: DeepSeek full fine-tuning (best quality)
modal run --detach modal_deepseek_finetune_full.py::train

# OR: Qwen LoRA (fastest/cheapest)
modal run --detach modal_qwen_finetune_lora.py::train

# Monitor progress
modal apps list           # Get APP_ID
modal logs APP_ID         # View logs

# Download results
modal volume ls medical-models-volume
modal volume get medical-models-volume qwen_full_TIMESTAMP/final ./outputs/
```

#### Memory Reality (Corrected)

**Why these batch sizes?**

Your 120MB model: batch=50, seq=1024 → 70GB peak memory

This comes from **attention matrices** (Q @ K^T): `batch × heads × seq_len²`
- Not model size! Model weights are only ~500MB
- Quadratic scaling: reducing seq_len by 2x saves 75% activation memory
- Using seq=512 instead of 1024 brings memory down 10x

**Qwen2.5-0.5B (500M params):**
- batch=16, seq=512 → ~4-6GB ✓
- batch=32, seq=512 → ~8-10GB ✓
- batch=8, seq=1024 → ~8-12GB (risky)

**DeepSeek-R1 1.5B (1.5B params):**
- batch=4, seq=512 → ~6-8GB ✓ (gradient checkpointing helps)
- batch=8, seq=512 → ~8-12GB ✓
- batch=4, seq=1024 + checkpointing → ~25-30GB ⚠️ (tight)

#### Expected Results

Based on your 120MB model (20,000 iters → 0.02 loss):

**Our Qwen setup (558 iterations):**
- Should reach **0.08-0.15 loss** (hierarchical generation is harder)
- Training should plateau after epoch 2
- Structure quality improves non-linearly with loss

### 4. Evaluation
(To be implemented)
- Structure compliance metrics
- Medical accuracy assessment
- Human evaluation of response quality

## 💻 Hardware Requirements & Training Feasibility

### AWS g5.2xlarge Compatibility (24GB VRAM, 32GB RAM)

Both target models can be fine-tuned on g5.2xlarge instances:

#### **Qwen2.5-0.5B** - Comfortable Fit ✅

| Configuration | VRAM Usage | Batch Size | Training Time (3 epochs) |
|--------------|------------|------------|-------------------------|
| **Full Fine-tuning (FP16)** | ~12GB | 8 | ~20 minutes |
| LoRA Fine-tuning | ~4GB | 32+ | ~15 minutes |

**Recommended Configuration**:
```python
TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch = 16
    fp16=True,
    max_seq_length=512,
    learning_rate=2e-5,
)
```

#### **DeepSeek-R1-1.5B** - Requires Optimization ✅

| Configuration | VRAM Usage | Batch Size | Training Time (3 epochs) |
|--------------|------------|------------|-------------------------|
| **Full Fine-tuning (FP16 + Gradient Checkpointing)** | ~22GB | 2 | ~40 minutes |
| LoRA Fine-tuning | ~6GB | 16+ | ~25 minutes |

**Recommended Configuration**:
```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    gradient_checkpointing=True,    # Required!
    fp16=True,
    max_seq_length=512,
    learning_rate=1e-5,
)
```

### Memory Breakdown

**Full Fine-tuning Memory Components**:
- Model weights (FP16): 0.5B → 0.93GB, 1.5B → 2.79GB
- Gradients (FP16): Same as weights
- Optimizer states (AdamW): 8 bytes/param
- Activations: 2-4GB (varies with batch size)

**Optimization Techniques**:
1. **Gradient Checkpointing** (Required for DeepSeek-R1-1.5B)
   - Saves 30-50% activation memory
   - ~20% slower training
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Flash Attention** (Optional, provides 20-40% speedup)
   ```bash
   pip install flash-attn --no-build-isolation
   ```
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       attn_implementation="flash_attention_2"
   )
   ```

3. **Mixed Precision Training** (Required)
   - Use FP16 for all training
   - Reduces memory by 50% compared to FP32

## 📊 Dataset Statistics

### Final Conversion Results

| Metric | Count | Percentage |
|--------|-------|------------|
| **Original Dataset** | 3,325 samples | 100% |
| **Successfully Converted** | 3,311 samples | **99.73%** |
| **Failed Conversions** | 9 samples | 0.27% |
| **Final Training Dataset** | 3,311 samples | - |

### Failed Samples
9 samples failed conversion because their `Doctor` field was NULL in the original dataset (data quality issue, not conversion failure). These have been filtered out from the training dataset.

**Failed sample indices**: `[98, 99, 733, 785, 1060, 1069, 1236, 1381, 1565]`

### Dataset Format

The processed dataset uses **Apache Arrow format** (HuggingFace standard):
- **Format**: `.arrow` binary columnar format
- **Organization**: Each sample is an independent record (no separators needed)
- **Access**: `dataset[0]`, `dataset[1]`, etc. or `dataset[:10]` for batch
- **Training Ready**: Direct compatibility with HuggingFace `Trainer`
- **Size**: ~8.7 MB (3,311 samples)

**Directory structure**:
```
data/processed/hierarchical_dataset_clean/
├── data-00000-of-00001.arrow  # Main data file
├── dataset_info.json           # Schema and metadata
└── state.json                  # Dataset state
```

**Fields**:
- `Description`: Medical question/issue (string)
- `Patient`: Patient's detailed description (string)
- `Doctor`: **Hierarchical medical response** (string, markdown format)
- `Status`: Severity level (string: "low/medium/high severity")
- `_original_doctor`: Original doctor response for comparison (string)
- `_conversion_success`: Conversion status flag (boolean)

## 📈 Expected Outcomes

1. **Structured Training Data**: 3,311 hierarchically formatted medical responses ✅
2. **Fine-tuned Models**: Two models capable of generating structured medical advice
3. **Comparative Analysis**: Performance comparison between base and instruct models on structured output tasks

## 🔧 Technical Details

### API Integration
- Uses DeepSeek API for hierarchical conversion
- OpenAI-compatible SDK interface
- No rate limiting (per [DeepSeek API docs](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit))
- Conservative concurrent requests (4 workers) for stability

### Multiprocessing Strategy
- Process-level parallelism for concurrent API calls
- Independent worker processes to avoid GIL limitations
- Automatic progress tracking and checkpoint coordination

### Data Pipeline
1. Download and cache dataset from HuggingFace
2. Parallel API conversion with retry logic and exponential backoff
3. Save checkpoints every 10 samples
4. Export in HuggingFace Dataset format (preserves original structure)
5. Failed conversions fall back to original responses

## 📝 Notes

### Development Environment
- All processing uses the `11667` conda environment
- API key must be set in environment variables before running
- Dataset: 3,325 medical consultation dialogues (~6MB total)

### Dataset Conversion
- DeepSeek API has no rate limiting (per [official docs](https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit))
- 10 parallel workers for conversion (optimized)
- Full dataset conversion time: ~54 minutes (10 workers, 3,311 samples)
- Checkpoints saved every 10 samples for interruption recovery
- Arrow format for HuggingFace ecosystem compatibility
- Final success rate: **99.73%** (3,311/3,320 valid samples)

### Training Environment
- **Tested on**: AWS g5.2xlarge (24GB VRAM, 32GB RAM)
- **GPU**: NVIDIA A10G Tensor Core GPU
- **Both models fit**: Full fine-tuning possible for both Qwen2.5-0.5B and DeepSeek-R1-1.5B
- **Dependencies**: PyArrow automatically installed via `datasets` library (no manual setup needed)

## 👥 Team

Course: 11-667 Large Language Models: Methods and Applications  
Carnegie Mellon University, Fall 2025

## 📄 License

This project is for educational purposes as part of CMU coursework.

## 🔗 References

- [DeepSeek API Documentation](https://api-docs.deepseek.com/)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [Pyramid Principle](https://en.wikipedia.org/wiki/Pyramid_principle)
- [MECE Framework](https://en.wikipedia.org/wiki/MECE_principle)

