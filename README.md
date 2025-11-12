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

### 3. Evaluation
(To be implemented)
- Structure compliance metrics
- Medical accuracy assessment
- Human evaluation of response quality

## 📈 Expected Outcomes

1. **Structured Training Data**: ~3,000+ hierarchically formatted medical responses
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

- All processing uses the `11667` conda environment
- API key must be set in environment variables before running
- Large dataset conversion may take several hours depending on API rate limits
- Checkpoints allow resuming from interruptions

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

