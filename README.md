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
├── config.py                    # Configuration file
├── convert_dataset.py           # Multi-process dataset conversion
├── main.py                      # Original test script
└── data/
    └── processed/               # Converted hierarchical datasets
        ├── hierarchical_medical_conversations.json
        └── hf_dataset/          # HuggingFace format for fine-tuning
```

## 🔄 Dataset Conversion

Convert the original doctor responses to hierarchical format using DeepSeek API:

```bash
python convert_dataset.py
```

### Features
- **Multi-process Parallel Processing**: Utilizes multiple CPU cores for faster conversion
- **Automatic Checkpointing**: Saves progress every 100 samples (configurable)
- **Retry Logic**: Handles API failures with exponential backoff
- **Progress Tracking**: Real-time progress bar with success rate statistics
- **Dual Format Output**: Saves both JSON and HuggingFace dataset formats

### Configuration
Adjust settings in `config.py`:
- `MAX_WORKERS`: Number of parallel processes (default: 8)
- `BATCH_SIZE`: Checkpoint frequency (default: 100)
- `MAX_RETRIES`: API retry attempts (default: 3)
- `TIMEOUT`: API call timeout in seconds (default: 30)

### Output Format
```json
{
  "index": 0,
  "Description": "what does abutment of the nerve root mean",
  "Patient": "hi doctor I am just wondering...",
  "Doctor_Original": "hi I have gone through your query...",
  "Doctor_Hierarchical": "- Assessment\n  - Query Understanding\n    - ...",
  "Status": "medium severity",
  "conversion_success": true
}
```

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
# Convert dataset to hierarchical format
python convert_dataset.py
```

### 2. Model Fine-tuning
(To be implemented)
- Load base models (Qwen2.5-0.5B, DeepSeek-R1 1.5B)
- Fine-tune on hierarchical dataset
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
- Uses DeepSeek API for initial hierarchical conversion
- OpenAI-compatible SDK interface
- Robust error handling and rate limiting

### Multiprocessing Strategy
- Process-level parallelism for API calls
- Independent worker processes to avoid GIL limitations
- Shared progress tracking and checkpoint coordination

### Data Pipeline
1. Load raw dataset from HuggingFace
2. Parallel API conversion with retry logic
3. Save checkpoints incrementally
4. Export in multiple formats (JSON, HuggingFace Dataset)

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

