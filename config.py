"""
Configuration file for the hierarchical medical response generation project.
"""
import os

# API Configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Dataset Configuration
DATASET_NAME = "mahfoos/Patient-Doctor-Conversation"
DATASET_SPLIT = "train"

# Output Configuration
OUTPUT_DIR = "./data/processed"
OUTPUT_FILENAME = "hierarchical_medical_conversations.json"

# Processing Configuration
MAX_WORKERS = 8  # Number of parallel processes
BATCH_SIZE = 100  # Save progress every N samples
MAX_RETRIES = 3  # Retry failed API calls
TIMEOUT = 30  # API call timeout in seconds

# Prompt Configuration
SYSTEM_PROMPT = """You are a helpful assistant that provides concise and accurate medical information. 
Always organize your response into a hierarchical structure with:
- At least 3 levels in depth
- No more than 5 points at each level
- Follow the Pyramid Principle (top-down reasoning)
- Follow MECE structure (Mutually Exclusive, Collectively Exhaustive)

Use markdown list formatting with proper indentation:
- Main point 1
  - Sub-point 1.1
    - Detail 1.1.1
    - Detail 1.1.2
  - Sub-point 1.2
- Main point 2

Response should only contain the hierarchical content, no meta-commentary."""

