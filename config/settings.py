"""
Configuration settings for the hierarchical medical response generation project.
"""
import os

# API Configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Dataset Configuration
DATASET_NAME = "mahfoos/Patient-Doctor-Conversation"
DATASET_SPLIT = "train"

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Processing Configuration
MAX_WORKERS = 10  # Concurrent API requests (DeepSeek has no rate limit)
TEST_MODE = False  # Set to True to process only a subset
TEST_SIZE = 16  # Number of samples to process in test mode
SAVE_INTERVAL = 10  # Save progress every N samples
MAX_RETRIES = 3  # Retry failed API calls
TIMEOUT = 60  # API call timeout in seconds (DeepSeek has no rate limit but use reasonable timeout)
RESUME_FROM_CHECKPOINT = True  # Resume from checkpoint if available

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

