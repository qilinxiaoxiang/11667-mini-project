"""
Converter for transforming doctor responses to hierarchical format.
"""
import time
from typing import Optional
from openai import OpenAI
from config import settings


class HierarchicalConverter:
    """Converts medical responses to hierarchical format using DeepSeek API."""
    
    def __init__(self, api_key=None, base_url=None, model=None):
        """
        Initialize the converter with API client.
        
        Args:
            api_key: DeepSeek API key (default from config)
            base_url: API base URL (default from config)
            model: Model name (default from config)
        """
        self.api_key = api_key or settings.DEEPSEEK_API_KEY
        self.base_url = base_url or settings.DEEPSEEK_BASE_URL
        self.model = model or settings.DEEPSEEK_MODEL
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables!")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def convert(self, text: str, max_retries: int = None) -> Optional[str]:
        """
        Convert a single doctor response to hierarchical format.
        
        Args:
            text: Original doctor response text
            max_retries: Maximum number of retry attempts (default from config)
            
        Returns:
            Hierarchical formatted text or None if conversion fails
        """
        max_retries = max_retries or settings.MAX_RETRIES
        
        # Skip null/empty responses
        if not text or text == "null" or str(text).strip() == "":
            return None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": settings.SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    timeout=settings.TIMEOUT,
                    stream=False
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"  ⚠ Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)[:80]}")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts: {str(e)[:100]}")
                    return None
        
        return None

