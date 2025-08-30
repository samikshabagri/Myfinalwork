"""
Configuration file for Maritime AI Agent
"""

import os
from typing import Optional

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"  # Change to "gpt-4" if you have access
MAX_TOKENS = 1000
TEMPERATURE = 0.7
TOP_P = 0.9

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True

# Streamlit Configuration
STREAMLIT_PORT = 8502

# Document Processing
MAX_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOCUMENT = 50

# RAG Configuration
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.3

def get_openai_config() -> dict:
    """Get OpenAI configuration"""
    return {
        "api_key": OPENAI_API_KEY,
        "model": DEFAULT_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }

def is_openai_configured() -> bool:
    """Check if OpenAI is properly configured"""
    return bool(OPENAI_API_KEY)
