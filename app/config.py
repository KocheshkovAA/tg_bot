import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH"))
