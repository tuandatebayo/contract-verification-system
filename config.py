# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file if it exists

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.200.2.34:32411")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large:335m")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 3000))

# --- Qdrant Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "labour_collection") # Make sure this is the correct name

# --- RAG Configuration ---
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 1))

# --- Workflow Configuration ---
WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", 3000))
VERBOSE_WORKFLOW = os.getenv("VERBOSE_WORKFLOW", "False").lower() == "true" # Set to True for debugging workflow steps

# --- UI Configuration ---
DEFAULT_QUERY = "Kiểm tra hợp đồng lao động theo luật Việt Nam"