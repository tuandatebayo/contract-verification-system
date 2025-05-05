# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Tải biến môi trường từ file .env nếu có

# --- Cấu hình Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.200.2.34:32411")
# LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b") # Có thể chọn model khác nếu cần
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b") # Sử dụng model đã test gần nhất
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large:335m")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 3000)) # Tăng timeout

# --- Cấu hình Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "labour_collection") # Đảm bảo tên collection chính xác

# --- Cấu hình RAG ---
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 3)) # Tăng K để RAG có nhiều ngữ cảnh hơn

# --- Cấu hình Workflow ---
# WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", 600)) # Giữ timeout hợp lý
WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", 3000)) # Tăng timeout cho các tác vụ LLM/RAG
VERBOSE_WORKFLOW = os.getenv("VERBOSE_WORKFLOW", "True").lower() == "true" # Bật verbose để debug

# --- Cấu hình UI ---
DEFAULT_QUERY = "Kiểm tra hợp đồng lao động theo luật Việt Nam"

# --- Tên Tool RAG ---
RAG_TOOL_NAME = "labor_law_tool"
RAG_TOOL_DESC = "Công cụ truy vấn thông tin chi tiết về các quy định pháp luật lao động tại Việt Nam, bao gồm Bộ luật Lao động, Luật Việc làm, Luật An toàn vệ sinh lao động, và các văn bản liên quan."