import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DB_DIR = BASE_DIR / "db"

# Notion
NOTION_API_KEY = os.getenv("NOTION_API_KEY")

# Root pages to index (add Notion page URLs or IDs here)
# Example: "https://notion.so/My-Page-abc123" or just "abc123"
ROOT_PAGES = ["https://www.notion.so/baymac/Workout-2dfc9c6bdc62803f9ad6fc5f3ec30a34"]

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4
