from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env into os.environ

class Config:
    # MongoDB
    MONGO_URI = os.getenv("MONGO_URI")

    # API Keys
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

    # Token limits per tier
    TOKEN_LIMITS = {
        "free": int(os.getenv("FREE_TOKEN_LIMIT", 1000)),
        "plus": int(os.getenv("PLUS_TOKEN_LIMIT", 3000)),
        "pro": int(os.getenv("PRO_TOKEN_LIMIT", 8000)),
    }

    # Exchange count caps
    EXCHANGE_LIMITS = {
        "free": int(os.getenv("FREE_MAX_EXCHANGES", 30)),
        "plus": int(os.getenv("PLUS_MAX_EXCHANGES", 70)),
        "pro": int(os.getenv("PRO_MAX_EXCHANGES", 100)),
    }

    # File size caps (MB)
    FILE_SIZE_MB_LIMITS = {
        "free": int(os.getenv("FREE_FILE_SIZE_MB", 5)),
        "plus": int(os.getenv("PLUS_FILE_SIZE_MB", 10)),
        "pro": int(os.getenv("PRO_FILE_SIZE_MB", 20)),
    }
    
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Typesense Config
    TYPESENSE_HOST = os.getenv("TYPESENSE_HOST")
    TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY")
