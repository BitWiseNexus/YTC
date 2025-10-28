from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.2
    CHUNK_SIZE = 1700
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 4
    VECTOR_STORE_DIR = "vector_stores"
    DATABASE_PATH = "ytc_database.db"
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
