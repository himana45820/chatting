import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR.parent / ".env")


@lru_cache(maxsize=1)
def get_vector_store():
    from app.rag.vector_store import FAISSVectorStore

    return FAISSVectorStore(
        index_path=str(DATA_DIR / "index.faiss"),
        metadata_path=str(DATA_DIR / "metadata.json"),
    )


@lru_cache(maxsize=1)
def get_document_processor():
    from app.rag.document_processor import PDFDocumentProcessor

    return PDFDocumentProcessor()


@lru_cache(maxsize=1)
def get_generator():
    from app.rag.generator import RAGGenerator

    return RAGGenerator()


def get_google_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY", "").strip()


def has_valid_google_api_key() -> bool:
    key = get_google_api_key()
    return bool(key) and key != "put_your_google_api_key_here"
