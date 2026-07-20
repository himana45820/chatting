from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[dict]] = []


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_document: str
    page_number: int
    similarity_score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk] = []
    fallback_used: bool = False


class UploadResponse(BaseModel):
    message: str
    document_name: str
    chunks_added: int
    total_chunks: int
