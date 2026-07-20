from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import ChatRequest, ChatResponse, RetrievedChunk, UploadResponse
from app.utils.config import get_document_processor, get_generator, get_vector_store, get_google_api_key, has_valid_google_api_key

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.filename is None or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    if not has_valid_google_api_key():
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is missing or still set to the placeholder value. Add your real key to the .env file before uploading a PDF.")

    processor = get_document_processor()
    vector_store = get_vector_store()

    contents = await file.read()
    try:
        chunks = processor.process_pdf_bytes(contents, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

    try:
        vector_store.add_documents(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index the PDF: {exc}") from exc

    return UploadResponse(
        message="PDF uploaded and indexed successfully.",
        document_name=file.filename,
        chunks_added=len(chunks),
        total_chunks=len(vector_store.get_all_metadata()),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    vector_store = get_vector_store()
    generator = get_generator()

    chunks = vector_store.search(request.message, k=4)
    if not chunks:
        return ChatResponse(answer="I could not find relevant information in the uploaded PDF(s).", retrieved_chunks=[], fallback_used=True)

    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            source_document=chunk["source_document"],
            page_number=chunk["page_number"],
            similarity_score=chunk["similarity_score"],
            text=chunk["text"],
        )
        for chunk in chunks
    ]

    answer = generator.generate_answer(request.message, retrieved_chunks)
    return ChatResponse(answer=answer, retrieved_chunks=retrieved_chunks, fallback_used=False)
