import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class PDFDocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def process_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        temp_dir = Path(tempfile.gettempdir())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4().hex}_{filename}"

        with open(temp_path, "wb") as handle:
            handle.write(pdf_bytes)

        try:
            reader = PdfReader(str(temp_path))
            documents: List[Document] = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                chunks = self.splitter.split_text(text)
                for index, chunk_text in enumerate(chunks):
                    chunk_id = hashlib.sha256(f"{filename}:{page_number}:{index}:{chunk_text}".encode("utf-8")).hexdigest()[:16]
                    documents.append(
                        Document(
                            page_content=chunk_text,
                            metadata={
                                "source_document": filename,
                                "page_number": page_number,
                                "chunk_id": chunk_id,
                            },
                        )
                    )

            return documents
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
