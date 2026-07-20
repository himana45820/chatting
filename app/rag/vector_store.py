import hashlib
import json
import os
import re
from typing import List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.config import get_google_api_key, has_valid_google_api_key


class LocalFallbackEmbeddings:
    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _embed_text(self, text: str) -> List[float]:
        vector = np.zeros(self.dimension, dtype="float32")
        tokens = self._tokenize(text)
        if not tokens:
            return vector.tolist()
        for token in tokens:
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimension
            vector[index] += 1.0
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_text(query)


class FAISSVectorStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embeddings = None
        self.embedding_dimension = 256
        if has_valid_google_api_key():
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-2",
                    google_api_key=get_google_api_key(),
                )
            except Exception:
                self.embeddings = LocalFallbackEmbeddings(self.embedding_dimension)
        else:
            self.embeddings = LocalFallbackEmbeddings(self.embedding_dimension)
        self.index: Optional[faiss.IndexIDMap2] = None
        self.metadata: List[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            if self.index is not None and getattr(self.index, "d", None) != self.embedding_dimension:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.embedding_dimension))
                self.metadata = []
            else:
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.embedding_dimension))
            self.metadata = []

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        if not self.embeddings:
            raise RuntimeError("Google API key is not configured. Please add a real key to the .env file.")

        texts = [doc.page_content for doc in documents]
        try:
            vectors = np.array(self.embeddings.embed_documents(texts), dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        start_id = len(self.metadata)
        ids = np.arange(start_id, start_id + len(documents), dtype="int64")

        if self.index.ntotal == 0:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(vectors.shape[1]))
            self.index.add_with_ids(vectors, ids)
        else:
            incoming_index = faiss.IndexIDMap2(faiss.IndexFlatL2(vectors.shape[1]))
            incoming_index.add_with_ids(vectors, ids)
            self.index.merge_from(incoming_index, 0)

        for doc in documents:
            self.metadata.append(
                {
                    "text": doc.page_content,
                    "source_document": doc.metadata.get("source_document", "unknown"),
                    "page_number": doc.metadata.get("page_number", 0),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                }
            )

        self._save()

    def search(self, query: str, k: int = 4):
        if self.index is None or self.index.ntotal == 0:
            return []
        if not self.embeddings:
            return []

        query_vector = np.array(self.embeddings.embed_query(query), dtype="float32")
        distances, indices = self.index.search(query_vector.reshape(1, -1), min(k, self.index.ntotal))
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if distance > 2.0:
                continue
            item = self.metadata[idx]
            results.append(
                {
                    "text": item["text"],
                    "source_document": item["source_document"],
                    "page_number": item["page_number"],
                    "chunk_id": item["chunk_id"],
                    "similarity_score": float(distance),
                }
            )
        return results

    def get_all_metadata(self) -> List[dict]:
        return self.metadata
