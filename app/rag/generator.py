from typing import List

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models.schemas import RetrievedChunk
from app.utils.config import get_google_api_key, has_valid_google_api_key


class RAGGenerator:
    def __init__(self):
        self.llm = None
        if has_valid_google_api_key():
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                google_api_key=get_google_api_key(),
                temperature=0.2,
            )

    def generate_answer(self, question: str, chunks: List[RetrievedChunk]) -> str:
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"Source: {chunk.source_document} | Page: {chunk.page_number} | Chunk ID: {chunk.chunk_id}\n{chunk.text}"
            )

        context = "\n\n---\n\n".join(context_parts)
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=(
                "You are a helpful assistant. Answer the user's question using only the provided context.\n"
                "If the context does not contain enough information, say that clearly.\n\n"
                "Question: {question}\n\n"
                "Relevant Context:\n{context}"
            ),
        )
        filled_prompt = prompt.format(question=question, context=context)

        if not self.llm:
            if not chunks:
                return "I could not find relevant information in the uploaded PDF(s)."
            preview = " ".join(chunk.text[:220] for chunk in chunks[:2])
            return f"Based on the uploaded PDF content, I found relevant information: {preview}"

        response = self.llm.invoke(filled_prompt)
        if hasattr(response, "content") and isinstance(response.content, list):
            text_blocks = []
            for item in response.content:
                if isinstance(item, dict) and "text" in item:
                    text_blocks.append(str(item["text"]))
            return "".join(text_blocks).strip() if text_blocks else str(response.content)

        if hasattr(response, "content"):
            return str(response.content).strip()

        return str(response)
