"""
LangChain-based RAG pipeline for NutriMind AI.

Same public interface as src/rag_pipeline.RAGPipeline (answer, answer_stream,
clear_memory, chat_history) so it's a drop-in replacement in main_app.py --
nothing in rag_pipeline.py / llm.py / vector_store.py is removed or changed.

What's different / better:
    - answer_stream() yields REAL incremental tokens (via QwenLangChainLLM),
      instead of blocking for the full generation and faking a replay.
    - Retrieval + prompting + memory are expressed as LangChain building
      blocks (BaseRetriever, PromptTemplate, RunnableSequence).
"""

from typing import Any, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from src.langchain_llm import QwenLangChainLLM
from src.llm_factory import get_groq_llm


PROMPT_TEMPLATE = """
You are NutriMind AI, an assistant that answers questions using the user's InBody report.

Rules:
- Use ONLY the information provided in the context. Do not invent values.
- Answer ONLY what is asked. Do not add extra metrics, health interpretations,
  risk assessments, or lifestyle advice unless the question explicitly asks for them.
- Report numbers exactly as they appear in the context (do not label them as
  "healthy", "normal", "moderate", "low risk", etc. unless that exact label
  appears in the context).
- If the question asks for advice, a meal plan, or recommendations, base it
  directly on the CALCULATED NUTRITION TARGETS section if present (calories,
  protein, fat, carbs) rather than generic advice. Explicitly mention the
  numbers you're using.
- This is general informational content, not professional medical or
  dietary advice. If the question asks for a meal plan, calorie target, or
  specific dietary prescription, end your answer with a brief note
  recommending they confirm with a registered dietitian before making
  significant diet changes.
- If the answer is not available in the context, say:
  "I couldn't find this information in your InBody report."

Context:
{context}

{history}

Question:
{question}

Answer:
"""


class VectorStoreRetriever(BaseRetriever):
    """Wraps the existing (FAISS + sentence-transformers) VectorStore so it
    can be used anywhere LangChain expects a retriever."""

    vector_store: Any
    top_k: int = 3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        results = self.vector_store.search(query, top_k=self.top_k)
        docs = []
        for result in results:
            if isinstance(result, dict):
                docs.append(Document(page_content=result["text"], metadata={"score": result.get("score")}))
            else:
                docs.append(Document(page_content=str(result)))
        return docs


class LangChainRAGPipeline:
    """Drop-in replacement for RAGPipeline, backed by LangChain components."""

    def __init__(self, vector_store, tokenizer=None, model=None, inbody_data=None,
                 activity_level="moderate", top_k=3):
        self.vector_store = vector_store
        self.inbody_data = inbody_data
        self.activity_level = activity_level
        self.top_k = top_k
        self.chat_history: List[dict] = []

        self.retriever = VectorStoreRetriever(vector_store=vector_store, top_k=top_k)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        # Try Groq first (fast, cloud). Falls back to local Qwen if no
        # GROQ_API_KEY is configured -- see src/llm_factory.py.
        groq_llm = get_groq_llm(streaming=True)
        if groq_llm is not None:
            self.llm = groq_llm
            self.backend = "groq"
        else:
            if tokenizer is None or model is None:
                raise ValueError(
                    "No GROQ_API_KEY configured and no local model was provided. "
                    "Either set GROQ_API_KEY in .env, or pass a loaded (tokenizer, model) pair."
                )
            self.llm = QwenLangChainLLM(tokenizer=tokenizer, model=model, max_new_tokens=150)
            self.backend = "local"

    @staticmethod
    def _chunk_text(chunk) -> str:
        """Normalizes streaming chunks from either backend: ChatGroq yields
        AIMessageChunk objects (`.content`), the local Qwen LLM yields
        plain strings."""
        return getattr(chunk, "content", chunk) or ""

    def _build_history_text(self) -> str:
        if not self.chat_history:
            return ""
        history = "\n\nPrevious conversation:\n"
        for message in self.chat_history:
            history += f"User: {message['user']}\n"
            history += f"Assistant: {message['assistant']}\n"
        return history

    def _build_prompt(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        history = self._build_history_text()
        return self.prompt.format(context=context, history=history, question=question)

    def answer(self, question: str, top_k: Optional[int] = None) -> str:
        if top_k is not None:
            self.retriever.top_k = top_k

        prompt_text = self._build_prompt(question)
        result = self.llm.invoke(prompt_text)
        answer_text = self._chunk_text(result) if not isinstance(result, str) else result

        self.chat_history.append({"user": question, "assistant": answer_text})
        return answer_text

    def answer_stream(self, question: str, top_k: Optional[int] = None, chunk_size: int = 1) -> Iterator[str]:
        """Real streaming: yields each new text fragment as it's actually
        produced by the model (not a post-hoc replay). The caller is
        expected to accumulate fragments itself (`full_answer += token`),
        matching main_app.py's existing consumer."""
        if top_k is not None:
            self.retriever.top_k = top_k

        prompt_text = self._build_prompt(question)

        full_answer = ""
        for chunk in self.llm.stream(prompt_text):
            delta = self._chunk_text(chunk)
            if not delta:
                continue
            full_answer += delta
            yield delta

        self.chat_history.append({"user": question, "assistant": full_answer})

    def clear_memory(self):
        self.chat_history = []
