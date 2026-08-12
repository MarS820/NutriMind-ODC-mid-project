"""
Quick test script: ask a question against ANY InBody report PDF using the
RAG pipeline (vector store + Qwen LLM). Every run re-parses the given PDF
and rebuilds the vector store from it, instead of reusing old cached data.

Run locally with:
    python test_query.py "data/inbody.pdf" "What is my BMI?"
    python test_query.py "data/my_new_scan.pdf" "Can you build me a meal plan?"

If you omit the PDF path, it defaults to data/inbody.pdf.
"""

import sys

from src.pdf_processor import load_pdf_text, parse_inbody, create_chunks
from src.vector_store import VectorStore
from src.llm import load_llm
from src.rag_pipeline import RAGPipeline

DEFAULT_PDF_PATH = "data/inbody.pdf"


def main():
    # Usage: test_query.py [pdf_path] [question]
    # Both are optional; sensible defaults are used if omitted.
    args = sys.argv[1:]

    if len(args) >= 2:
        pdf_path, question = args[0], args[1]
    elif len(args) == 1:
        # If the single argument looks like a PDF path, treat it as such;
        # otherwise treat it as the question against the default PDF.
        if args[0].lower().endswith(".pdf"):
            pdf_path, question = args[0], "What is my BMI?"
        else:
            pdf_path, question = DEFAULT_PDF_PATH, args[0]
    else:
        pdf_path, question = DEFAULT_PDF_PATH, "What is my BMI?"

    print(f"PDF: {pdf_path}")
    print(f"Question: {question}\n")

    # Always parse THIS run's PDF fresh.
    raw_text = load_pdf_text(pdf_path)
    inbody_data = parse_inbody(raw_text)
    chunks = create_chunks(inbody_data)

    # force_rebuild=True -> always re-embed and re-index from this PDF's
    # chunks, instead of silently reusing whatever was saved before.
    vector_store = VectorStore(
        chunks=chunks,
        path="data/vector_store",
        force_rebuild=True,
    )

    tokenizer, model = load_llm()

    pipeline = RAGPipeline(
        vector_store,
        tokenizer,
        model,
        inbody_data=inbody_data,
        activity_level="moderate",  # change to sedentary/light/active/very_active as needed
    )

    answer = pipeline.answer(question)

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(answer)


if __name__ == "__main__":
    main()