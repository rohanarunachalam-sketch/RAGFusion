from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_processor import create_chunks
from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)


def create_vector_store(
    pdf_path: str,
    output_path: str,
    document_id: str | None = None
):

    print("\n" + "=" * 60)
    print("VECTOR INGESTION")
    print("=" * 60)

    # ---------------------------------
    # 1. Load PDF
    # ---------------------------------

    print("\n[1/4] Loading PDF...")

    pages = load_pdf(pdf_path)

    print(f"Pages loaded: {len(pages)}")

    # ---------------------------------
    # 2. Create chunks
    # ---------------------------------

    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(pages)

    print(f"Chunks created: {len(chunks)}")

    # ---------------------------------
    # 3. Convert to LangChain Documents
    # ---------------------------------

    documents = []

    for item in chunks:

        metadata = dict(
            item.get("metadata", {})
        )

        # Add document ID to every chunk
        if document_id:
            metadata["document_id"] = document_id

        documents.append(
            Document(
                page_content=item["text"],
                metadata=metadata
            )
        )

    # ---------------------------------
    # 4. Create embeddings + FAISS
    # ---------------------------------

    print("\n[3/4] Creating embeddings...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    print("\n[4/4] Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    # ---------------------------------
    # Save vector store
    # ---------------------------------

    output_path = Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(output_path)
    )

    print("\n" + "=" * 60)
    print("VECTOR INGESTION COMPLETE")
    print("=" * 60)

    print(f"Pages      : {len(pages)}")
    print(f"Chunks     : {len(chunks)}")
    print(f"Document ID: {document_id}")
    print(f"FAISS      : {output_path}")

    return vector_store


if __name__ == "__main__":

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    document_id = input(
        "Enter document ID (optional): "
    ).strip()

    output_path = "data/vectorstores"

    create_vector_store(
        pdf_path,
        output_path,
        document_id=document_id or None
    )