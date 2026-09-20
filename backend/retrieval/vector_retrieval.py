from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VECTORSTORE_BASE_PATH = (
    PROJECT_ROOT / "data" / "vectorstores"
)


# ============================================================
# VECTOR STORE
# ============================================================

def load_vector_store(
    document_id: str
):

    if not document_id:
        raise ValueError(
            "Document ID is required."
        )

    vectorstore_path = (
        VECTORSTORE_BASE_PATH / document_id
    )

    if not vectorstore_path.exists():

        raise FileNotFoundError(
            f"FAISS vector store not found for "
            f"document: {document_id}"
        )

    index_file = (
        vectorstore_path / "index.faiss"
    )

    if not index_file.exists():

        raise FileNotFoundError(
            f"FAISS index not found: {index_file}"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vector_store = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_from_vector(
    question: str,
    document_id: str,
    k: int = 5
):

    if not question.strip():
        return []

    if not document_id:
        raise ValueError(
            "Document ID is required for vector retrieval."
        )

    vector_store = load_vector_store(
        document_id
    )

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    results = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        results.append({

            "rank": index,

            "text": document.page_content,

            "metadata": document.metadata,

        })

    return results


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_vector_results(
    results
):

    if not results:

        return (
            "No relevant vector results found."
        )

    output = []

    for result in results:

        metadata = result.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        document_id = metadata.get(
            "document_id",
            "Unknown"
        )

        output.append(
            f"\n{result['rank']}. "
            f"Source: {source} | "
            f"Page: {page} | "
            f"Document: {document_id}"
        )

        output.append(
            f"   {result['text']}"
        )

    return "\n".join(
        output
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "VECTOR RETRIEVAL TEST"
    )

    print(
        "=" * 60
    )

    document_id = input(
        "\nEnter document ID: "
    ).strip()

    question = input(
        "Enter your question: "
    ).strip()

    results = retrieve_from_vector(
        question,
        document_id,
        k=5
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "VECTOR RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        format_vector_results(
            results
        )
    )