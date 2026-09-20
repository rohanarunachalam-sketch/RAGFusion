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

VECTORSTORE_PATH = Path("data/vectorstores")


# ============================================================
# VECTOR STORE
# ============================================================

def load_vector_store():

    if not VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            f"FAISS vector store not found: {VECTORSTORE_PATH}"
        )

    index_file = VECTORSTORE_PATH / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_file}"
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vector_store = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_from_vector(
    question: str,
    k: int = 5
):

    vector_store = load_vector_store()

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

def format_vector_results(results):

    if not results:
        return "No relevant vector results found."

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

        output.append(
            f"\n{result['rank']}. "
            f"Source: {source} | Page: {page}"
        )

        output.append(
            f"   {result['text']}"
        )

    return "\n".join(output)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("VECTOR RETRIEVAL TEST")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    results = retrieve_from_vector(
        question,
        k=5
    )

    print("\n" + "=" * 60)
    print("VECTOR RESULTS")
    print("=" * 60)

    print(
        format_vector_results(results)
    )