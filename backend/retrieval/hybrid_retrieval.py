from backend.retrieval.vector_retrieval import retrieve_from_vector
from backend.retrieval.kg_retrieval import retrieve_from_kg


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    question: str,
    vector_k: int = 5,
    kg_limit: int = 10
):
    """
    Retrieve context from both:

    1. FAISS vector database
    2. Neo4j knowledge graph

    Returns both contexts separately so the generation
    layer can combine them later.
    """

    # --------------------------------------------------------
    # VECTOR RETRIEVAL
    # --------------------------------------------------------

    vector_results = retrieve_from_vector(
        question,
        k=vector_k
    )

    # --------------------------------------------------------
    # KNOWLEDGE GRAPH RETRIEVAL
    # --------------------------------------------------------

    kg_results = retrieve_from_kg(
        question,
        limit=kg_limit
    )

    # --------------------------------------------------------
    # RETURN BOTH
    # --------------------------------------------------------

    return {
        "question": question,
        "vector_results": vector_results,
        "kg_results": kg_results,
    }


# ============================================================
# FORMAT VECTOR CONTEXT
# ============================================================

def format_vector_context(vector_results):

    if not vector_results:
        return "No vector context found."

    output = []

    for result in vector_results:

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
            f"""
SOURCE: {source}
PAGE: {page}

{result['text']}
"""
        )

    return "\n".join(output)


# ============================================================
# FORMAT KG CONTEXT
# ============================================================

def format_kg_context(kg_results):

    if not kg_results:
        return "No knowledge graph context found."

    output = []

    for result in kg_results:

        entity = result.get(
            "entity",
            "Unknown"
        )

        entity_type = result.get(
            "type",
            "Unknown"
        )

        output.append(
            f"ENTITY: {entity}\n"
            f"TYPE: {entity_type}"
        )

        relationships = result.get(
            "relationships",
            []
        )

        seen = set()

        for relationship in relationships:

            if not relationship:
                continue

            source = relationship.get(
                "source"
            )

            relation = relationship.get(
                "relationship"
            )

            target = relationship.get(
                "target"
            )

            key = (
                source,
                relation,
                target
            )

            if key in seen:
                continue

            seen.add(key)

            output.append(
                f"RELATIONSHIP: "
                f"{source} --{relation}--> {target}"
            )

        output.append("")

    return "\n".join(output)


# ============================================================
# FORMAT COMPLETE HYBRID CONTEXT
# ============================================================

def format_hybrid_context(hybrid_results):

    vector_context = format_vector_context(
        hybrid_results["vector_results"]
    )

    kg_context = format_kg_context(
        hybrid_results["kg_results"]
    )

    return f"""
================ VECTOR CONTEXT ================

{vector_context}

================ KNOWLEDGE GRAPH CONTEXT ================

{kg_context}

==========================================================
"""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    results = hybrid_retrieve(
        question,
        vector_k=5,
        kg_limit=10
    )

    print("\n" + "=" * 60)
    print("HYBRID CONTEXT")
    print("=" * 60)

    print(
        format_hybrid_context(results)
    )