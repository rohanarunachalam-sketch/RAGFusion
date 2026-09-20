import json

from langchain_openai import ChatOpenAI

from backend.config import (
    OPENAI_API_KEY,
    LLM_MODEL
)

from backend.ingestion.pdf_loader import load_pdf

from backend.database.neo4j_store import Neo4jStore


# ============================================================
# KNOWLEDGE EXTRACTION
# ============================================================

def extract_knowledge(text):

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    prompt = f"""
You are a Knowledge Graph extraction system.

Analyze the following document text.

Extract important:

1. Entities
2. Relationships between entities

The document can belong to ANY domain.

Do not assume a specific industry.

Return ONLY valid JSON in this exact structure:

{{
    "entities": [
        {{
            "name": "entity name",
            "type": "entity type"
        }}
    ],
    "relationships": [
        {{
            "source": "source entity",
            "relation": "relationship",
            "target": "target entity"
        }}
    ]
}}

Rules:

- Extract only information explicitly present in the text.
- Do not invent entities.
- Do not invent relationships.
- Keep entity names consistent.
- Avoid duplicate entities.
- Use short, meaningful relationship names.
- Include important technologies, systems, organizations,
  people, products, components, concepts and other relevant entities.

DOCUMENT:

{text}
"""

    response = llm.invoke(
        prompt
    )

    content = response.content

    if isinstance(content, list):

        content = "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = content.strip()

    if content.startswith("```"):

        content = content.replace(
            "```json",
            ""
        )

        content = content.replace(
            "```",
            ""
        )

        content = content.strip()

    return json.loads(
        content
    )


# ============================================================
# ADD DOCUMENT ID TO ENTITIES
# ============================================================

def add_document_id(
    entities,
    document_id
):

    updated_entities = []

    for entity in entities:

        updated_entity = dict(
            entity
        )

        updated_entity["document_id"] = (
            document_id
        )

        updated_entities.append(
            updated_entity
        )

    return updated_entities


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path,
    document_id=None
):

    print(
        "\n" + "=" * 60
    )

    print(
        "KNOWLEDGE GRAPH INGESTION"
    )

    print(
        "=" * 60
    )

    if not document_id:

        raise ValueError(
            "Document ID is required for "
            "document-specific KG ingestion."
        )

    print(
        f"\nDocument ID: {document_id}"
    )

    # --------------------------------
    # Load PDF
    # --------------------------------

    print(
        "\n[1/4] Loading PDF..."
    )

    pages = load_pdf(
        pdf_path
    )

    print(
        f"Pages loaded: {len(pages)}"
    )

    # --------------------------------
    # Combine text
    # --------------------------------

    print(
        "\n[2/4] Preparing document text..."
    )

    full_text = "\n\n".join(
        f"PAGE {page['page']}\n{page['text']}"
        for page in pages
    )

    print(
        f"Characters: {len(full_text)}"
    )

    # --------------------------------
    # Extract knowledge
    # --------------------------------

    print(
        "\n[3/4] Extracting entities "
        "and relationships..."
    )

    knowledge = extract_knowledge(
        full_text
    )

    entities = knowledge.get(
        "entities",
        []
    )

    relationships = knowledge.get(
        "relationships",
        []
    )

    print(
        f"Entities      : {len(entities)}"
    )

    print(
        f"Relationships : {len(relationships)}"
    )

    # --------------------------------
    # Add document ID
    # --------------------------------

    entities = add_document_id(
        entities,
        document_id
    )

    # --------------------------------
    # Store in Neo4j
    # --------------------------------

    print(
        "\n[4/4] Storing knowledge graph..."
    )

    store = Neo4jStore()

    try:

        message = store.test_connection()

        print(
            message
        )

        store.create_graph(
            entities,
            relationships,
            document_id=document_id
        )

    finally:

        store.close()

    print(
        "\n" + "=" * 60
    )

    print(
        "KNOWLEDGE GRAPH INGESTION COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = input(
        "\nEnter PDF path: "
    ).strip()

    document_id = input(
        "Enter document ID: "
    ).strip()

    process_pdf(
        pdf_path,
        document_id=document_id
    )