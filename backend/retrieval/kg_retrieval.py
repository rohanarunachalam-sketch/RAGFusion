import json
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase

from backend.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)


# ============================================================
# NEO4J DATABASE
# ============================================================

class KGDatabase:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def get_entity_types(self):

        query = """
        MATCH (n:Entity)
        RETURN DISTINCT n.type AS type
        ORDER BY type
        """

        with self.driver.session() as session:
            result = session.run(query)

            return [
                record["type"]
                for record in result
                if record["type"]
            ]

    # --------------------------------------------------------
    # ENTITY SEARCH
    # --------------------------------------------------------

    def search_specific_entity(self, search_terms, limit=10):

        query = """
        MATCH (n:Entity)

        WHERE any(
            term IN $search_terms
            WHERE
                toLower(n.name) = toLower(term)
                OR
                toLower(n.name) CONTAINS toLower(term)
        )

        OPTIONAL MATCH (n)-[r:RELATED_TO]-(connected:Entity)

        RETURN
            n.name AS entity,
            n.type AS type,
            collect(
                CASE
                    WHEN r IS NOT NULL
                    THEN {
                        source: startNode(r).name,
                        relationship: r.type,
                        target: endNode(r).name
                    }
                    ELSE NULL
                END
            ) AS relationships

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                search_terms=search_terms,
                limit=limit
            )

            return [
                record.data()
                for record in result
            ]

    # --------------------------------------------------------
    # ENTITY TYPE SEARCH
    # --------------------------------------------------------

    def search_by_type(
        self,
        entity_types,
        search_terms=None,
        limit=20
    ):

        search_terms = search_terms or []

        query = """
        MATCH (n:Entity)

        WHERE n.type IN $entity_types

        OPTIONAL MATCH (n)-[r:RELATED_TO]-(connected:Entity)

        RETURN
            n.name AS entity,
            n.type AS type,
            collect(
                CASE
                    WHEN r IS NOT NULL
                    THEN {
                        source: startNode(r).name,
                        relationship: r.type,
                        target: endNode(r).name
                    }
                    ELSE NULL
                END
            ) AS relationships

        ORDER BY n.name

        LIMIT $limit
        """

        with self.driver.session() as session:

            result = session.run(
                query,
                entity_types=entity_types,
                limit=limit
            )

            return [
                record.data()
                for record in result
            ]


# ============================================================
# LLM QUERY UNDERSTANDING
# ============================================================

def understand_question(question, available_types):

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    prompt = f"""
You are a generic Knowledge Graph query planner.

The Knowledge Graph can contain documents from ANY domain.

Available entity types:

{json.dumps(available_types, indent=2)}

User question:

{question}

Your task is to determine how the Knowledge Graph should
be searched.

Return ONLY valid JSON.

The JSON MUST have exactly this structure:

{{
    "search_terms": [],
    "entity_types": [],
    "search_mode": "entity"
}}

------------------------------------------------------------
SEARCH TERMS
------------------------------------------------------------

search_terms should contain specific entities mentioned
or strongly implied by the question.

Examples:

"What technologies does Next.js use?"

search_terms:
["Next.js"]

"What databases does the application use?"

search_terms:
["application"]

"What does Kubernetes provide?"

search_terms:
["Kubernetes"]

Do NOT put generic words such as:

"what"
"which"
"used"
"technologies"
"application"
"information"
"system"
"things"

unless they are genuinely specific entities.

------------------------------------------------------------
ENTITY TYPES
------------------------------------------------------------

entity_types MUST contain only types from the available
entity types list.

If the question asks about technologies:

["Technology"]

If the question asks about microservices:

["Microservice"]

If the question asks about frontend pages:

["Frontend Page"]

If the question asks about database tables:

["Database Table"]

Never invent an entity type.

------------------------------------------------------------
SEARCH MODE
------------------------------------------------------------

search_mode MUST be either:

"entity"

or

"type"

Use "entity" when the question refers to a specific
entity.

Example:

"What technologies does Next.js use?"

Return:

{{
    "search_terms": ["Next.js"],
    "entity_types": ["Technology"],
    "search_mode": "entity"
}}

Another example:

"What does Kubernetes use?"

Return:

{{
    "search_terms": ["Kubernetes"],
    "entity_types": [],
    "search_mode": "entity"
}}

Use "type" when the question asks generally about a
category of entities.

Example:

"What technologies are used in the web application?"

Return:

{{
    "search_terms": [],
    "entity_types": ["Technology"],
    "search_mode": "type"
}}

Another example:

"Which microservices are used?"

Return:

{{
    "search_terms": [],
    "entity_types": ["Microservice"],
    "search_mode": "type"
}}

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

If a specific entity is mentioned, prefer:

search_mode = "entity"

Do NOT return every entity of that type.

For example:

Question:
"What technologies does Next.js use?"

CORRECT:

{{
    "search_terms": ["Next.js"],
    "entity_types": ["Technology"],
    "search_mode": "entity"
}}

INCORRECT:

{{
    "search_terms": ["Next.js"],
    "entity_types": ["Technology"],
    "search_mode": "type"
}}

Return ONLY JSON.
"""

    response = llm.invoke(prompt)

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
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    return json.loads(content)


# ============================================================
# FORMAT RESULTS
# ============================================================

def format_results(results):

    if not results:
        return "No relevant knowledge graph results found."

    output = []

    for index, item in enumerate(results, start=1):

        output.append(
            f"{index}. {item['entity']} ({item['type']})"
        )

        relationships = item.get("relationships", [])

        # Remove duplicate relationships
        seen = set()

        for rel in relationships:

            if not rel:
                continue

            source = rel.get("source")
            relationship = rel.get("relationship")
            target = rel.get("target")

            key = (
                source,
                relationship,
                target
            )

            if key in seen:
                continue

            seen.add(key)

            output.append(
                f"   {source} --{relationship}--> {target}"
            )

    return "\n".join(output)


# ============================================================
# MAIN KG RETRIEVAL
# ============================================================

def retrieve_from_kg(question, limit=20):

    database = KGDatabase()

    try:

        # Get actual entity types from Neo4j
        available_types = database.get_entity_types()

        # Ask LLM how the question should be searched
        plan = understand_question(
            question,
            available_types
        )

        search_terms = plan.get(
            "search_terms",
            []
        )

        entity_types = plan.get(
            "entity_types",
            []
        )

        search_mode = plan.get(
            "search_mode",
            "entity"
        )

        print("\nSearch terms:")
        print(search_terms)

        print("\nEntity types:")
        print(entity_types)

        print("\nSearch mode:")
        print(search_mode)

        # ----------------------------------------------------
        # SPECIFIC ENTITY SEARCH
        # ----------------------------------------------------

        if search_mode == "entity":

            results = database.search_specific_entity(
                search_terms=search_terms,
                limit=limit
            )

        # ----------------------------------------------------
        # TYPE / CATEGORY SEARCH
        # ----------------------------------------------------

        elif search_mode == "type":

            results = database.search_by_type(
                entity_types=entity_types,
                search_terms=search_terms,
                limit=limit
            )

        else:

            results = []

        return results

    finally:

        database.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH RETRIEVAL TEST")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    results = retrieve_from_kg(question)

    print("\n" + "=" * 60)
    print("GRAPH RESULTS")
    print("=" * 60)

    print(
        format_results(results)
    )