from neo4j import GraphDatabase

from backend.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


class Neo4jStore:

    # ========================================================
    # INITIALIZE
    # ========================================================

    def __init__(self):

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(
                NEO4J_USERNAME,
                NEO4J_PASSWORD
            )
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.driver.close()

    # ========================================================
    # TEST CONNECTION
    # ========================================================

    def test_connection(self):

        with self.driver.session() as session:

            result = session.run(
                "RETURN 'Neo4j connection successful' AS message"
            )

            return result.single()["message"]

    # ========================================================
    # CLEAR DATABASE
    # ========================================================

    def clear_database(self):

        with self.driver.session() as session:

            session.run(
                "MATCH (n) DETACH DELETE n"
            )

    # ========================================================
    # CREATE GRAPH
    # ========================================================

    def create_graph(
        self,
        entities,
        relationships,
        document_id
    ):

        if not document_id:

            raise ValueError(
                "Document ID is required."
            )

        with self.driver.session() as session:

            # ------------------------------------------------
            # CREATE ENTITIES
            # ------------------------------------------------

            for entity in entities:

                session.run(
                    """
                    MERGE (
                        e:Entity {
                            name: $name,
                            document_id: $document_id
                        }
                    )

                    SET e.type = $type
                    """,

                    name=entity["name"],

                    type=entity["type"],

                    document_id=document_id
                )

            # ------------------------------------------------
            # CREATE RELATIONSHIPS
            # ------------------------------------------------

            for relationship in relationships:

                session.run(
                    """
                    MATCH (
                        source:Entity {
                            name: $source,
                            document_id: $document_id
                        }
                    )

                    MATCH (
                        target:Entity {
                            name: $target,
                            document_id: $document_id
                        }
                    )

                    MERGE (
                        source
                    )-[r:RELATED_TO {
                        type: $relation
                    }]->(
                        target
                    )
                    """,

                    source=relationship["source"],

                    target=relationship["target"],

                    relation=relationship["relation"],

                    document_id=document_id
                )

    # ========================================================
    # DOCUMENT STATISTICS
    # ========================================================

    def get_document_stats(
        self,
        document_id
    ):

        if not document_id:

            raise ValueError(
                "Document ID is required."
            )

        with self.driver.session() as session:

            result = session.run(
                """
                MATCH (n:Entity)

                WHERE n.document_id = $document_id

                OPTIONAL MATCH (
                    n
                )-[r:RELATED_TO]->(
                    target:Entity
                )

                WHERE target.document_id = $document_id

                RETURN
                    count(DISTINCT n) AS entities,
                    count(DISTINCT r) AS relationships
                """,

                document_id=document_id
            )

            record = result.single()

            return {
                "entities": record["entities"],
                "relationships": record["relationships"]
            }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "NEO4J STORE TEST"
    )

    print(
        "=" * 60
    )

    store = Neo4jStore()

    try:

        print(
            "\n" + store.test_connection()
        )

        print(
            "\nNeo4j connection is working."
        )

    finally:

        store.close()