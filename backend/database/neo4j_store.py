from neo4j import GraphDatabase

from backend.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


class Neo4jStore:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(
                NEO4J_USERNAME,
                NEO4J_PASSWORD
            )
        )

    def close(self):
        self.driver.close()

    def test_connection(self):

        with self.driver.session() as session:

            result = session.run(
                "RETURN 'Neo4j connection successful' AS message"
            )

            return result.single()["message"]

    def clear_database(self):

        with self.driver.session() as session:

            session.run(
                "MATCH (n) DETACH DELETE n"
            )

    def create_graph(self, entities, relationships):

        with self.driver.session() as session:

            for entity in entities:

                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    """,
                    name=entity["name"],
                    type=entity["type"]
                )

            for relationship in relationships:

                session.run(
                    """
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})

                    MERGE (source)-[r:RELATED_TO {
                        type: $relation
                    }]->(target)
                    """,
                    source=relationship["source"],
                    target=relationship["target"],
                    relation=relationship["relation"]
                )