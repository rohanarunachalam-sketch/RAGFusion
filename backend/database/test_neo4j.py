from neo4j import GraphDatabase

from backend.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)


print("URI:", NEO4J_URI)
print("Username:", NEO4J_USERNAME)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    with driver.session() as session:
        result = session.run(
            "RETURN 'Connection successful' AS message"
        )

        print(result.single()["message"])

except Exception as e:
    print("\nConnection failed:")
    print(type(e).__name__)
    print(e)

finally:
    driver.close()