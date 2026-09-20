import os
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# OpenAI
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-5.6-luna"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)


# -----------------------------
# Neo4j
# -----------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME",
    "neo4j"
)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# -----------------------------
# Validation
# -----------------------------
def validate_config():
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if not NEO4J_URI:
        missing.append("NEO4J_URI")

    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")

    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}"
        )


if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully.")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Neo4j URI: {NEO4J_URI}")