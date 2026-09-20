import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REGISTRY_FILE = DATA_DIR / "document_registry.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FILE HASH
# =========================================================

def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of the complete file.

    The hash is based on file content, not the filename.
    Identical files will therefore have the same hash.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# LOAD REGISTRY
# =========================================================

def load_registry() -> dict:
    """
    Load the document registry.

    Creates an empty registry if the file does not exist.
    """

    if not REGISTRY_FILE.exists():
        return {}

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

            return {}

    except (json.JSONDecodeError, OSError):

        return {}


# =========================================================
# SAVE REGISTRY
# =========================================================

def save_registry(registry: dict):
    """
    Save the document registry to disk.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = REGISTRY_FILE.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_file.replace(REGISTRY_FILE)


# =========================================================
# FIND DOCUMENT
# =========================================================

def get_document(file_hash: str):
    """
    Return document information if the hash already exists.
    """

    registry = load_registry()

    return registry.get(file_hash)


# =========================================================
# CHECK DUPLICATE
# =========================================================

def document_exists(file_hash: str) -> bool:
    """
    Check whether a document has already been processed.
    """

    registry = load_registry()

    return file_hash in registry


# =========================================================
# REGISTER DOCUMENT
# =========================================================

def register_document(
    file_hash: str,
    filename: str,
    file_size: int,
    vectorstore_path: str | None = None
):
    """
    Register a successfully processed document.
    """

    registry = load_registry()

    registry[file_hash] = {
        "filename": filename,
        "file_hash": file_hash,
        "file_size": file_size,
        "vectorstore_path": vectorstore_path,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "processed"
    }

    save_registry(registry)


# =========================================================
# REMOVE DOCUMENT
# =========================================================

def remove_document(file_hash: str):
    """
    Remove a document from the registry.
    """

    registry = load_registry()

    if file_hash in registry:

        del registry[file_hash]

        save_registry(registry)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DOCUMENT REGISTRY TEST")
    print("=" * 60)

    print(f"\nRegistry file:")
    print(REGISTRY_FILE)

    registry = load_registry()

    print(f"\nRegistered documents: {len(registry)}")

    for file_hash, document in registry.items():

        print("\nHash:", file_hash)

        print(
            "Filename:",
            document.get("filename")
        )

        print(
            "Status:",
            document.get("status")
        )

    print("\nRegistry test completed.")