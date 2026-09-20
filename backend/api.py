from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.ingestion.vector_ingestion import create_vector_store
from backend.ingestion.kg_ingestion import process_pdf
from backend.generation.answer_generator import generate_answer

from backend.database.document_registry import (
    calculate_file_hash,
    get_document,
    register_document,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstores"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Hybrid RAG Backend",
    description="Backend API for Vector Search + Knowledge Graph RAG",
    version="1.0.0",
)


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):
    question: str


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Hybrid RAG Backend is running.",
        "docs": "/docs",
        "health": "/health"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "hybrid-rag-backend"
    }


# =========================================================
# PROCESS PDF
# =========================================================

@app.post("/process")
async def process_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    filename = Path(file.filename).name
    pdf_path = UPLOAD_DIR / filename

    try:

        # -------------------------------------------------
        # Save uploaded PDF
        # -------------------------------------------------

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -------------------------------------------------
        # Calculate file hash
        # -------------------------------------------------

        file_hash = calculate_file_hash(
            str(pdf_path)
        )

        print("\n" + "=" * 60)
        print("DOCUMENT CHECK")
        print("=" * 60)
        print(f"Filename : {filename}")
        print(f"SHA-256  : {file_hash}")

        # -------------------------------------------------
        # Check whether document already exists
        # -------------------------------------------------

        existing_document = get_document(
            file_hash
        )

        if existing_document:

            print("\nDocument already processed.")
            print("Skipping ingestion.")

            return {
                "success": True,
                "already_processed": True,
                "message": (
                    "This document has already been processed. "
                    "Ingestion was skipped."
                ),
                "filename": existing_document.get(
                    "filename",
                    filename
                ),
                "file_hash": file_hash
            }

        # -------------------------------------------------
        # New document
        # -------------------------------------------------

        print("\nNew document detected.")
        print("Starting ingestion...")

        # -------------------------------------------------
        # Vector ingestion
        # -------------------------------------------------

        print("\n[1/2] Creating vector database...")

        create_vector_store(
            str(pdf_path),
            str(VECTORSTORE_DIR)
        )

        # -------------------------------------------------
        # Knowledge graph ingestion
        # -------------------------------------------------

        print("\n[2/2] Creating knowledge graph...")

        process_pdf(
            str(pdf_path)
        )

        # -------------------------------------------------
        # Register document
        # -------------------------------------------------

        register_document(
            file_hash=file_hash,
            filename=filename,
            file_size=pdf_path.stat().st_size,
            vectorstore_path=str(VECTORSTORE_DIR)
        )

        print("\nDocument registered successfully.")

        print("=" * 60)

        return {
            "success": True,
            "already_processed": False,
            "message": (
                "PDF processed and registered successfully."
            ),
            "filename": filename,
            "file_hash": file_hash
        }

    except Exception as e:

        print(
            f"\nDocument processing error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:

        await file.close()


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        answer = generate_answer(
            question
        )

        return {
            "success": True,
            "question": question,
            "answer": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )