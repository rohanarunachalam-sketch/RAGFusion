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

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="RAGFusion Backend",
    description="Backend API for Vector Search + Knowledge Graph RAG",
    version="1.0.0",
)


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(BaseModel):

    question: str

    document_id: str | None = None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "RAGFusion Backend is running.",
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
        "service": "ragfusion-backend"
    }


# =========================================================
# PROCESS PDF
# =========================================================

@app.post("/process")
async def process_document(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # VALIDATE FILE
    # -----------------------------------------------------

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

    filename = Path(
        file.filename
    ).name

    pdf_path = UPLOAD_DIR / filename

    try:

        # -------------------------------------------------
        # SAVE PDF
        # -------------------------------------------------

        with open(
            pdf_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # -------------------------------------------------
        # CALCULATE DOCUMENT HASH
        # -------------------------------------------------

        file_hash = calculate_file_hash(
            str(pdf_path)
        )

        document_id = file_hash

        print(
            "\n" + "=" * 60
        )

        print(
            "DOCUMENT CHECK"
        )

        print(
            "=" * 60
        )

        print(
            f"Filename    : {filename}"
        )

        print(
            f"Document ID : {document_id}"
        )

        # -------------------------------------------------
        # CHECK DUPLICATE
        # -------------------------------------------------

        existing_document = get_document(
            document_id
        )

        if existing_document:

            print(
                "\nDocument already processed."
            )

            print(
                "Skipping ingestion."
            )

            return {

                "success": True,

                "already_processed": True,

                "message": (
                    "This document has already been "
                    "processed. Ingestion was skipped."
                ),

                "filename": existing_document.get(
                    "filename",
                    filename
                ),

                "document_id": document_id

            }

        # -------------------------------------------------
        # NEW DOCUMENT
        # -------------------------------------------------

        print(
            "\nNew document detected."
        )

        print(
            "Starting ingestion..."
        )

        # -------------------------------------------------
        # DOCUMENT-SPECIFIC VECTOR STORE
        # -------------------------------------------------

        document_vectorstore_dir = (
            VECTORSTORE_DIR / document_id
        )

        print(
            "\nVector Store:"
        )

        print(
            document_vectorstore_dir
        )

        # -------------------------------------------------
        # VECTOR INGESTION
        # -------------------------------------------------

        print(
            "\n[1/2] Creating document vector database..."
        )

        create_vector_store(

            str(pdf_path),

            str(
                document_vectorstore_dir
            ),

            document_id=document_id

        )

        # -------------------------------------------------
        # KNOWLEDGE GRAPH INGESTION
        # -------------------------------------------------

        print(
            "\n[2/2] Creating knowledge graph..."
        )

        process_pdf(

            str(pdf_path),

            document_id=document_id

        )

        # -------------------------------------------------
        # REGISTER DOCUMENT
        # -------------------------------------------------

        register_document(

            file_hash=document_id,

            filename=filename,

            file_size=pdf_path.stat().st_size,

            vectorstore_path=str(
                document_vectorstore_dir
            )

        )

        print(
            "\nDocument registered successfully."
        )

        print(
            "=" * 60
        )

        # -------------------------------------------------
        # SUCCESS RESPONSE
        # -------------------------------------------------

        return {

            "success": True,

            "already_processed": False,

            "message": (
                "PDF processed and registered successfully."
            ),

            "filename": filename,

            "document_id": document_id

        }

    except Exception as e:

        print(
            f"\nDocument processing error: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Document processing failed: {str(e)}"
            )

        )

    finally:

        await file.close()


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    # -----------------------------------------------------
    # CLEAN QUESTION
    # -----------------------------------------------------

    question = request.question.strip()

    # -----------------------------------------------------
    # VALIDATE QUESTION
    # -----------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # -----------------------------------------------------
    # VALIDATE DOCUMENT ID
    # -----------------------------------------------------

    if not request.document_id:

        raise HTTPException(

            status_code=400,

            detail=(
                "Document ID is required. "
                "Please process a document first."
            )

        )

    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    try:

        answer = generate_answer(

            question=question,

            document_id=request.document_id

        )

        return {

            "success": True,

            "question": question,

            "document_id": request.document_id,

            "answer": answer

        }

    except Exception as e:

        print(
            f"\nAnswer generation error: {e}"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Answer generation failed: {str(e)}"
            )

        )