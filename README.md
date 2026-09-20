# RAGFusion

RAGFusion is a **Hybrid RAG application** that allows users to upload PDF documents and ask questions about their content using AI.

It combines **Vector Search** and a **Knowledge Graph** to retrieve relevant information from the uploaded document and generate accurate answers using **GPT-4.1-mini**.

## How It Works

```text
Upload PDF
    ↓
Process Document
    ↓
Vector Search + Knowledge Graph
    ↓
Retrieve Relevant Information
    ↓
GPT-4.1-mini
    ↓
Answer
Key Features
Upload PDF documents
Ask questions using natural language
Vector-based semantic search with FAISS
Knowledge Graph retrieval using Neo4j
AI-generated answers using GPT-4.1-mini
Detects duplicate documents and avoids unnecessary re-processing
Streamlit frontend with FastAPI backend
Docker support for deployment
Technology

Python • Streamlit • FastAPI • LangChain • FAISS • Neo4j • OpenAI • Docker

Project Status

RAGFusion is currently a working prototype and is being developed further for production deployment.

Author

Rohan Arunachalam


This is much better for a GitHub repository because someone can understand the project in **30 s
