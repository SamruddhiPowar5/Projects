# Enterprise-grade RAG System with Grounded Citations

This project implements a retrieval-augmented generation (RAG) system for enterprise documents.

Features:
- PDF ingestion
- Chunking with metadata
- Vector search using FAISS
- Query rewriting to improve retrieval recall
- LLM-based answering
- Chunk-level inline citations
- Automatic grounding validation
- Multi-question CLI interface

Architecture:
PDFs → chunking → embeddings → FAISS → retriever → citation-controlled prompt → LLM

How to run:

1. Install dependencies
   pip install -r requirements.txt

2. Set environment variable
   export OPENAI_API_KEY=...

3. Build the index
   python build_index.py

4. Run the app
   python test_retrieval.py
