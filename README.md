# Legal_Assistant_Agent

A comprehensive pipeline for processing legal documents that performs text extraction, named entity recognition, semantic embedding, and vector-based search capabilities.

## Features

- **PDF Text Extraction**: Extract text content from legal PDF documents
- **Named Entity Recognition**: Identify legal entities using GLiNER model
- **Document Chunking**: Intelligently split documents into manageable chunks for better processing
- **Semantic Embedding**: Convert text into vector representations using BERT
- **Vector Storage**: Store and index document vectors using FAISS for efficient similarity search
- **Semantic Search**: Perform semantic searches to find relevant documents based on meaning, not just keywords

## System Architecture

The system is built with a modular architecture:
- `config.py`: Configuration settings
- `models.py`: Model loading and initialization
- `text_extraction.py`: PDF text extraction
- `ner_processor.py`: Named Entity Recognition
- `embedding.py`: Text embedding 
- `chunking.py`: Document chunking functionality
- `vector_store.py`: FAISS vector storage operations
- `search.py`: Semantic search capabilities
- `pipeline.py`: Main document processing pipeline
- `main.py`: Entry point for processing and searching
- `run_search.py`: Standalone script for search functionality

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AayushChhabra42/Legal_Assistant_Agent/edit/main/README.md
cd legal-document-pipeline
