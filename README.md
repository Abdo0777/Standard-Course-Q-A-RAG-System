# Course Q&A RAG System

A Retrieval-Augmented Generation (RAG) system that answers student questions using course materials across multiple file formats. Answers are generated strictly from retrieved content and returned with source attribution.

## Overview

This project implements a complete RAG pipeline from scratch: document ingestion, chunking, embedding, vector search, and answer generation, wrapped in a Gradio interface. It supports course materials in four different formats and can scope retrieval to a specific course or search across all of them.

## Features

- Multi-format ingestion: PDF, TXT, CSV, DOCX
- Text cleaning and recursive chunking for consistent retrieval quality
- Semantic search via sentence embeddings and a FAISS vector index
- Local LLM inference (no API key required)
- Source-attributed answers, with optional filtering by course
- Clean, minimal Gradio web interface

## Architecture

```
Course Files (PDF / TXT / CSV / DOCX)
            |
            v
   Load + Clean + Chunk
            |
            v
  Sentence-Transformer Embeddings
            |
            v
        FAISS Index
            |
            v
   Query -> Retrieve Top-K Chunks
            |
            v
    LLM (Qwen2.5-1.5B-Instruct)
            |
            v
   Answer + Cited Sources
```

## Tech Stack

| Component      | Library                                      |
|-----------------|-----------------------------------------------|
| PDF parsing     | `pypdf`                                       |
| DOCX parsing    | `python-docx`                                 |
| CSV parsing     | `pandas`                                       |
| Text chunking   | `langchain-text-splitters`                     |
| Embeddings      | `sentence-transformers` (`all-MiniLM-L6-v2`)   |
| Vector search   | `faiss-cpu`                                    |
| Language model  | `transformers` (`Qwen2.5-1.5B-Instruct`)       |
| Interface       | `gradio`                                       |

## Project Structure

```
course-qa-rag/
├── app.py                  # Full pipeline: ingestion, indexing, retrieval, LLM, UI
├── requirements.txt
├── course_materials/       # Source documents, organized by course
│   ├── course1_python_book.pdf
│   ├── course1_syllabus.txt
│   ├── course2_faq.csv
│   └── course3_ml_notes.docx
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/course-qa-rag.git
cd course-qa-rag
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your own course materials inside `course_materials/`, or use the sample set provided. File-to-course mappings are defined in the `FILES` list in `app.py`.

## Usage

```bash
python app.py
```

The app loads and indexes all course materials, downloads the embedding and language models on first run (cached afterward), then launches a local Gradio interface.

## How It Works

1. **Ingestion** — Each file type is parsed with a dedicated loader and normalized into plain text.
2. **Cleaning & Chunking** — Text is cleaned of noise and split into overlapping chunks (500 characters, 50-character overlap) to preserve context across boundaries.
3. **Embedding** — Each chunk is encoded into a dense vector using `all-MiniLM-L6-v2`.
4. **Indexing** — Vectors are stored in a FAISS flat index for exact nearest-neighbor search.
5. **Retrieval** — A user query is embedded and matched against the index, optionally filtered to a single course.
6. **Generation** — Retrieved chunks are inserted into a constrained prompt, and the LLM is instructed to answer only from the provided context.
7. **Attribution** — The originating source file(s) for the retrieved chunks are returned alongside the answer.

## Example

**Question:** What is a stack?
**Course:** Data Structures
**Answer:** A stack is a linear data structure that follows Last In First Out (LIFO) order. Elements are added and removed from the same end, called the top.
**Sources:** course2_faq.csv

## Notes

- Runs entirely on local compute; no external API keys required.
- For faster inference, a CUDA-enabled GPU is recommended when loading the LLM.
- Large source files (e.g. full textbook PDFs) may need to be excluded from version control — see `.gitignore`.

