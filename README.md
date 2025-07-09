# Multimodal Retrieval-Augmented Generation (RAG) System

This repository implements a multimodal Retrieval-Augmented Generation (RAG) system capable of processing and retrieving information from both textual and visual content (e.g., tables, graphs, and charts) within PDF and DOCX documents. The system supports semantic search over text and images, integrates with an LLM (Mistral), and includes a web-based user interface for querying and interaction.

## Features

- Multimodal data extraction from PDF and DOCX (text, tables, images)
- Sentence-BERT and CLIP-based dense embeddings
- FAISS vector indices for efficient retrieval
- Unified semantic search over text and images
- LLM integration with Mistral (CoT, zero-shot, few-shot prompting)
- Web interface with query + image upload support
- Embedding space visualization (t-SNE)

## System Architecture

Documents (PDF/DOCX) <br>
│ <br>
▼ <br>
Data Extraction (text, tables, images, OCR, captions) <br>
│ <br>
▼ <br>
Embeddings (Sentence-BERT, CLIP) → FAISS indices <br>
│ <br>
▼ <br>
Semantic Search (text/image/multimodal queries) <br>
│ <br>
▼ <br>
LLM (Mistral) → Answer Generation <br>
│ <br>
▼ <br>
Frontend (Flask-based interface) <br>


## Directory Structure

├── data/ # Input documents (PDF, DOCX)
├── output/ # Extracted chunks and images
├── extracted_data/ # Preprocessed chunk pickle
├── embeddings_data/ # FAISS indices and stats
├── src/
│ ├── extraction/ # data_extraction.py
│ ├── embeddings/ # text_image_embeddings.py
│ ├── retrieval/ # retrieval.py
│ ├── llm/ # llm_integration.py
│ └── semantic/ # semantic_search.py
├── frontend/ # Flask app and templates
├── prepare_chunks_for_embeddings.py
├── requirements.txt
└── README.md


## 🛠️ Technologies Used

### 🧑‍💻 Programming Language
- **Python 3.8+**

### 📄 Document Processing
- **PyMuPDF** – PDF text, table, and image extraction
- **EasyOCR** – Optical character recognition for text in images
- **Tesseract** – Fallback OCR engine

### 🧠 Embedding Models
- **Sentence-BERT** – `all-MiniLM-L6-v2` for textual embeddings
- **CLIP** – `ViT-B/32` for image embeddings
- **BLIP** – Image captioning for descriptive embeddings

### 🔍 Vector Search
- **FAISS** – Efficient similarity search for high-dimensional vectors

### 🤖 Language Model
- **Mistral LLM API** – For zero-shot, few-shot, and CoT-based answer generation

### 🌐 Web Application
- **Flask** – Lightweight backend for chat and file serving

### 📊 Visualization
- **Matplotlib** – For plotting
- **t-SNE (via scikit-learn)** – Dimensionality reduction for embedding visualization


## Setup Instructions

### 1. Clone the repository


```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag

python -m venv venv  # create the virtual environment
venv\Scripts\activate

pip install -r requirements.txt #install dependencies

```