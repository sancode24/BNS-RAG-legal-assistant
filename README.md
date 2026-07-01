# BNS 2023 Legal Assistant (RAG)

A Streamlit-based Retrieval-Augmented Generation (RAG) assistant for the **Bharatiya Nyaya Sanhita (BNS), 2023**.

The app extracts text from a BNS PDF, chunks it, creates local sentence-transformer embeddings, retrieves relevant excerpts for a user question, and asks Gemini to answer using only the retrieved context.

> ⚠️ This project is for educational purposes only. It is **not legal advice**.

## Features

- PDF text extraction using `pypdf`
- Chunking with overlap for better retrieval
- Local vector search using NumPy cosine similarity
- Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Optional cross-encoder reranking for better retrieval quality
- Gemini REST API integration
- Streamlit chat interface
- Local indexing, no vector database server required

## Project structure

```text
.
├── legal_assistant/
│   ├── __init__.py
│   ├── chunking.py          # Splits extracted PDF text into chunks
│   ├── config.py            # Environment variables and constants
│   ├── gemini_rest.py       # Gemini REST API call
│   ├── ingest.py            # Builds local embedding index
│   ├── local_index.py       # Local vector search and optional reranking
│   ├── pdf_extract.py       # Extracts text page-wise from PDF
│   └── streamlit_app.py     # Streamlit UI
├── data/
│   └── .gitkeep             # Place BNS2023.pdf here locally
├── .env.example             # Example environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## What NOT to upload to GitHub

Do **not** upload:

- `.env` — contains your private Gemini API key
- `bns_index/` — generated embeddings/index files
- `embeddings.npy`, `chunks.jsonl` — generated files
- `.venv/` or `venv/` — local virtual environment
- `__pycache__/`
- Large PDFs such as `BNS2023.pdf`, unless you are sure you have permission to redistribute them

These are already covered in `.gitignore`.

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Copy `.env.example` to `.env`:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Then edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
BNS_PDF_PATH=./data/BNS2023.pdf
INDEX_DIR=./bns_index
```

### 5. Add the PDF locally

Create a `data/` folder if it does not exist and place your BNS PDF there:

```text
data/BNS2023.pdf
```

The PDF is not committed to GitHub by default.

### 6. Build the index

```bash
python -m legal_assistant.ingest
```

To rebuild from scratch:

```bash
python -m legal_assistant.ingest --reset
```

### 7. Run the app

```bash
streamlit run legal_assistant/streamlit_app.py
```

## How it works

1. `pdf_extract.py` extracts text from the PDF page by page.
2. `chunking.py` splits the text into overlapping chunks.
3. `ingest.py` embeds the chunks using Sentence Transformers.
4. `local_index.py` stores embeddings locally and retrieves relevant chunks.
5. `rag_pipeline.py` builds a grounded prompt using retrieved excerpts.
6. `gemini_rest.py` sends the prompt to Gemini.
7. `streamlit_app.py` provides the chat UI.

## Notes

- If the PDF is scanned/image-only, text extraction may not work well.
- The first indexing run may take time because the embedding model is downloaded.
- Gemini free-tier limits may cause 429 errors. You can reduce context size using the variables in `.env.example`.

## Future improvements

- Add official source links/citations
- Add tests for chunking and retrieval
- Add a PDF upload option in the Streamlit UI
- Add ChromaDB or FAISS as an optional vector store
- Deploy the Streamlit app
