from pathlib import Path
import shutil

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Project paths

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DB_DIR = BASE_DIR / "chroma_db"


# Load PDF documents

documents = []

pdf_files = list(DOCS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in the data folder."
    )


for pdf_path in pdf_files:

    print(f"Loading: {pdf_path.name}")

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        text = page.extract_text()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )
        )


print(
    f"Loaded {len(documents)} document pages."
)


# Split documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(
    documents
)


print(
    f"Created {len(chunks)} text chunks."
)


# Reset vector database when rebuilding

if DB_DIR.exists():

    shutil.rmtree(DB_DIR)


# Create local Qwen embeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# Create Chroma vector database

vector_store = Chroma(
    collection_name="geospatial_report",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


# Add documents to vector database

vector_store.add_documents(
    documents=chunks
)


print(
    "Vector database created successfully."
)

print(
    f"Database location: {DB_DIR}"
)