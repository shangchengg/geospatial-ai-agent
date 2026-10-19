from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Project paths

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


# Load the same embedding model used during ingestion

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# Load the existing Chroma vector database

vector_store = Chroma(
    collection_name="geospatial_report",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


# User query

query = input(
    "\nAsk a question about the report: "
)


# Retrieve relevant chunks

results = vector_store.similarity_search(
    query,
    k=4,
)


# Display retrieved chunks

print("\nTop retrieved results:")


for index, document in enumerate(
    results,
    start=1,
):

    source = document.metadata.get(
        "source",
        "Unknown",
    )

    page = document.metadata.get(
        "page",
        "Unknown",
    )


    print(
        f"\n--- Result {index} ---"
    )

    print(
        f"Source: {source}"
    )

    print(
        f"Page: {page}"
    )

    print(
        document.page_content[:900]
    )