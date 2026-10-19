from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)


# Project paths

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"


# Configuration

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:4b"
TOP_K = 4


# Load embedding model

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# Load Chroma vector database

vector_store = Chroma(
    collection_name="geospatial_report",
    embedding_function=embeddings,
    persist_directory=str(DB_DIR),
)


# Load local Qwen LLM

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.2,
    reasoning=False,
    num_predict=1024,
)


def retrieve_context(query: str):

    results = vector_store.similarity_search(
        query,
        k=TOP_K,
    )

    return results


def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
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

        context_parts.append(
            f"""
[Document {index}]
Source: {source}
Page: {page}

{document.page_content}
"""
        )

    return "\n".join(context_parts)



def answer_question(query: str):
    # 1. Retrieve documents
    documents = retrieve_context(query)

    # 2. Retrieval Debug
    print("\n===== RETRIEVAL DEBUG =====")

    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}")
        print("Page:", doc.metadata.get("page"))
        print(
            "Contains pzabove2:",
            "pzabove2" in doc.page_content.lower()
        )
        print("Text:", doc.page_content[:600])

    print("===========================\n")

    
    # Inspect terminology in retrieved documents
    import re

    print("\n===== VARIABLE CONTEXT CHECK =====")

    for i, doc in enumerate(documents, 1):
        text = doc.page_content
        page = doc.metadata.get("page")

        for match in re.finditer(
            r"\b(?:pzabove2|pzabove)\b",
            text,
            flags=re.IGNORECASE
        ):
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 180)

            snippet = text[start:end].replace("\n", " ")

            print(f"\nDocument {i}, Page {page}")
            print("Matched:", match.group())
            print("Context:", snippet)

    print("===============================\n")

    # 3. Build context
    context = build_context(documents)

    # 4. Build prompt
    prompt = f"""
You are a geospatial and remote sensing research assistant.

Answer the user's question using ONLY the retrieved context.

Rules:

1. Use only information supported by the retrieved context.

2. If evidence is insufficient, explicitly state that
   the available documents cannot confirm the answer.

3. Preserve exact technical variable names, including
   numbers, suffixes, capitalization and abbreviations.

4. Technical variable verification:
   - Before answering, identify the exact variable names
     associated with the requested model.
   - Copy each variable name character by character.
   - Preserve all numeric suffixes.
   - Do not substitute similarly named variables.
   - Verify the final answer against the retrieved context.

5. Preserve numerical values, statistical measures
   and units accurately.

6. Distinguish between explicitly stated information
   and information not mentioned in the documents.

7. Answer in the same language as the user's question.

8. Cite only supporting documents and page numbers
   found in the retrieved context, using:
   [Source: filename, Page: page number]

9. Do not claim that a statistical issue has been
   resolved unless the context supports that conclusion.

Retrieved Context:

{context}

User Question:

{query}

Answer:
"""

    # 5. Generate answer
    response = llm.invoke(prompt)

    # 6. LLM Debug
    print("\n===== LLM DEBUG =====")
    print("Content:", repr(response.content))
    print("Additional kwargs:", response.additional_kwargs)
    print("Metadata:", response.response_metadata)
    print("=====================\n")

    # 7. Process answer
    answer = response.content

    if not answer:
        answer = (
            "模型未返回有效答案。"
            "请检查 LLM DEBUG 输出。"
        )

    # 8. Technical terminology validation
    import re

    context_terms = set(
        re.findall(r"\b(?:zmax|zcv|pzabove2|pzabove)\b",
                   context.lower())
    )

    answer_terms = set(
        re.findall(r"\b(?:zmax|zcv|pzabove2|pzabove)\b",
                   answer.lower())
    )

    print("\n===== TERMINOLOGY CHECK =====")
    print("Context terms:", sorted(context_terms))
    print("Answer terms:", sorted(answer_terms))

    if "pzabove" in answer_terms and "pzabove2" in context_terms:
        print(
            "WARNING: Model used 'pzabove' while "
            "'pzabove2' exists in retrieved context."
        )

    print("=============================\n")

    # 9. Return
    return answer, documents


def print_sources(documents):

    print(
        "\nRetrieved Sources:"
    )

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            "Unknown",
        )

        source_key = (
            source,
            page,
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        print(
            f"- {source}, Page {page}"
        )


def main():

    print(
        "\nGeoSpatial RAG Assistant"
    )

    print(
        "Type 'exit' to quit.\n"
    )


    while True:

        query = input(
            "Question: "
        ).strip()


        if query.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye."
            )

            break


        if not query:
            continue


        print(
            "\nGenerating answer...\n"
        )


        answer, documents = (
            answer_question(
                query
            )
        )


        print(
            "Answer:\n"
        )

        print(
            answer
        )


        print_sources(
            documents
        )


        print(
            "\n"
            + "-" * 60
            + "\n"
        )


if __name__ == "__main__":
    main()