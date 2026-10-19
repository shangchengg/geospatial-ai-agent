
from langchain_core.tools import tool

from rag import retrieve_context


@tool
def search_research_report(question: str) -> str:
    """
    Search the forest research report for information about
    LiDAR, AGB modeling, land-cover classification,
    forest changes, methods, and ecological explanations.

    Use this tool for research knowledge questions.
    Do not use it to calculate stand-level GIS statistics.
    """
    documents = retrieve_context(question)

    if not documents:
        return "No relevant report content was retrieved."

    sections = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        sections.append(
            f"[Source: {source}, Page: {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(sections)
