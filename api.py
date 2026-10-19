
import json
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from agent import ask_agent
from schemas import ChatRequest, ChatResponse, SourceInfo


app = FastAPI(
    title="GeoSpatial AI Agent API",
    description="Local RAG + GIS Analysis Agent",
    version="0.1.0",
)

# Only allow local frontend development by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def extract_sources(text: str) -> list[SourceInfo]:
    """Extract source/page metadata from RAG tool results."""

    pattern = r"\[Source:\s*([^,\]\n]+),\s*Page:\s*([^\]\n]+)\]"

    matches = re.findall(pattern, text)

    sources = []
    seen = set()

    for filename, page in matches:
        filename = filename.strip()
        page = page.strip()

        try:
            page_value = int(page)
        except ValueError:
            page_value = page

        key = (filename, str(page_value))

        if key not in seen:
            seen.add(key)
            sources.append(
                SourceInfo(
                    source=filename,
                    page=page_value,
                )
            )

    return sources


def build_response(result: dict) -> ChatResponse:
    """Convert LangGraph messages into a structured response."""

    messages = result["messages"]

    tool_names = []
    stand_ids = []
    feature_ids = []
    sources = []
    seen_sources = set()

    answer = ""

    # Record tool calls made by the model.
    called_tools = {}

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                called_tools[call["id"]] = call["name"]
                tool_names.append(call["name"])

    # Extract data from executed tools, not from LLM prose.
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_name = called_tools.get(
            message.tool_call_id,
            message.name,
        )

        # Do not treat failed tool executions as valid data.
        if message.status == "error":
            continue

        content = message.content

        if not isinstance(content, str):
            continue

        # GIS tool results
        if tool_name in {
            "get_biomass_loss",
            "get_stable_forest",
        }:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            for stand in data.get("stands", []):
                if not isinstance(stand, dict):
                    continue

                polygon_id = stand.get("POLYGON_ID")

                if polygon_id is not None:
                    try:
                        stand_ids.append(int(polygon_id))
                    except (TypeError, ValueError):
                        pass
                
                feature_id = stand.get("FEATURE_ID")

                if feature_id is not None:
                    try:
                        feature_ids.append(int(feature_id))
                    except (TypeError, ValueError):
                        pass

        # Research document citations
        elif tool_name == "search_research_report":
            for source in extract_sources(content):
                key = (source.source, str(source.page))

                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append(source)

    # Retrieve final AI response
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if not message.tool_calls:
                content = message.content

                if isinstance(content, str):
                    answer = content.strip()

                elif isinstance(content, list):
                    answer = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                    ).strip()

                break

    if not answer:
        answer = "模型未生成有效回答，请检查 Agent 日志。"

    # Deduplicate stand IDs while preserving order.
    stand_ids = list(dict.fromkeys(stand_ids))

    feature_ids = list(dict.fromkeys(feature_ids))

    return ChatResponse(
        answer=answer,
        stand_ids=stand_ids,
        feature_ids=feature_ids,
        sources=sources,
        tool_calls=tool_names,
    )


@app.get("/")
def root():
    return {
        "message": "GeoSpatial AI Agent API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = ask_agent(request.question)

        return build_response(result)

    except Exception as exc:
        # Log detailed errors on the server.
        print(f"Agent error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed. Check server logs.",
        ) from exc
