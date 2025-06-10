# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
import time
from typing import List, Optional, Dict, Any, AsyncIterator
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fastapi.exceptions
import asyncio
import json
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph
from agent.configuration import Configuration
from agent.state import ResearchState
from agent.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="DEBUG", name="agent.app")


class ResearchJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for research-related types."""

    def default(self, obj):
        if isinstance(obj, (HumanMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs,
            }
        return super().default(obj)


def serialize_event(event: Dict[str, Any]) -> str:
    """Serialize an event to JSON string, handling special types."""
    return json.dumps(event, cls=ResearchJSONEncoder)


# Define request/response models
class ChatMessage(BaseModel):
    role: str = "user"  # or "ai"
    content: str
    id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    initial_search_query_count: int = Field(
        default=3,
        ge=1,
        json_schema_extra={
            "description": "Number of initial search queries to generate",
            "title": "Initial Search Query Count",
        },
    )
    max_research_loops: int = Field(
        default=3,
        ge=1,
        json_schema_extra={
            "description": "Maximum number of research loops to perform",
            "title": "Max Research Loops",
        },
    )
    ollama_llm: str = Field(
        default="deepseek-r1",
        json_schema_extra={
            "description": "Model to use for research",
            "title": "LLM Model",
        },
    )

    model_config = {
        "json_schema_extra": {
            "title": "Chat Request",
            "description": "A request to the research agent to perform research and respond to a chat message",
        }
    }


# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Research agent endpoint using LangGraph
@app.post("/app/agent")
async def chat_stream(request: ChatRequest):
    """Research agent endpoint using LangGraph."""
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")
        logger.debug(f"Chat request details: {request.model_dump()}")

        # Create graph input from chat request
        graph_input = ResearchState(
            messages=[
                (
                    HumanMessage(content=msg.content)
                    if msg.role == "user"
                    else AIMessage(content=msg.content)
                )
                for msg in request.messages
            ],
            research_loop_count=[],
            search_query=[],
            web_research_result=[],
            sources_gathered=[],
            is_sufficient=None,
            knowledge_gap=None,
            follow_up_queries=[],
            number_of_ran_queries=None,
            query_list=None,
            current_query=None,
            query_id=None,
        )

        logger.debug(f"Prepared graph input: {graph_input}")
        logger.info("Starting research process")
        logger.debug("Invoking research graph")

        # Configure graph with request parameters
        config = Configuration(
            number_of_initial_queries=request.initial_search_query_count,
            max_research_loops=request.max_research_loops,
            ollama_llm=request.ollama_llm,  # Set the global model override
        )

        # Run graph with updated config
        return StreamingResponse(
            event_generator(graph_input, config),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Error during request processing: {str(e)}", exc_info=True)
        error_event = {"error": {"status": f"An error occurred: {str(e)}"}}
        return StreamingResponse(
            iter([f"data: {serialize_event(error_event)}\n\n"]),
            media_type="text/event-stream",
        )


async def event_generator(graph_input: ResearchState, config: Configuration):
    """Generate events for the streaming response."""
    try:
        async for event in graph.astream(
            graph_input, {"configurable": config.model_dump()}
        ):
            logger.debug(f"Graph event: {event}")
            yield f"data: {serialize_event(event)}\n\n"

        # Indicate research completion
        research_complete_event = {
            "research": {"status": "Research completed, generating final response..."}
        }
        yield f"data: {serialize_event(research_complete_event)}\n\n"

    except Exception as e:
        logger.error(f"Error during research process: {str(e)}", exc_info=True)
        error_event = {
            "error": {"status": f"An error occurred during research: {str(e)}"}
        }
        yield f"data: {serialize_event(error_event)}\n\n"


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir
    static_files_path = build_path / "assets"  # Vite uses 'assets' subdir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    build_dir = pathlib.Path(build_dir)

    react = FastAPI(openapi_url="")
    react.mount(
        "/assets", StaticFiles(directory=static_files_path), name="static_assets"
    )

    @react.get("/{path:path}")
    async def handle_catch_all(request: Request, path: str):
        fp = build_path / path
        if not fp.exists() or not fp.is_file():
            fp = build_path / "index.html"
        return fastapi.responses.FileResponse(fp)

    return react


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
