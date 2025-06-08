# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import fastapi.exceptions
import asyncio
import json
from agent.graph import graph  # Add this import for the LangGraph agent


# Define request/response models
class ChatMessage(BaseModel):
    role: str = "user"  # or "ai"
    content: str
    id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    initial_search_query_count: int = 3
    max_research_loops: int = 3
    ollama_llm: str = "deepseek-r1"  # Default model


# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Research agent endpoint using LangGraph
@app.post("/app/agent")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        # Convert ChatRequest to graph input format
        graph_input = {
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "max_research_loops": request.max_research_loops,
            "initial_search_query_count": request.initial_search_query_count,
            "ollama_llm": request.ollama_llm,
        }

        # Start research process
        query_event = {
            "generate_query": {"status": "Started generating research queries..."}
        }
        yield f"data: {json.dumps(query_event)}\n\n"

        try:
            # Run the graph
            state = graph.invoke(graph_input)

            # Indicate research completion
            research_complete_event = {
                "research": {
                    "status": "Research completed, generating final response..."
                }
            }
            yield f"data: {json.dumps(research_complete_event)}\n\n"

            # Get the final response
            last_message = state["messages"][-1]
            # Handle LangChain message objects
            content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

            # Convert messages to list format
            msg_dicts = [
                {"role": msg.role, "content": msg.content, "id": msg.id}
                for msg in request.messages
            ]
            response_message = {
                "role": "ai",
                "id": f"msg_{int(time.time())}",
                "content": content,
            }
            yield f"data: {json.dumps({'data': {'messages': [*msg_dicts, response_message]}})}\n\n"
        except Exception as e:
            error_event = {"error": {"status": f"An error occurred: {str(e)}"}}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


from fastapi.responses import StreamingResponse


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
