import os
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from agent.configuration import Configuration
from agent.nodes import (
    generate_query,
    continue_to_web_research,
    web_research,
    reflection,
    evaluate_research,
    finalize_answer,
)
from agent.state import ResearchState


# In concurrent LangGraph, we need to define the types of update operations that can happen
class UpdateType(TypedDict):
    is_sufficient: bool


load_dotenv()

if os.getenv("OLLAMA_URL") is None:
    raise ValueError("The `OLLAMA_URL` environment variable is not set")


# Create our Agent Graph with the proper state typing
builder = StateGraph(ResearchState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add edges for parallel branches and conditional flow based on research state
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="local-research-agent")
