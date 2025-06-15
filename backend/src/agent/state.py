from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Any, Optional
from typing_extensions import Annotated, NotRequired

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


class Query(TypedDict):
    query: str
    rationale: str


class ResearchState(TypedDict):
    # Core research data
    messages: List[BaseMessage]
    research_loop_count: List[int]
    search_query: List[str]
    web_research_result: List[str]
    sources_gathered: List[str]

    # Reflection data (optional)
    is_sufficient: Optional[bool]  # No annotation needed, we handle concurrency in the graph
    knowledge_gap: Optional[str]
    follow_up_queries: List[str]
    number_of_ran_queries: NotRequired[Optional[int]]

    # Query data (optional)
    query_list: NotRequired[Optional[List[str]]]

    # Web search data (optional)
    current_query: NotRequired[Optional[str]]
    query_id: NotRequired[Optional[str]]


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
