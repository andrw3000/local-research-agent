from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Any

from langgraph.graph import add_messages
from typing_extensions import Annotated

import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


class Query(TypedDict):
    query: str
    rationale: str


class ResearchState(TypedDict):
    # Core research data
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    research_loop_count: Annotated[List[int], operator.add]

    # Reflection data (optional)
    is_sufficient: bool | None
    knowledge_gap: str | None
    follow_up_queries: Annotated[list, operator.add]
    number_of_ran_queries: int | None

    # Query data (optional)
    query_list: list[Query] | None

    # Web search data (optional)
    current_query: str | None
    query_id: str | None


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
