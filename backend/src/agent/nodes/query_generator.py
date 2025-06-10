import os
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.types import Send
from agent.state import ResearchState
from agent.configuration import Configuration
from agent.prompts import get_current_date, query_writer_instructions
from agent.tools_and_schemas import SearchQueryList
from agent.utils import get_research_topic
from agent.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="DEBUG", name="agent.query_generator")


def generate_query(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """LangGraph node that generates search queries based on the User's question.

    Uses LLM to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current research state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Updated research state with generated queries
    """
    logger.info("Generating initial search queries")

    configurable = Configuration.from_runnable_config(config)
    query_model = config.get("ollama_llm") or configurable.query_generator_model

    # Get initial search query count from config
    number_of_queries = config.get("configurable", {}).get(
        "number_of_initial_queries", 3
    )

    # init LLM
    llm = ChatOllama(
        model=query_model,
        temperature=1.0,
        base_url=os.getenv("OLLAMA_URL"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    logger.debug(f"Using model: {query_model}")

    # Format the prompt
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    logger.debug(f"Research topic: {research_topic}")

    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=number_of_queries,
    )
    # Generate the search queries
    logger.debug("Invoking LLM for query generation")
    result = structured_llm.invoke(formatted_prompt)
    logger.info(f"Generated {len(result.query)} search queries")
    logger.debug(f"Generated queries: {result.query}")

    return {"query_list": result.query}


def continue_to_web_research(state: ResearchState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    Returns a list of Send objects, one for each query to be processed in parallel.
    """
    logger.debug(
        f"Processing {len(state['query_list'] or [])} queries for web research"
    )
    result = []

    if not state.get("query_list"):
        logger.warning("No queries generated to research")
        return result

    for i, query in enumerate(state["query_list"]):
        query_id = f"query_{i}"
        # Send each query directly to web_research with minimal required state
        logger.debug(f"Added query to web research: {query} with id {query_id}")
        research_state = ResearchState(
            messages=state.get("messages", []),
            search_query=state.get("search_query", []),
            web_research_result=state.get("web_research_result", []),
            sources_gathered=state.get("sources_gathered", []),
            research_loop_count=state.get("research_loop_count", []),
            current_query=query,  # Changed: query is now used directly as it's a string
            query_id=query_id,
            # Initialize optional fields
            is_sufficient=None,
            knowledge_gap=None,
            follow_up_queries=[],
            number_of_ran_queries=None,
            query_list=None,
        )
        result.append(Send("web_research", research_state))

    return result
