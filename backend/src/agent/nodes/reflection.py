import os
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.types import Send
from agent.state import ResearchState
from agent.configuration import Configuration
from agent.prompts import get_current_date, reflection_instructions
from agent.tools_and_schemas import Reflection
from agent.utils import get_research_topic
from agent.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="DEBUG", name="agent.reflection")


def reflection(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries."""
    logger.info("Starting reflection phase")

    # Get configuration and model
    cfg = Configuration.from_runnable_config(config)
    reasoning_model = config.get("ollama_llm") or cfg.reasoning_model
    logger.debug(f"Using reasoning model: {reasoning_model}")

    # Validate and log state
    if "web_research_result" not in state:
        logger.error("No web research results in input state!")
        raise ValueError("Missing web research results in state")
    logger.debug(
        f"Received web research results: {len(state['web_research_result'])} items"
    )

    # Format prompt for reflection
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    logger.debug(f"Research topic: {research_topic}")

    formatted_prompt = reflection_instructions.format(
        research_topic=research_topic,
        current_date=current_date,
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    # Initialize reasoning model
    llm = ChatOllama(
        model=reasoning_model,
        temperature=1.0,
        base_url=os.getenv("OLLAMA_URL"),
    )

    logger.debug("Invoking LLM for reflection")
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    logger.info(f"Reflection completed - Sufficient: {result.is_sufficient}")

    if not result.is_sufficient:
        logger.info(f"Knowledge gap identified: {result.knowledge_gap}")
        logger.debug(f"Follow-up queries: {result.follow_up_queries}")

    # For operator.add, we return a list with 1 if we need more research, empty list otherwise
    research_loop_increment = [1] if not result.is_sufficient else []

    return ResearchState(
        # Core reflection results
        is_sufficient=result.is_sufficient,
        knowledge_gap=result.knowledge_gap,
        follow_up_queries=result.follow_up_queries,
        research_loop_count=research_loop_increment,
        number_of_ran_queries=len(state.get("search_query", [])),
        # Pass through existing state
        messages=state.get("messages", []),
        web_research_result=state.get("web_research_result", []),
        sources_gathered=state.get("sources_gathered", []),
        search_query=state.get("search_query", []),
        # Initialize optional fields
        query_list=None,
        current_query=None,
        query_id=None,
    )


def evaluate_research(
    state: ResearchState,
    config: RunnableConfig,
) -> ResearchState | Send:
    """Evaluate whether research should continue or finalize."""
    # Get config graph state for state preservation
    current_loop_count = sum(state.get("research_loop_count", []))
    logger.debug(f"Current research loop count: {current_loop_count}")

    # Check conditions for continuing research
    if (
        not state["is_sufficient"]
        and state["follow_up_queries"]
        and current_loop_count
        < config.get("configurable", {}).get("max_research_loops", 3)
    ):
        logger.info("Continuing research with follow-up queries")
        logger.debug(f"Follow-up queries: {state['follow_up_queries']}")
        next_state = ResearchState(
            messages=state.get("messages", []),
            search_query=state.get("search_query", []),
            web_research_result=state.get("web_research_result", []),
            sources_gathered=state.get("sources_gathered", []),
            research_loop_count=[],  # Reset for next iteration
            current_query=state["follow_up_queries"][0],
            query_id=f"followup_{current_loop_count}",
            # Initialize optional fields
            is_sufficient=None,
            knowledge_gap=None,
            follow_up_queries=[],
            number_of_ran_queries=None,
            query_list=None,
        )
        return Send("web_research", next_state)

    logger.info("Research complete, moving to answer finalization")

    # Construct finalize state with all preserved information
    finalize_state = ResearchState(
        # Pass through research results
        web_research_result=state.get("web_research_result", []),
        sources_gathered=state.get("sources_gathered", []),
        search_query=state.get("search_query", []),
        messages=state.get("messages", []),
        research_loop_count=[],  # Empty list since we're finalizing
        number_of_ran_queries=state["number_of_ran_queries"],
        # Initialize optional fields
        is_sufficient=None,
        knowledge_gap=None,
        follow_up_queries=[],
        query_list=None,
        current_query=None,
        query_id=None,
    )

    logger.debug(f"Sending state to finalize_answer: {finalize_state}")
    return Send("finalize_answer", finalize_state)
