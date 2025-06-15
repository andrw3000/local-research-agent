import os
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from agent.state import ResearchState
from agent.configuration import Configuration
from agent.prompts import get_current_date, answer_instructions
from agent.utils import get_research_topic
from agent.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="DEBUG", name="agent.answer")


def finalize_answer(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including messages with the final answer and sources.
    """
    logger.info("Starting answer finalization")

    # Get configuration and model
    cfg = Configuration.from_runnable_config(config)
    logger.debug(f"Ollama model configuration: {cfg}")
    answer_model = cfg.ollama_llm or cfg.answer_model
    logger.debug(f"Using answer model: {answer_model}")

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    logger.debug(f"Research topic: {research_topic}")
    logger.debug(f"Input state: {state}")

    # Verify web research results are present
    if not state.get("web_research_result"):
        logger.error("No web research results found in state!")
        return {
            "messages": state["messages"] + [
                AIMessage(
                    content="Error: No research results available to generate answer"
                )
            ],
            "sources_gathered": [],
            "research_loop_count": [],
        }

    # Format the final prompt
    formatted_prompt = answer_instructions.format(
        research_topic=research_topic,
        current_date=current_date,
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    logger.debug(f"Formatted prompt with {len(state['web_research_result'])} summaries")

    # Initialize the LLM
    llm = ChatOllama(
        model=answer_model,
        temperature=0.7,
        base_url=os.getenv("OLLAMA_URL"),
    )

    logger.debug("Invoking LLM for final answer generation")
    result = llm.invoke(formatted_prompt)
    logger.info("Final answer generated")

    # Deduplicate sources and keep only those that contributed to the final answer
    logger.debug("Processing source citations")
    unique_sources = list(set(state["sources_gathered"]))
    logger.info(f"Answer finalized with {len(unique_sources)} unique sources")

    return {
        "messages": state["messages"] + [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
        "research_loop_count": [],  # Empty list since answer node doesn't increment the counter
        # Pass through required fields with empty/None values since this is the final state
        "search_query": [],
        "web_research_result": [],
        "is_sufficient": None,
        "knowledge_gap": None,
        "follow_up_queries": [],
        "number_of_ran_queries": None,
        "query_list": None,
        "current_query": None,
        "query_id": None,
    }
