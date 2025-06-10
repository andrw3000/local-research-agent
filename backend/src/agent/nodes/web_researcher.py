import os
import re
import json
import time
import asyncio
import tempfile
import nest_asyncio
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from agent.state import ResearchState
from agent.configuration import Configuration
from agent.prompts import get_current_date, web_searcher_instructions
from agent.utils import insert_citation_markers
from agent.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="DEBUG", name="agent.web_researcher")

# Enable nested event loops
nest_asyncio.apply()


def ensure_event_loop():
    """Ensure there is an event loop available in the current thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def web_searcher(
    research_topic: str,
    model_name: str = "gemma3",
    temperature: float = 0.0,
    max_results: int = 5,
    max_context_length: int = 3000,
):
    context = ""

    try:
        # Perform DuckDuckGo search with retries
        tool = DuckDuckGoSearchResults()
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                results = tool.run(research_topic)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"Failed to get search results after {max_retries} attempts: {e}"
                    )
                    results = []
                    break
                print(
                    f"Search attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {e}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2

        # Process search results based on their type
        urls = []
        if isinstance(results, str):
            try:
                # Try parsing as JSON
                parsed_results = json.loads(results)
                if isinstance(parsed_results, list):
                    urls = [
                        r.get("href") or r.get("link") or ""
                        for r in parsed_results[:max_results]
                    ]
            except json.JSONDecodeError:
                # If not JSON, look for URLs in the text
                url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
                found_urls = re.findall(url_pattern, results)
                urls = found_urls[:max_results]
        else:
            # Handle list of dictionaries
            urls = [
                r.get("href") or r.get("link") or ""
                for r in (results or [])[:max_results]
            ]

        # Filter out empty URLs
        urls = [url for url in urls if url]

        # Crawl with crawl4ai
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_paths = []

            async def crawl_urls(urls, tmpdir):
                run_config = CrawlerRunConfig()
                run_config.crawl_scope = "single"  # Only crawl the main page
                run_config.output_dir = tmpdir
                async with AsyncWebCrawler() as crawler:
                    results = await crawler.arun_many(urls=urls, config=run_config)
                for res in results:
                    if res.success and hasattr(res, "markdown"):
                        fname = Path(tmpdir) / (
                            res.url.replace("https://", "").replace("/", "_") + ".md"
                        )
                        fname.write_text(res.markdown)
                    else:
                        print(
                            f"Failed: {res.url} → {str(res.error) if hasattr(res, 'error') else 'Unknown error'}"
                        )

            # Run crawling in current event loop or create new one
            loop = ensure_event_loop()
            try:
                loop.run_until_complete(crawl_urls(urls, tmpdir))
            except Exception as e:
                print(f"Error during crawling: {e}")

            # Get all markdown files from temp directory
            markdown_paths = list(Path(tmpdir).glob("*.md"))

            # Read and concatenate content
            for path in markdown_paths:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()[:max_context_length]
                        if content:
                            context += content + "\n\n"
                except Exception as e:
                    print(f"Failed to read {path}: {e}")

            current_date = datetime.now().strftime("%Y-%m-%d")
            final_prompt = web_searcher_instructions.format(
                research_topic=research_topic, current_date=current_date
            )
            final_prompt += (
                f"\n\nWeb Results:\n{context if context else '[No content found]'}\n\n"
            )

            llm = ChatOllama(model=model_name, temperature=temperature)

            messages = [
                SystemMessage(
                    content="You are a fact-checking AI that only uses sourced data."
                ),
                HumanMessage(content=final_prompt),
            ]

            # Get response from the model
            try:
                response = llm.invoke(messages)
                return response.content if response else "No response generated"
            except Exception as e:
                print(f"Error from LLM: {e}")
                return f"Error while processing request: {e}"

    except Exception as e:
        print(f"Error in web_searcher: {e}")
        return f"Error while processing web search: {e}"


def web_research(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """LangGraph node that performs web research for a given query."""
    if not state.get("current_query"):
        if state.get("query_list"):
            # Handle initial queries from query generator
            state["current_query"] = state["query_list"][0]
            state["query_id"] = f"query_0"
            state["query_list"] = state["query_list"][1:]  # Remove used query
        elif state.get("follow_up_queries"):
            # Handle follow-up queries from reflection
            state["current_query"] = state["follow_up_queries"][0]
            current_loop = len(state.get("research_loop_count", []))
            state["query_id"] = f"followup_{current_loop}"
            state["follow_up_queries"] = state["follow_up_queries"][1:]  # Remove used query

    logger.info(f"Starting web research for query: {state['current_query']}")

    # Get configuration and model
    cfg = Configuration.from_runnable_config(config)
    web_search_model = cfg.ollama_llm or cfg.web_search_model
    logger.debug(f"Using web search model: {web_search_model}")

    # Ensure we have a query to research
    if not state["current_query"]:
        raise ValueError("No query provided for web research")

    # Perform web research
    modified_text, citations = web_searcher(
        research_topic=state["current_query"],
        model_name=web_search_model,
    )

    # Extract sources and prepare return state
    sources_gathered = [citation["segments"][0] for citation in citations]
    logger.info(f"Completed web research with {len(sources_gathered)} sources gathered")

    # Merge current research with existing state
    return {
        # Research results
        "sources_gathered": (state.get("sources_gathered", []) + sources_gathered),
        "search_query": (state.get("search_query", []) + [state["current_query"]]),
        "web_research_result": (state.get("web_research_result", []) + [modified_text]),
        "research_loop_count": state.get("research_loop_count", []),
        # Preserve other state
        "messages": state.get("messages", []),
        "is_sufficient": state.get("is_sufficient"),
        "knowledge_gap": state.get("knowledge_gap"),
        "follow_up_queries": state.get("follow_up_queries", []),
        "number_of_ran_queries": len(state.get("search_query", [])) + 1,
        "query_list": state.get("query_list"),
        "current_query": None,  # Clear current query as it's been processed
        "query_id": None,  # Clear query ID as it's been processed
    }
