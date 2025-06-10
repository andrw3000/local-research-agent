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
    """Perform web research based on a search query."""
    logger.info(f"Starting web research for query: {state['current_query']}")

    # Get configuration and model
    cfg = Configuration.from_runnable_config(config)
    web_search_model = config.get("ollama_llm") or cfg.web_search_model
    logger.debug(f"Using web search model: {web_search_model}")

    # Perform web search
    try:
        logger.debug("Initiating web search")
        response = web_searcher(
            research_topic=state["current_query"],
            model_name=web_search_model,
            temperature=0.0,
            max_results=5,
            max_context_length=5000,
        )
        logger.debug("Web search completed successfully")
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}", exc_info=True)
        response = f"Could not retrieve current information about '{state['search_query']}' due to search API limitations."

    # Extract text from response if it's an AIMessage
    if isinstance(response, AIMessage):
        response_text = response.content
    else:
        response_text = str(response)

    citations = []
    logger.debug("Processing search results for citations")

    try:
        # Get DuckDuckGo search results for citations
        tool = DuckDuckGoSearchResults()
        results = []
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1} to fetch search results")
                results = tool.run(state["current_query"])
                logger.info("Successfully retrieved search results")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to get search results after {max_retries} attempts: {str(e)}",
                        exc_info=True,
                    )
                    break
                logger.warning(
                    f"Search attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {str(e)}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2

        # Process search results
        logger.debug("Processing search results")
        if isinstance(results, str):
            try:
                search_results = json.loads(results)
            except json.JSONDecodeError:
                lines = results.split("\n")
                search_results = []
                for line in lines:
                    if "http" in line:
                        search_results.append(
                            {
                                "title": line.split(" http")[0].strip(),
                                "href": "http" + line.split(" http")[1].strip(),
                            }
                        )
        else:
            search_results = results

        # Create citations from search results
        for idx, result in enumerate(search_results[:5]):
            try:
                if isinstance(result, dict):
                    title = (
                        result.get("title", "").split(" - ")[0]
                        if " - " in result.get("title", "")
                        else result.get("title", "No Title")
                    )
                    url = result.get("href", result.get("link", "No URL"))
                elif isinstance(result, str):
                    title = (
                        result.split("http")[0].strip() if "http" in result else result
                    )
                    url = (
                        "http" + result.split("http")[1].strip()
                        if "http" in result
                        else "No URL"
                    )
                else:
                    logger.warning(f"Skipping result with unexpected format: {result}")
                    continue

                citation = {
                    "start_index": len(response_text),
                    "end_index": len(response_text),
                    "segments": [
                        {"label": title, "short_url": f"[{idx + 1}]", "value": url}
                    ],
                }
                citations.append(citation)
                logger.debug(f"Added citation: {citation}")
            except Exception as e:
                logger.error(
                    f"Error processing search result {idx}: {str(e)}", exc_info=True
                )
                continue

    except Exception as e:
        logger.error(f"Error processing search results: {str(e)}", exc_info=True)
        citations = [
            {
                "start_index": len(response_text),
                "end_index": len(response_text),
                "segments": [
                    {
                        "label": "Search Error",
                        "short_url": "[!]",
                        "value": "Search results unavailable",
                    }
                ],
            }
        ]

    # Add citation markers
    logger.debug("Adding citation markers to response text")
    modified_text = insert_citation_markers(response_text, citations)

    # Extract sources gathered
    sources_gathered = [citation["segments"][0] for citation in citations]
    logger.info(f"Completed web research with {len(sources_gathered)} sources gathered")

    # Get config graph state for state preservation
    graph_state = config.get("graph_state", {}) if config else {}

    # Initialize lists if they don't exist
    current_sources = graph_state.get("sources_gathered", [])
    current_queries = graph_state.get("search_query", [])
    current_results = graph_state.get("web_research_result", [])

    if not isinstance(current_sources, list):
        current_sources = []
    if not isinstance(current_queries, list):
        current_queries = []
    if not isinstance(current_results, list):
        current_results = []

    # Merge current state with graph state
    return ResearchState(
        # Research results
        sources_gathered=current_sources + sources_gathered,
        search_query=current_queries + [state["current_query"]],
        web_research_result=current_results + [modified_text],
        research_loop_count=[],  # Empty list since web_research doesn't increment the counter
        # Preserve state
        messages=state.get("messages", []),
        # Initialize optional fields
        is_sufficient=None,
        knowledge_gap=None,
        follow_up_queries=[],
        number_of_ran_queries=None,
        query_list=None,
        current_query=None,
        query_id=None,
    )
