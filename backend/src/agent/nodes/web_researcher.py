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
from agent.state import OverallState, WebSearchState
from agent.configuration import Configuration
from agent.prompts import get_current_date, web_searcher_instructions
from agent.utils import insert_citation_markers

# Enable nested event loops - needed for Jupyter/IPython environments
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
                run_config = CrawlerRunConfig(format="markdown")
                async with AsyncWebCrawler() as crawler:
                    results = await crawler.arun_many(urls=urls, config=run_config)
                for res in results:
                    if res.success:
                        fname = Path(tmpdir) / (
                            res.url.replace("https://", "").replace("/", "_") + ".md"
                        )
                        fname.write_text(res.markdown.raw_markdown)
                    else:
                        print(f"Failed: {res.url} → {res.error_message}")

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


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using DuckDuckGo or Google Search API.

    This function:
    1. Performs web search using DuckDuckGo
    2. Processes search results and extracts useful content
    3. Creates citations with proper formatting
    4. Returns the results with citations in a standardized format

    Args:
        state: Current graph state containing the search query
        config: Configuration for the runnable

    Returns:
        Dictionary with state update, including sources_gathered and web_research_result
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    web_search_model = state.get("ollama_llm") or configurable.web_search_model

    # Custom web search using DuckDuckGo and crawl4ai
    try:
        response = web_searcher(
            research_topic=state["search_query"],
            model_name=web_search_model,
            temperature=0.0,
            max_results=5,
            max_context_length=5000,
        )
    except Exception as e:
        print(f"Error during web search: {e}")
        # Provide a default response if web search fails
        response = f"Could not retrieve current information about '{state['search_query']}' due to search API limitations."

    # Extract text from response if it's an AIMessage
    if isinstance(response, AIMessage):
        response_text = response.content
    else:
        response_text = str(response)

    citations = []

    try:
        # Get DuckDuckGo search results for citations with retries
        tool = DuckDuckGoSearchResults()
        results = []
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                results = tool.run(state["search_query"])
                break
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(
                        f"Failed to get search results after {max_retries} attempts: {e}"
                    )
                    # Continue with empty results
                    break
                else:
                    print(
                        f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {e}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        # Process search results - handle both string and list results
        if isinstance(results, str):
            try:
                # Try to parse as JSON if it's a string
                search_results = json.loads(results)
            except json.JSONDecodeError:
                # If not JSON, split into lines and extract URLs
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
        for idx, result in enumerate(search_results[:5]):  # Limit to top 5 results
            try:
                # Try to get title and URL - handle different result formats
                if isinstance(result, dict):
                    title = (
                        result.get("title", "").split(" - ")[0]
                        if " - " in result.get("title", "")
                        else result.get("title", "No Title")
                    )
                    url = result.get("href", result.get("link", "No URL"))
                elif isinstance(result, str):
                    # Handle case where result is a single string (URL or title)
                    title = (
                        result.split("http")[0].strip() if "http" in result else result
                    )
                    url = (
                        "http" + result.split("http")[1].strip()
                        if "http" in result
                        else "No URL"
                    )
                else:
                    continue  # Skip if result format is unknown

                citation = {
                    "start_index": len(response_text),  # Add citations at the end
                    "end_index": len(response_text),
                    "segments": [
                        {"label": title, "short_url": f"[{idx + 1}]", "value": url}
                    ],
                }
                citations.append(citation)
            except Exception as e:
                print(f"Error processing search result {idx}: {e}")
                continue

    except Exception as e:
        print(f"Error processing search results: {e}")
        # Add a placeholder citation if citation processing fails
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

    # Add citation markers using the existing utils function
    modified_text = insert_citation_markers(response_text, citations)

    # Extract sources gathered in same format as before
    sources_gathered = [citation["segments"][0] for citation in citations]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }
