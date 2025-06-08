import asyncio
import tempfile
import nest_asyncio
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from crawl4ai import AsyncWebCrawler
from pathlib import Path
from datetime import datetime
import time
from agent.prompts import web_searcher_instructions

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
            import json

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
                import re

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

            # Create crawler instance
            crawler = AsyncWebCrawler()

            async def crawl_urls():
                tasks = []
                for url in urls:
                    task = crawler.crawl(
                        url=url,
                        output_file=str(
                            Path(tmpdir)
                            / (url.replace("https://", "").replace("/", "_") + ".md")
                        ),
                        format="markdown",
                    )
                    tasks.append(task)
                if tasks:  # Only gather if there are tasks
                    await asyncio.gather(*tasks)

            # Run crawling in current event loop or create new one
            loop = ensure_event_loop()
            try:
                loop.run_until_complete(crawl_urls())
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
