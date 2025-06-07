import asyncio
import tempfile
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from crawl4ai import AsyncWebCrawler
from pathlib import Path
from datetime import datetime
import time
from agent.prompts import web_searcher_instructions


def web_searcher(
    research_topic: str,
    model_name: str = "gemma3",
    temperature: float = 0.0,
    max_results: int = 5,
    max_context_length: int = 3000,
):
    context = ""

    # Perform DuckDuckGo search
    tool = DuckDuckGoSearchResults()
    results = tool.run(research_topic)
    urls = [r["href"] for r in results[:max_results] if "href" in r]

    # Crawl with crawl4ai
    with tempfile.TemporaryDirectory() as tmpdir:
        markdown_paths = []

        # Create crawler instance
        crawler = AsyncWebCrawler()

        # Run crawling asynchronously
        asyncio.run(
            asyncio.gather(
                *[
                    crawler.crawl(
                        url=url,
                        output_file=str(
                            Path(tmpdir)
                            / (url.replace("https://", "").replace("/", "_") + ".md")
                        ),
                        format="markdown",
                    )
                    for url in urls
                ]
            )
        )

        # Get all markdown files from temp directory
        markdown_paths = list(Path(tmpdir).glob("*.md"))

        # Read and concatenate content
        for path in markdown_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    context += f.read()[:max_context_length] + "\n\n"
            except Exception as e:
                print(f"    Failed to read {path}: {e}")

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

    response = llm.invoke(messages)
    response_text = response.content
    return response_text
