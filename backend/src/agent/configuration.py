import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    # Global LLM override
    ollama_llm: Optional[str] = Field(
        default=None,
        metadata={
            "description": "If set, this model will be used for all LLM operations, overriding individual model settings."
        },
    )

    # LLM with output structure and function calling
    query_generator_model: str = Field(
        default="mistral-small3.1",  # gemini-2.0-flash
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    web_search_model: str = Field(
        default="gemma3",  # gemini-2.0-flash
        metadata={
            "description": "The name of the language model to use for the agent's web search."
        },
    )

    # Reasoning model with output structure
    reasoning_model: str = Field(
        default="qwq",  # gemini-2.5-flash-preview-04-17
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    # Summarization model
    answer_model: str = Field(
        default="gemma3",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
