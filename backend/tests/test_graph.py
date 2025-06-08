import pytest
from agent import graph
from agent.state import OverallState


def test_graph_invoke():
    """Test basic graph invocation with a simple question"""
    # Run graph with test input
    state = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Who won the euro 2024"}],
            "max_research_loops": 3,
            "initial_search_query_count": 3,
        }
    )

    # Verify the response structure
    assert isinstance(state, dict)
    assert "messages" in state
    assert len(state["messages"]) > 1  # Should have at least question and answer

    # Verify message structure
    last_message = state["messages"][-1]
    assert isinstance(last_message.content, str)
    assert len(last_message.content) > 0  # Response shouldn't be empty

    # The response should contain factual information about Euro 2024
    response_text = last_message.content.lower()
    assert any(keyword in response_text for keyword in ["euro", "2024"])
