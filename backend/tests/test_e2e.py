import pytest
import asyncio
import json
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from agent.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_agent_endpoint_structure(client):
    """Test that the agent endpoint accepts requests and has correct structure"""
    response = client.post(
        "/app/agent",
        json={
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_research_loops": 1,
            "initial_search_query_count": 1,
            "ollama_llm": "test",
        },
        headers={"Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Collect all events
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            event_data = json.loads(line.replace("data: ", ""))
            events.append(event_data)

    # Verify event structure
    assert len(events) >= 1  # Should have at least one event

    # Check that we got either a successful flow or an error message
    has_error = any("error" in event for event in events)
    if not has_error:
        assert (
            len(events) >= 3
        )  # Should have query, research, and final response events
        assert any("generate_query" in event for event in events)
        assert any("research" in event for event in events)
        final_events = [
            event for event in events if "data" in event and "messages" in event["data"]
        ]
        assert len(final_events) > 0
        final_messages = final_events[-1]["data"]["messages"]
        assert len(final_messages) >= 2  # Should have question and answer
        assert isinstance(final_messages[-1]["content"], str)
        assert len(final_messages[-1]["content"]) > 0
    else:
        # For error case, verify error structure
        error_events = [event for event in events if "error" in event]
        assert len(error_events) > 0
        assert "status" in error_events[0]["error"]
        assert isinstance(error_events[0]["error"]["status"], str)
        assert len(error_events[0]["error"]["status"]) > 0


def test_agent_validates_input(client):
    """Test that the agent endpoint properly validates input"""
    # Test missing required fields
    response = client.post(
        "/app/agent",
        json={},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 422

    # Test invalid message format
    response = client.post(
        "/app/agent",
        json={
            "messages": [{"invalid": "format"}],
            "max_research_loops": 1,
            "initial_search_query_count": 1,
            "ollama_llm": "test",
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 422

    # Test invalid research loops value
    response = client.post(
        "/app/agent",
        json={
            "messages": [{"role": "user", "content": "test"}],
            "max_research_loops": -1,
            "initial_search_query_count": 1,
            "ollama_llm": "test",
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 422


def test_agent_cancellation_handling(client):
    """Test that the agent endpoint can handle client disconnection"""
    response = client.post(
        "/app/agent",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What is a very complex topic that requires lots of research?",
                }
            ],
            "max_research_loops": 5,
            "initial_search_query_count": 3,
            "ollama_llm": "test",
        },
        headers={"Accept": "text/event-stream", "Connection": "close"},
    )

    # Verify the response starts correctly
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Get first event to verify stream started
    first_line = next(response.iter_lines())
    assert first_line.startswith("data:")
