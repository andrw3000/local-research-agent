import pytest
from fastapi.testclient import TestClient
from agent.app import app
import json


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_stream():
    """Test the chat stream endpoint with various scenarios"""
    client = TestClient(app)

    test_cases = [
        {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "initial_search_query_count": 2,
            "max_research_loops": 2,
            "ollama_llm": "deepseek-r1",
        },
        # Test with multiple messages
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "ai", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "What's the weather like in Paris?"},
            ],
            "initial_search_query_count": 1,
            "max_research_loops": 1,
            "ollama_llm": "deepseek-r1",
        },
        # Test with default parameters
        {
            "messages": [{"role": "user", "content": "Tell me about TypeScript"}],
        },
    ]

    for test_case in test_cases:
        response = client.post(
            "/app/agent",
            json=test_case,
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Read the response content as SSE stream
        events = []
        for line in response.iter_lines():
            if line:
                # Handle both str and bytes type for response lines
                line_str = line.decode() if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    try:
                        event_data = json.loads(line_str.replace("data: ", ""))
                        events.append(event_data)
                    except json.JSONDecodeError:
                        continue

        # Verify we got events
        assert len(events) > 0

        # Check event structure
        has_error = any("error" in event for event in events)
        if not has_error:
            # Should see research events
            assert any(
                "research" in event or "web_research_result" in event
                for event in events
            )
        else:
            # For error case, verify error structure
            error_events = [event for event in events if "error" in event]
            assert len(error_events) > 0
            assert "status" in error_events[0]["error"]


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
        headers={"Accept": "text/event-stream"},
    )

    # Verify the response starts correctly
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # Read a few events then simulate disconnection
    for line in response.iter_lines():
        if line:
            break  # Exit after first event
