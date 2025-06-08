import pytest
from fastapi.testclient import TestClient
from agent.app import app
import json


@pytest.mark.anyio
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
        response = client.post("/app/agent", json=test_case)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        # Convert response content to a list of events
        content = response.content.decode()
        events = [
            line for line in content.split("\n\n") if line and line.startswith("data: ")
        ]

        # Should have at least 2 events:
        # 1. Initial "generating queries" event
        # 2. Final response with messages
        assert len(events) >= 2

        # First event should be query generation status
        first_event = json.loads(events[0].replace("data: ", ""))
        assert "generate_query" in first_event
        assert "status" in first_event["generate_query"]

        # Process events and check for expected flow
        research_done = False
        error_occurred = False
        for event in events[1:-1]:  # Skip first and last events
            event_data = json.loads(event.replace("data: ", ""))
            if "research" in event_data:
                research_done = True
            if "error" in event_data:
                error_occurred = True
                break

        # If no error occurred, check the final response
        if not error_occurred:
            last_event = json.loads(events[-1].replace("data: ", ""))
            assert "data" in last_event
            assert "messages" in last_event["data"]
            assert isinstance(last_event["data"]["messages"], list)
            assert len(last_event["data"]["messages"]) == len(test_case["messages"]) + 1
            assert last_event["data"]["messages"][-1]["role"] == "ai"
            assert "content" in last_event["data"]["messages"][-1]
