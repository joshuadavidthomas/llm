"""Tests for the /v1/responses code path in the default OpenAI plugin."""

import json
import os

import llm
import pytest

API_KEY = os.environ.get("PYTEST_OPENAI_API_KEY", None) or "badkey"


def test_responses_model_is_registered():
    model = llm.get_model("gpt-5.5")
    assert "Responses" in type(model).__name__
    # The chat_completions opt-out option must be exposed.
    assert "chat_completions" in model.Options.model_fields


def test_chat_completions_opt_out_dispatches_to_chat(httpx_mock):
    """When chat_completions=1 is passed, the request must hit
    /v1/chat/completions, not /v1/responses."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "model": "gpt-5.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi from chat"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
        headers={"Content-Type": "application/json"},
    )
    model = llm.get_model("gpt-5.5")
    response = model.prompt("hello", stream=False, chat_completions=True, key="test")
    assert response.text() == "hi from chat"


def test_default_routes_to_responses_endpoint(httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/responses",
        json={
            "id": "resp_test_1",
            "object": "response",
            "created_at": 1,
            "model": "gpt-5.5",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "hi from responses", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
            },
            "status": "completed",
        },
        headers={"Content-Type": "application/json"},
    )
    model = llm.get_model("gpt-5.5")
    response = model.prompt("hello", stream=False, key="test")
    assert response.text() == "hi from responses"
    # Ensure we sent to the right endpoint
    requests = [r for r in httpx_mock.get_requests()]
    assert any("/v1/responses" in str(r.url) for r in requests)


def test_responses_input_translation():
    """Unit-test the message-to-input translator without hitting the API."""
    from llm.parts import (
        Message,
        TextPart,
        ToolCallPart,
        ToolResultPart,
        AttachmentPart,
    )

    model = llm.get_model("gpt-5.5")

    class FakePrompt:
        messages = [
            Message(role="system", parts=[TextPart(text="be brief")]),
            Message(role="user", parts=[TextPart(text="2 + 2?")]),
            Message(
                role="assistant",
                parts=[
                    ToolCallPart(
                        name="add",
                        arguments={"a": 2, "b": 2},
                        tool_call_id="call_abc",
                    )
                ],
            ),
            Message(
                role="tool",
                parts=[
                    ToolResultPart(
                        name="add", output="4", tool_call_id="call_abc"
                    )
                ],
            ),
        ]

    items, instructions = model._build_responses_input(FakePrompt())
    assert instructions == "be brief"
    # First user message is a plain string content
    assert items[0] == {"role": "user", "content": "2 + 2?"}
    # function_call from assistant
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_abc"
    assert items[1]["name"] == "add"
    assert json.loads(items[1]["arguments"]) == {"a": 2, "b": 2}
    # tool result
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "4",
    }


def test_responses_kwargs_packs_reasoning_and_verbosity():
    model = llm.get_model("gpt-5.5")
    options = model.Options(reasoning_effort="low", verbosity="low")

    class FakePrompt:
        pass

    p = FakePrompt()
    p.options = options
    p.tools = []
    p.schema = None
    kwargs = model._build_responses_kwargs(p, stream=False)
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["text"]["verbosity"] == "low"


@pytest.mark.vcr
def test_responses_basic_non_streaming(vcr):
    model = llm.get_model("gpt-5.5")
    response = model.prompt(
        "Reply with exactly: pong",
        stream=False,
        reasoning_effort="low",
        key=API_KEY,
    )
    text = response.text()
    assert "pong" in text.lower()
    # response_json should reflect the Responses API shape
    assert response.response_json["object"] == "response"


@pytest.mark.vcr
def test_responses_basic_streaming(vcr):
    model = llm.get_model("gpt-5.5")
    response = model.prompt(
        "Reply with exactly: pong",
        reasoning_effort="low",
        key=API_KEY,
    )
    chunks = list(response)
    text = "".join(chunks)
    assert "pong" in text.lower()


@pytest.mark.vcr
def test_responses_tool_use(vcr):
    model = llm.get_model("gpt-5.5")

    def multiply(a: int, b: int) -> int:
        "Multiply two numbers."
        return a * b

    chain = model.chain(
        "What is 1231 * 2331? Use the multiply tool.",
        tools=[multiply],
        stream=False,
        options={"reasoning_effort": "low"},
        key=API_KEY,
    )
    output = chain.text()
    assert "2869461" in output.replace(",", "")
    first, second = chain._responses
    assert first.tool_calls()[0].name == "multiply"
    assert first.tool_calls()[0].arguments == {"a": 1231, "b": 2331}
    assert second.prompt.tool_results[0].output == "2869461"


@pytest.mark.vcr
def test_responses_tool_use_streaming(vcr):
    model = llm.get_model("gpt-5.5")

    def multiply(a: int, b: int) -> int:
        "Multiply two numbers."
        return a * b

    chain = model.chain(
        "What is 1231 * 2331? Use the multiply tool.",
        tools=[multiply],
        options={"reasoning_effort": "low"},
        key=API_KEY,
    )
    output = "".join(chain)
    assert "2869461" in output.replace(",", "")
    first, second = chain._responses
    assert first.tool_calls()[0].arguments == {"a": 1231, "b": 2331}


@pytest.mark.vcr
def test_responses_round_trips_encrypted_reasoning(vcr):
    """Reasoning items returned by the API in the first turn must be
    echoed back verbatim on the second turn so the model can pick up
    its hidden chain of thought after the tool result arrives."""
    from llm.parts import ReasoningPart

    model = llm.get_model("gpt-5.5")

    def lookup_population(country: str) -> int:
        "Returns the current population of the specified fictional country."
        return 123124

    def can_have_dragons(population: int) -> bool:
        "Returns True if the specified population can have dragons."
        return population > 10000

    chain = model.chain(
        "Pick a clever country name, look up its population, then check "
        "whether it can have dragons. Be brief.",
        tools=[lookup_population, can_have_dragons],
        stream=False,
        options={"reasoning_effort": "high"},
        key=API_KEY,
    )
    chain.text()  # drain the chain

    first = chain._responses[0]

    # The first response must produce at least one ReasoningPart carrying
    # the opaque encrypted_content + id.
    reasoning_parts = [
        p for m in first.messages() for p in m.parts if isinstance(p, ReasoningPart)
    ]
    assert reasoning_parts, "first turn should expose at least one ReasoningPart"
    pm = reasoning_parts[0].provider_metadata or {}
    assert "openai" in pm
    assert pm["openai"].get("encrypted_content"), "encrypted_content must be captured"
    assert pm["openai"].get("id"), "reasoning id must be captured"

    # The second turn's outgoing input must echo back that reasoning
    # item, otherwise the model loses its chain of thought.
    second = chain._responses[1]
    second_input = (second._prompt_json or {}).get("input") or []
    reasoning_inputs = [it for it in second_input if it.get("type") == "reasoning"]
    assert reasoning_inputs, "second turn must echo a reasoning input item"
    assert (
        reasoning_inputs[0]["encrypted_content"]
        == pm["openai"]["encrypted_content"]
    )
    assert reasoning_inputs[0]["id"] == pm["openai"]["id"]
