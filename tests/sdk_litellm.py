import pytest
from litellm import CustomStreamWrapper, acompletion
from litellm.types.utils import Choices, ModelResponse, StreamingChoices, Usage

# Dummy message
MESSAGES = [
    {
        "content": "What is the status of the world?",
        "role": "user",
    },
]


@pytest.mark.asyncio
async def test_litellm_sync():
    # Test API
    res: ModelResponse = await acompletion(  # pyright: ignore[reportAssignmentType]
        api_key="mock",
        base_url="http://localhost:8080/v1",
        messages=MESSAGES,
        model="openai/gpt-4.1-nano",
    )

    # Validate response
    assert len(res.choices) == 1

    # Validate choice
    choice = res.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "stop"
    assert choice.message.content is not None
    print(choice.message.content)  # noqa: T201

    # Validate usage
    usage = res.get("usage")
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


@pytest.mark.asyncio
async def test_litellm_stream():
    # Test API
    res: CustomStreamWrapper = await acompletion(  # pyright: ignore[reportAssignmentType]
        api_key="mock",
        base_url="http://localhost:8080/v1",
        messages=MESSAGES,
        model="openai/gpt-4.1-nano",
        stream=True,
    )

    # Consume stream
    async for delta in res:
        last_delta = False

        # Validate response
        assert len(delta.choices) == 1

        # Validate choice
        choice = delta.choices[0]
        assert isinstance(choice, StreamingChoices)
        if choice.delta.content is None:
            assert choice.finish_reason == "stop"
            last_delta = True
        else:
            assert choice.finish_reason is None
        print(choice.delta.content)  # noqa: T201

        # Validate usage
        usage = delta.get("usage", None)
        if last_delta:
            pass
            # TODO: Validate usage, "usage" field is never present in the object
        else:
            assert usage is None
