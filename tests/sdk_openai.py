import os

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion import Choice as SyncChoice
from openai.types.chat.chat_completion_chunk import Choice as StreamChoice

# Dummy message
MESSAGES = [
    ChatCompletionUserMessageParam(
        content="What is the status of the world?",
        role="user",
    ),
]


@pytest.mark.asyncio
async def test_openai_sync():
    async with AsyncOpenAI(
        api_key="mock",
        base_url="http://localhost:8080/v1",
    ) as client:
        # Test API
        res = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=MESSAGES,
        )

        # Validate response
        assert len(res.choices) == 1

        # Validate choice
        choice = res.choices[0]
        assert isinstance(choice, SyncChoice)
        assert choice.finish_reason == "stop"
        assert choice.message.content is not None
        print(choice.message.content)  # noqa: T201

        # Validate usage
        usage = res.usage
        assert usage is not None
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


@pytest.mark.asyncio
async def test_openai_stream():
    async with AsyncOpenAI(
        api_key="mock",
        base_url="http://localhost:8080/v1",
    ) as client:
        # Test API
        res = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=MESSAGES,
            stream=True,
        )

        # Consume stream
        async for delta in res:
            last_delta = False

            # Validate response
            assert len(delta.choices) == 1

            # Validate choice
            choice = delta.choices[0]
            assert isinstance(choice, StreamChoice)
            if choice.delta.content is None:
                assert choice.finish_reason == "stop"
                last_delta = True
            else:
                assert choice.finish_reason is None
            print(choice.delta.content)  # noqa: T201

            # Validate usage
            usage = delta.usage
            if last_delta:
                assert usage is not None
                assert usage.prompt_tokens > 0
                assert usage.completion_tokens > 0
                assert (
                    usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
                )
            else:
                assert usage is None
