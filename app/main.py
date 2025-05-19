import asyncio
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from app.helpers.logging import VERSION, logger
from app.models.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from app.think import think_stream, think_sync

# First log
logger.info(
    "deepsearch-4-all v%s",
    VERSION,
)

# FastAPI
api = FastAPI(
    contact={
        "url": "https://github.com/clemlesne/deepsearch-4-all",
    },
    description="DeepSearch exposed as a standard LLM API with thinking. Swappable model backend.",
    license_info={
        "name": "Apache-2.0",
        "url": "https://github.com/clemlesne/deepsearch-4-all/blob/master/LICENSE",
    },
    title="deepsearch-4-all",
    version=VERSION,
)


@api.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
async def v1_chat_completions_sync(
    req: ChatCompletionRequest,
) -> ChatCompletionResponse | EventSourceResponse:
    if not req.stream:
        logger.debug("Sync request")
        return await think_sync(req)

    logger.debug("Async request")
    completions_queue: asyncio.Queue[ChatCompletionResponse] = asyncio.Queue()
    think_task = asyncio.create_task(
        think_stream(
            completions_queue=completions_queue,
            req=req,
        )
    )

    async def _generator() -> AsyncGenerator[str]:
        try:
            while True:
                # Consume
                completion = await completions_queue.get()
                yield completion.model_dump_json()
                completions_queue.task_done()

                # Stop if finish reason is not empty
                if any(
                    choice.finish_reason is not None for choice in completion.choices
                ):
                    break
        finally:
            # Cancel task
            think_task.cancel()

    return EventSourceResponse(
        content=_generator(),
        media_type="text/event-stream",
    )
