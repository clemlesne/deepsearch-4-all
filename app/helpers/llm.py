import asyncio
from collections.abc import Awaitable, Callable
from json import loads
from textwrap import dedent
from typing import TypeVar

import litellm
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig, CrawlResult
from diskcache import Cache as DiskCache
from litellm import ChatCompletionMessageToolCall, Choices, acompletion
from litellm.files.main import ModelResponse
from litellm.types.caching import LiteLLMCacheType
from litellm.types.completion import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from litellm.types.utils import Message
from litellm.utils import function_to_dict, token_counter
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.helpers.logging import logger
from app.models.chat_completion import (
    Usage,
)

# Type hints
MessagesList = list[
    Message
    | ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
]

# Enable Litellm cache
litellm.enable_cache(
    disk_cache_dir=".litellm_cache",
    type=LiteLLMCacheType.DISK,
)

# Crawler configuration
CRAWL_CONFIG = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    verbose=False,
)
CRAWL_CACHE = DiskCache(".crawl4ai_cache")

# Type validation
P = TypeVar("P", bound=bool | float | int | str | BaseModel)
T = TypeVar("T")

# LLM
MAX_SIMULTANEOUS_TOOLS = 5


class CompletionException(Exception):
    pass


class ValidationException(Exception):
    pass


async def non_empty_completion(  # noqa: PLR0913
    model: str,
    system: str,
    temperature: float,
    top_p: float,
    usage: Usage,
    json: bool = False,
    existing_history: MessagesList = [],
    tools: list[Callable[..., Awaitable[str]]] = [],
) -> str:
    """
    Ask a LLM to generate a type from a LLM.

    Returns None if the response is invalid or empty.
    """
    # Explicit the response type in the system message
    system = f"""
        {dedent(system)}

        # Response format
        string
    """

    def _validate(
        req: str | None,
    ) -> str:
        """
        Validate the string, make sure it is not empty.
        """
        # Raise if empty response
        if not req:
            raise ValidationException("Empty response")

        return req

    # Ask LLM
    return await _raw_completion(
        existing_history=existing_history,
        json=json,
        model=model,
        system=system,
        temperature=temperature,
        tools=tools,
        top_p=top_p,
        usage=usage,
        validation_callback=_validate,
    )


async def validated_completion(  # noqa: PLR0913
    model: str,
    res_type: type[P],
    system: str,
    temperature: float,
    top_p: float,
    usage: Usage,
    existing_history: MessagesList = [],
    tools: list[Callable[..., Awaitable[str]]] = [],
) -> P:
    """
    Ask a LLM to generate a type from a LLM.

    Returns None if the response is invalid or empty.
    """

    class _Native(BaseModel):
        value: P

    def _validate(
        req: str | None,
    ) -> P:
        """
        Validate the response, make sure it is not empty and is of the expected type.

        Type is either a Pydantic model or a Python native type.
        """
        # Raise if empty response
        if not req:
            raise ValidationException("Empty response")

        # Return as is if matching type
        if isinstance(req, res_type):
            return req

        # Validate a Pydantic model
        if issubclass(res_type, BaseModel):
            # Validate JSON
            try:
                return res_type.model_validate_json(req)
            # Pydantic validation error
            except ValidationError as e:
                raise ValidationException(
                    e.json(
                        # Lower LLM response size and API cost
                        include_input=False,
                        include_url=False,
                    )
                )

        # Validate a native type
        try:
            return TypeAdapter(_Native).validate_python(req).value
        # Pydantic validation error
        except ValidationError as e:
            raise ValidationException(
                e.json(
                    # Lower LLM response size and API cost
                    include_input=False,
                    include_url=False,
                )
            )

    # System prompt with the response schema
    system = f"""
        {system}

        # Response JSON schema
        {res_type.model_json_schema() if issubclass(res_type, BaseModel) else _Native.model_json_schema()}
    """

    # Ask LLM
    return await _raw_completion(
        existing_history=existing_history,
        json=True,
        model=model,
        system=system,
        temperature=temperature,
        tools=tools,
        top_p=top_p,
        usage=usage,
        validation_callback=_validate,
    )


async def _raw_completion(  # noqa: PLR0913
    existing_history: MessagesList,
    json: bool,
    model: str,
    system: str,
    temperature: float,
    tools: list[Callable[..., Awaitable[str]]],
    top_p: float,
    usage: Usage,
    validation_callback: Callable[[str | None], T],
    _previous_result: str | None = None,
    _retries_remaining: int = 10,
    _validation_error: str | None = None,
) -> T:
    """
    Get a response from the LLM.

    Exception is raised if the response is truncated or empty.
    """
    local_history = existing_history.copy()
    system_message = ChatCompletionSystemMessageParam(
        content=dedent(system),
        role="system",
    )

    # Convert functions to expected API schema
    tools_dict = [
        {
            "type": "function",
            "function": function_to_dict(tool),
        }
        for tool in tools
    ]

    # Add previous result and validation error
    if _validation_error:
        logger.debug(
            "LLM validation error, retrying (%i retries left)", _retries_remaining
        )

        # Add previous result if available
        if _previous_result:
            local_history.append(
                ChatCompletionAssistantMessageParam(
                    content=_previous_result,
                    role="assistant",
                )
            )

        # Add validation error
        local_history.append(
            ChatCompletionUserMessageParam(
                content=f"A validation error occurred, please retry: {_validation_error}",
                role="user",
            )
        )

    # Build history for the request
    sent_history: MessagesList = [
        system_message,
        *local_history,
    ]  # History + system message

    res: ModelResponse = await acompletion(
        caching=True,  # Use Litellm cache
        messages=sent_history,
        model=model,
        response_format={"type": "json_object"} if json else None,
        seed=42,  # Enhance reproducibility
        temperature=temperature,
        tools=tools_dict,
        top_p=top_p,
    )  # pyright: ignore[reportAssignmentType]
    choice: Choices = res.choices[0]  # pyright: ignore[reportAssignmentType]

    # Update usage
    # TODO: Consumption is not accurate at all with that method, use data from the response
    usage.completion_tokens += token_counter(
        messages=[choice.message.model_dump(exclude={"annotations"})],
        model=model,
    )  # Increment completion from the response string
    usage.prompt_tokens += token_counter(
        messages=[
            message.model_dump(exclude={"annotations"})
            if isinstance(message, BaseModel)
            else message
            for message in sent_history
        ],
        model=model,
    )  # Increment prompt from the request objects

    # Check for tools
    tool_calls = choice.message.tool_calls
    if tool_calls:
        # List available functions
        available_functions = {function.__name__: function for function in tools}

        # Extend conversation with assistant's reply
        local_history.append(choice.message)

        # Execute tools and add them to history
        local_history.extend(
            await asyncio.gather(
                *[
                    _execute_tool(
                        available_functions=available_functions,
                        position=i,
                        tool_call=tool_call,
                    )
                    for i, tool_call in enumerate(tool_calls)
                ]
            )
        )

        # Re-run the completion with the tool results
        return await _raw_completion(
            existing_history=local_history,
            json=json,
            model=model,
            system=system,
            temperature=temperature,
            tools=[],  # Don't execute tools twice
            top_p=top_p,
            usage=usage,
            validation_callback=validation_callback,
            _previous_result=_previous_result,
            _retries_remaining=_retries_remaining,
            _validation_error=_validation_error,
        )

    finish_reason = choice.finish_reason
    # Alert if response is truncated
    if finish_reason == "length":
        logger.warning("LLM response truncated")
    # Raise if response failed
    elif finish_reason != "stop":
        raise CompletionException(
            f"Completion did not finish correctly: {finish_reason}"
        )

    # Raise if no content
    content = choice.message.content
    if not content:
        raise CompletionException("Completion message is empty")

    # Return parsed content
    try:
        return validation_callback(content)

    # Retry if validation failed
    except ValidationException as e:
        validation_error = str(e)

        # Raise if no retries left
        if _retries_remaining == 0:
            raise CompletionException(
                f"Validation failed and no retries left: {validation_error}"
            )

        # Retry
        logger.debug(
            "LLM validation error, retrying (%i retries left)", _retries_remaining
        )
        return await _raw_completion(
            existing_history=local_history,
            json=json,
            model=model,
            system=system,
            temperature=temperature,
            tools=tools,
            top_p=top_p,
            usage=usage,
            validation_callback=validation_callback,
            _previous_result=content,
            _retries_remaining=_retries_remaining - 1,
            _validation_error=validation_error,
        )


async def _execute_tool(
    available_functions: dict[str, Callable[..., Awaitable[str]]],
    position: int,
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionToolMessageParam:
    # Skip if no function name
    function_name = tool_call.function.name
    if not function_name:
        return ChatCompletionToolMessageParam(
            content="No function name provided.",
            role="tool",
            tool_call_id=tool_call.id,
        )

    # Skip if function not available
    if function_name not in available_functions:
        return ChatCompletionToolMessageParam(
            content=f"Function '{function_name}' not available.",
            role="tool",
            tool_call_id=tool_call.id,
        )

    # Skip if too many tools
    if position >= MAX_SIMULTANEOUS_TOOLS:
        return ChatCompletionToolMessageParam(
            content=f"Too many tools called at once (limit is {MAX_SIMULTANEOUS_TOOLS}).",
            role="tool",
            tool_call_id=tool_call.id,
        )

    # Execute
    logger.debug("Executing tool: %s", function_name)
    function_to_call = available_functions[function_name]

    # Try parse arguments and execute
    try:
        function_response = await function_to_call(
            **loads(tool_call.function.arguments)
        )
    except TypeError as e:
        logger.debug("Tool execution failed: %s", e)
        return ChatCompletionToolMessageParam(
            content=f"Bad arguments: {e}",
            role="tool",
            tool_call_id=tool_call.id,
        )

    return ChatCompletionToolMessageParam(
        content=function_response,
        role="tool",
        tool_call_id=tool_call.id,
    )


def read_url_tool(
    model: str,
    temperature: float,
    top_p: float,
    usage: Usage,
) -> Callable[..., Awaitable[str]]:
    async def _read_url(
        url: str,
    ) -> str:
        """
        Read a URL and return the content.

        Content is cached for 7 days and is an analysis of the raw web page.
        """
        # Try cache
        cache_key = url
        cache = CRAWL_CACHE.get(cache_key)
        if cache:
            return cache  # pyright: ignore[reportReturnType]

        # Scrape
        async with AsyncWebCrawler() as crawler:
            result: CrawlResult = await crawler.arun(
                config=CRAWL_CONFIG,
                url=url,
            )  # pyright: ignore[reportAssignmentType]
        content = str(result.markdown)

        synthesis = await non_empty_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            usage=usage,
            existing_history=[
                ChatCompletionUserMessageParam(
                    content=content,
                    role="user",
                ),
            ],
            system=f"""
                Assistant is an archivist with 20 years of experience.

                # Objective
                Extract the meaning of the web page. It is not a summary, but a detailed analysis of the content, with persons, facts, sources and dates.

                # Rules
                - Be concise and precise
                - Don't make assumptions
            """,
        )

        # Update cache
        CRAWL_CACHE.set(
            expire=60 * 60 * 24 * 7,  # 7 days
            key=cache_key,
            value=synthesis,
        )

        # logger.debug("Parsed URL %s: %s", url, synthesis)
        return synthesis

    return _read_url
