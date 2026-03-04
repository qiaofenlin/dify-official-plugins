import json
import logging
from collections.abc import Generator
from typing import Any, Mapping, Optional, Union

import anthropic
import httpx
from anthropic import Anthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    I18nObject,
    ModelFeature,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from httpx import Timeout

logger = logging.getLogger(__name__)


class AnthropicCompatibleLargeLanguageModel(LargeLanguageModel):
    """
    Large Language Model implementation for Anthropic API compatible providers.
    Supports custom endpoints, authentication types, and custom headers.
    """

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        """
        Get the model schema for a customizable model.
        """
        # Determine features based on credentials
        features = []

        # Get display name from credentials
        display_name = credentials.get("display_name", "")
        label = I18nObject(
            en_US=display_name if display_name else model,
            zh_Hans=display_name if display_name else model
        )

        # Get context size
        context_size = int(credentials.get("context_size", "200000"))
        max_tokens = int(credentials.get("max_tokens", "4096"))

        entity = AIModelEntity(
            model=model,
            label=label,
            model_type=self.model_type,
            features=features,
            fetch_from=AIModelEntity.FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                "context_size": context_size,
                "max_tokens": max_tokens,
            },
            parameter_rules=[
                # Temperature
            ],
        )

        return entity

    def _build_headers(self, credentials: dict) -> dict:
        """
        Build request headers based on credentials.
        Supports API Key and Bearer Token authentication, plus custom headers.
        """
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Authentication
        auth_type = credentials.get("auth_type", "api_key")
        api_key = credentials.get("api_key", "")

        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key

        # Custom headers
        custom_headers_str = credentials.get("custom_headers", "")
        if custom_headers_str:
            for header in custom_headers_str.split(","):
                if ":" in header:
                    key, value = header.split(":", 1)
                    headers[key.strip()] = value.strip()

        return headers

    def _get_api_url(self, credentials: dict) -> str:
        """
        Get the API URL from credentials.
        """
        endpoint_url = credentials.get("endpoint_url", "https://api.anthropic.com")
        return endpoint_url.rstrip("/")

    def _to_credential_kwargs(self, credentials: dict) -> dict:
        """
        Transform credentials to kwargs for Anthropic client.
        """
        api_key = credentials.get("api_key", "")
        endpoint_url = self._get_api_url(credentials)
        auth_type = credentials.get("auth_type", "api_key")

        kwargs = {
            "api_key": api_key,
            "timeout": Timeout(315.0, read=300.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }

        if endpoint_url != "https://api.anthropic.com":
            kwargs["base_url"] = endpoint_url

        return kwargs

    def validate_credentials(self, model: str, credentials: Mapping) -> None:
        """
        Validate model credentials by making a simple request.
        """
        try:
            credentials_dict = dict(credentials)
            kwargs = self._to_credential_kwargs(credentials_dict)
            client = Anthropic(**kwargs)

            # Make a simple request to validate credentials
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )

            if not response:
                raise CredentialsValidateFailedError("Invalid credentials")

        except anthropic.AuthenticationError as e:
            raise CredentialsValidateFailedError(f"Authentication failed: {str(e)}")
        except anthropic.BadRequestError as e:
            raise CredentialsValidateFailedError(f"Bad request: {str(e)}")
        except Exception as e:
            raise CredentialsValidateFailedError(f"Validation failed: {str(e)}")

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke the large language model.
        """
        return self._chat_generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def _chat_generate(
        self,
        *,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Generate chat completion using Anthropic API.
        """
        extra_model_kwargs: dict[str, Any] = {}

        # Build client
        kwargs = self._to_credential_kwargs(credentials)
        client = Anthropic(**kwargs)

        # Handle max_tokens
        if "max_tokens" not in model_parameters:
            model_parameters["max_tokens"] = int(credentials.get("max_tokens", "4096"))

        # Handle stop sequences
        if stop:
            extra_model_kwargs["stop_sequences"] = stop

        # Convert prompt messages
        system, prompt_message_dicts = self._convert_prompt_messages(prompt_messages)
        if system:
            extra_model_kwargs["system"] = system

        # Build request
        request_params = {
            "model": model,
            "messages": prompt_message_dicts,
            "stream": stream,
            **model_parameters,
            **extra_model_kwargs,
        }

        # Handle tools
        if tools:
            extra_model_kwargs["tools"] = [
                self._transform_tool_prompt(tool) for tool in tools
            ]
            request_params["tools"] = extra_model_kwargs["tools"]

        try:
            response = client.messages.create(**request_params)
        except anthropic.APIError as e:
            raise self._handle_api_error(e)

        if stream:
            return self._handle_chat_generate_stream_response(
                model, credentials, response, prompt_messages
            )

        return self._handle_chat_generate_response(
            model, credentials, response, prompt_messages
        )

    def _convert_prompt_messages(
        self, prompt_messages: list[PromptMessage]
    ) -> tuple[Union[str, None], list[dict]]:
        """
        Convert prompt messages to Anthropic format.
        Returns (system, messages) tuple.
        """
        system = None
        messages = []

        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                if isinstance(message.content, str):
                    system = message.content
                elif isinstance(message.content, list):
                    system_parts = []
                    for content in message.content:
                        if hasattr(content, 'data'):
                            system_parts.append(content.data)
                    system = "\n".join(system_parts)
            elif isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    messages.append({"role": "user", "content": message.content})
                elif isinstance(message.content, list):
                    content_parts = []
                    for content in message.content:
                        if hasattr(content, 'type'):
                            if content.type == "text":
                                content_parts.append({
                                    "type": "text",
                                    "text": content.data
                                })
                            elif content.type == "image":
                                # Handle image content
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": content.data
                                    }
                                })
                    messages.append({"role": "user", "content": content_parts})
            elif isinstance(message, AssistantPromptMessage):
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        })
                messages.append({"role": "assistant", "content": content})
            elif isinstance(message, ToolPromptMessage):
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content
                    }]
                })

        return system, messages

    def _transform_tool_prompt(self, tool: PromptMessageTool) -> dict:
        """
        Transform tool prompt to Anthropic format.
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters
        }

    def _handle_chat_generate_response(
        self,
        model: str,
        credentials: dict,
        response: Message,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """
        Handle non-streaming response.
        """
        assistant_prompt_message = AssistantPromptMessage(content="", tool_calls=[])

        for content in response.content:
            if content.type == "text":
                assistant_prompt_message.content += content.text
            elif content.type == "tool_use":
                tool_call = AssistantPromptMessage.ToolCall(
                    id=content.id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=content.name,
                        arguments=json.dumps(content.input)
                    )
                )
                assistant_prompt_message.tool_calls.append(tool_call)

        prompt_tokens = response.usage.input_tokens if response.usage else 0
        completion_tokens = response.usage.output_tokens if response.usage else 0

        usage = self._calc_response_usage(
            model=model,
            credentials=credentials,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )

        return LLMResult(
            model=response.model,
            prompt_messages=prompt_messages,
            message=assistant_prompt_message,
            usage=usage,
        )

    def _handle_chat_generate_stream_response(
        self,
        model: str,
        credentials: dict,
        response: anthropic.Stream[MessageStreamEvent],
        prompt_messages: list[PromptMessage],
    ) -> Generator:
        """
        Handle streaming response.
        """
        full_assistant_content = ""
        return_model = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = None
        index = 0
        tool_calls: list[AssistantPromptMessage.ToolCall] = []

        current_tool_name = None
        current_tool_id = None
        current_tool_params = ""

        for chunk in response:
            if isinstance(chunk, MessageStartEvent):
                if chunk.message:
                    return_model = chunk.message.model
                    input_tokens = chunk.message.usage.input_tokens
            elif hasattr(chunk, "type") and chunk.type == "content_block_start":
                if hasattr(chunk, "content_block"):
                    content_block = chunk.content_block

                    if getattr(content_block, 'type', None) == "tool_use":
                        current_tool_name = getattr(content_block, 'name', None)
                        current_tool_id = getattr(content_block, 'id', None)

                        if current_tool_name and current_tool_id:
                            tool_call = AssistantPromptMessage.ToolCall(
                                id=current_tool_id,
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=current_tool_name,
                                    arguments=""
                                )
                            )
                            tool_calls.append(tool_call)
            elif isinstance(chunk, ContentBlockDeltaEvent):
                if hasattr(chunk.delta, "type") and chunk.delta.type == "input_json_delta":
                    if hasattr(chunk.delta, "partial_json"):
                        partial_json = chunk.delta.partial_json
                        if partial_json:
                            current_tool_params += partial_json

                            for tc in tool_calls:
                                if tc.id == current_tool_id:
                                    tc.function.arguments = current_tool_params
                                    break
                elif hasattr(chunk.delta, "text"):
                    chunk_text = chunk.delta.text or ""
                    full_assistant_content += chunk_text
                    assistant_prompt_message = AssistantPromptMessage(content=chunk_text)
                    index = chunk.index
                    yield LLMResultChunk(
                        model=return_model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=chunk.index,
                            message=assistant_prompt_message
                        )
                    )
            elif isinstance(chunk, MessageDeltaEvent):
                output_tokens = chunk.usage.output_tokens
                finish_reason = chunk.delta.stop_reason
            elif isinstance(chunk, MessageStopEvent):
                usage = self._calc_response_usage(
                    model,
                    credentials,
                    input_tokens,
                    output_tokens
                )

                for tool_call in tool_calls:
                    if not tool_call.function.arguments:
                        tool_call.function.arguments = "{}"

                yield LLMResultChunk(
                    model=return_model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=index + 1,
                        message=AssistantPromptMessage(
                            content="",
                            tool_calls=tool_calls
                        ),
                        finish_reason=finish_reason,
                        usage=usage,
                    )
                )

    def _handle_api_error(self, error: anthropic.APIError) -> InvokeError:
        """
        Handle API errors and convert to appropriate InvokeError.
        """
        if isinstance(error, anthropic.APIConnectionError):
            return InvokeConnectionError(str(error))
        elif isinstance(error, anthropic.APITimeoutError):
            return InvokeConnectionError(str(error))
        elif isinstance(error, anthropic.InternalServerError):
            return InvokeServerUnavailableError(str(error))
        elif isinstance(error, anthropic.RateLimitError):
            return InvokeRateLimitError(str(error))
        elif isinstance(error, anthropic.AuthenticationError):
            return InvokeAuthorizationError(str(error))
        elif isinstance(error, anthropic.PermissionDeniedError):
            return InvokeAuthorizationError(str(error))
        elif isinstance(error, anthropic.BadRequestError):
            return InvokeBadRequestError(str(error))
        elif isinstance(error, anthropic.NotFoundError):
            return InvokeBadRequestError(str(error))
        elif isinstance(error, anthropic.UnprocessableEntityError):
            return InvokeBadRequestError(str(error))
        else:
            return InvokeBadRequestError(str(error))

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error.
        """
        return {
            InvokeConnectionError: [
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ],
            InvokeServerUnavailableError: [anthropic.InternalServerError],
            InvokeRateLimitError: [anthropic.RateLimitError],
            InvokeAuthorizationError: [
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
            ],
            InvokeBadRequestError: [
                anthropic.BadRequestError,
                anthropic.NotFoundError,
                anthropic.UnprocessableEntityError,
                anthropic.APIError,
            ],
        }
