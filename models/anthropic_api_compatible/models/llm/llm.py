import json
from collections.abc import Generator, Mapping
from typing import Any, Optional, Union

from anthropic import Anthropic, Stream
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    Message,
    MessageStartEvent,
    MessageStreamEvent,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
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
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel


class AnthropicAPILargeLanguageModel(LargeLanguageModel):
    """
    Anthropic API Compatible Large Language Model
    """

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
    ) -> Union[Generator, Any]:
        """
        Invoke large language model
        """
        return self._handle_invoke(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def _handle_invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[Generator, Any]:
        """
        Handle invoke
        """
        client = self._init_client(credentials)

        system_prompt, messages = self._convert_prompt_messages(prompt_messages)

        extra_headers = {}
        headers_config = credentials.get("headers", "{}")
        if headers_config:
            try:
                custom_headers = json.loads(headers_config) if isinstance(headers_config, str) else headers_config
                if isinstance(custom_headers, dict):
                    extra_headers.update(custom_headers)
            except (json.JSONDecodeError, TypeError):
                pass

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": model_parameters.get(
                "max_tokens", int(credentials.get("max_tokens", 4096))
            ),
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        if stop:
            request_kwargs["stop_sequences"] = stop

        if "temperature" in model_parameters:
            request_kwargs["temperature"] = model_parameters["temperature"]

        if "top_p" in model_parameters:
            request_kwargs["top_p"] = model_parameters["top_p"]

        if "top_k" in model_parameters:
            request_kwargs["top_k"] = model_parameters["top_k"]

        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers

        if stream:
            request_kwargs["stream"] = True
            return self._handle_stream_response(
                client, request_kwargs, model, prompt_messages
            )
        else:
            response = client.messages.create(**request_kwargs)
            return self._handle_response(response, model, prompt_messages)

    def _init_client(self, credentials: dict) -> Anthropic:
        """
        Initialize Anthropic client
        """
        client_kwargs: dict[str, Any] = {
            "api_key": credentials.get("api_key"),
        }

        endpoint_url = credentials.get("endpoint_url")
        if endpoint_url:
            client_kwargs["base_url"] = endpoint_url

        headers_config = credentials.get("headers", "{}")
        if headers_config:
            try:
                custom_headers = json.loads(headers_config) if isinstance(headers_config, str) else headers_config
                if isinstance(custom_headers, dict) and custom_headers:
                    client_kwargs["default_headers"] = custom_headers
            except (json.JSONDecodeError, TypeError):
                pass

        return Anthropic(**client_kwargs)

    def _convert_prompt_messages(
        self, prompt_messages: list[PromptMessage]
    ) -> tuple[Optional[str], list[dict]]:
        """
        Convert prompt messages to Anthropic format
        """
        system_prompt = None
        messages = []

        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                system_prompt = message.content
            elif isinstance(message, UserPromptMessage):
                content = self._convert_user_message_content(message)
                messages.append({"role": "user", "content": content})
            elif isinstance(message, AssistantPromptMessage):
                content = self._convert_assistant_message_content(message)
                messages.append({"role": "assistant", "content": content})
            elif isinstance(message, ToolPromptMessage):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ]
                messages.append({"role": "user", "content": content})

        return system_prompt, messages

    def _convert_user_message_content(self, message: UserPromptMessage) -> Any:
        """
        Convert user message content
        """
        if isinstance(message.content, str):
            return message.content

        content = []
        for item in message.content:
            if isinstance(item, ImagePromptMessageContent):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": item.mime_type,
                        "data": item.data,
                    },
                })
            else:
                content.append({"type": "text", "text": item.data if hasattr(item, "data") else str(item)})

        return content if content else message.content

    def _convert_assistant_message_content(self, message: AssistantPromptMessage) -> Any:
        """
        Convert assistant message content
        """
        content = []

        if message.content:
            content.append({"type": "text", "text": message.content})

        if message.tool_calls:
            for tool_call in message.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": json.loads(tool_call.function.arguments)
                    if isinstance(tool_call.function.arguments, str)
                    else tool_call.function.arguments,
                })

        return content if content else ""

    def _convert_tools(self, tools: list[PromptMessageTool]) -> list[dict]:
        """
        Convert tools to Anthropic format
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.parameters,
            })
        return anthropic_tools

    def _handle_stream_response(
        self,
        client: Anthropic,
        request_kwargs: dict,
        model: str,
        prompt_messages: list[PromptMessage],
    ) -> Generator:
        """
        Handle stream response
        """
        with client.messages.stream(**request_kwargs) as stream:
            text_content = ""
            tool_calls = []

            for event in stream:
                if isinstance(event, MessageStartEvent):
                    pass
                elif isinstance(event, ContentBlockStartEvent):
                    if event.content_block.type == "tool_use":
                        tool_calls.append({
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "input": "",
                        })
                elif isinstance(event, ContentBlockDeltaEvent):
                    if event.delta.type == "text_delta":
                        text_content += event.delta.text
                        yield event.delta.text
                    elif event.delta.type == "input_json_delta":
                        if tool_calls:
                            tool_calls[-1]["input"] += event.delta.partial_json

            final_message = stream.get_final_message()

            usage = {
                "prompt_tokens": final_message.usage.input_tokens,
                "completion_tokens": final_message.usage.output_tokens,
            }

            assistant_message = AssistantPromptMessage(content=text_content)

            if tool_calls:
                assistant_message.tool_calls = []
                for tool_call in tool_calls:
                    try:
                        tool_input = json.loads(tool_call["input"]) if tool_call["input"] else {}
                    except json.JSONDecodeError:
                        tool_input = {}

                    assistant_message.tool_calls.append(
                        AssistantPromptMessage.ToolCall(
                            id=tool_call["id"],
                            type="function",
                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                name=tool_call["name"],
                                arguments=json.dumps(tool_input),
                            ),
                        )
                    )

            from dify_plugin.entities.model.llm import LLMResult, LLMUsage

            yield LLMResult(
                model=model,
                prompt_messages=prompt_messages,
                message=assistant_message,
                usage=LLMUsage(**usage),
            )

    def _handle_response(
        self,
        response: Message,
        model: str,
        prompt_messages: list[PromptMessage],
    ) -> Any:
        """
        Handle non-stream response
        """
        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        assistant_message = AssistantPromptMessage(content=text_content)

        if tool_calls:
            assistant_message.tool_calls = []
            for tool_call in tool_calls:
                assistant_message.tool_calls.append(
                    AssistantPromptMessage.ToolCall(
                        id=tool_call["id"],
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["input"]),
                        ),
                    )
                )

        from dify_plugin.entities.model.llm import LLMResult, LLMUsage

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }

        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=assistant_message,
            usage=LLMUsage(**usage),
        )

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        """
        Get customizable model schema
        """
        features = []

        vision_support = credentials.get("vision_support", "support")
        if vision_support == "support":
            features.append(ModelFeature.VISION)

        tool_call_support = credentials.get("tool_call_support", "support")
        if tool_call_support == "support":
            features.append(ModelFeature.TOOL_CALL)

        stream_mode = credentials.get("stream_mode", "enabled")
        if stream_mode == "enabled":
            features.append(ModelFeature.STREAM_TOOL_CALL)

        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=features,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(
                    credentials.get("context_size", 200000)
                ),
                ModelPropertyKey.MAX_OUTPUT: int(
                    credentials.get("max_tokens", 4096)
                ),
            },
            parameter_rules=[
                ParameterRule(
                    name="max_tokens",
                    type=ParameterType.INT,
                    use_template="max_tokens",
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大输出 Tokens"),
                    default=int(credentials.get("max_tokens", 4096)),
                    min=1,
                    max=100000,
                ),
                ParameterRule(
                    name="temperature",
                    type=ParameterType.FLOAT,
                    use_template="temperature",
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    default=1.0,
                    min=0.0,
                    max=1.0,
                ),
                ParameterRule(
                    name="top_p",
                    type=ParameterType.FLOAT,
                    use_template="top_p",
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    default=0.9,
                    min=0.0,
                    max=1.0,
                ),
                ParameterRule(
                    name="top_k",
                    type=ParameterType.INT,
                    use_template="top_k",
                    label=I18nObject(en_US="Top K", zh_Hans="Top K"),
                    default=40,
                    min=1,
                    max=100,
                ),
            ],
            pricing=PriceConfig(
                input=0.0,
                output=0.0,
                unit=0.0,
                currency="USD",
            ),
        )

        return entity

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials
        """
        try:
            client = self._init_client(credentials)

            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )

            if not response:
                raise CredentialsValidateFailedError("Failed to validate credentials")

        except Exception as e:
            raise CredentialsValidateFailedError(str(e))

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map invoke errors
        """
        return {
            InvokeAuthorizationError: [],
            InvokeBadRequestError: [],
            InvokeRateLimitError: [],
            InvokeConnectionError: [],
        }
