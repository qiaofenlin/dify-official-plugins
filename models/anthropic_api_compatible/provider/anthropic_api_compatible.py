from collections.abc import Mapping
from typing import Any

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError


class AnthropicAPICompatibleProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        """
        try:
            from anthropic import Anthropic

            api_key = credentials.get("api_key")
            api_base = credentials.get("endpoint_url")
            headers = credentials.get("headers", "{}")

            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if api_base:
                client_kwargs["base_url"] = api_base

            if headers:
                import json
                try:
                    custom_headers = json.loads(headers) if isinstance(headers, str) else headers
                    if custom_headers:
                        client_kwargs["default_headers"] = custom_headers
                except (json.JSONDecodeError, TypeError):
                    pass

            client = Anthropic(**client_kwargs)

            response = client.messages.create(
                model=credentials.get("model_name", "claude-3-haiku-20240307"),
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )

            if not response:
                raise CredentialsValidateFailedError("Failed to validate credentials")

        except Exception as e:
            raise CredentialsValidateFailedError(str(e))
