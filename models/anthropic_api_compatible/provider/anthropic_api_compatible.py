from typing import Mapping
from dify_plugin import ModelProvider


class AnthropicCompatibleProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        For customizable-model configuration, this method is empty.
        Credentials are validated at the model level.
        """
        pass
