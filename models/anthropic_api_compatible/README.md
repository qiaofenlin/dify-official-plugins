# Anthropic API Compatible Model Provider

This plugin allows you to use any Anthropic-compatible API endpoint with Dify.

## Features

- Support for Anthropic Claude models
- Custom API endpoint configuration
- Custom request headers support
- Vision support for multimodal models
- Tool calling support
- Streaming response support

## Configuration

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| API Key | Your Anthropic API key |
| Model Name | The model to use (e.g., claude-3-opus-20240229) |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| API Endpoint URL | Custom API endpoint | https://api.anthropic.com |
| Custom Headers | JSON format custom headers | {} |
| Context Size | Maximum context window | 200000 |
| Max Output Tokens | Maximum output tokens | 4096 |
| Vision Support | Enable vision capabilities | Support |
| Tool Call Support | Enable tool calling | Support |
| Stream Mode | Enable streaming | Enabled |

## Custom Headers

You can set custom request headers in JSON format:

```json
{
  "X-Custom-Header": "value",
  "X-Another-Header": "another-value"
}
```

## Supported Models

- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku
- Claude 3.5 Sonnet
- Any Anthropic-compatible model

## Usage

1. Install this plugin in Dify
2. Configure your model provider settings
3. Add your API key and endpoint
4. Start using Anthropic models in your applications
