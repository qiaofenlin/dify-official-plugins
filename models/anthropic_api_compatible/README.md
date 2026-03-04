# Anthropic API Compatible Plugin

A Dify plugin that provides compatibility with Anthropic's API standard. This plugin allows you to connect to any API endpoint that follows the Anthropic Messages API format.

## Features

- **Customizable Model**: Define your own model names
- **Multiple Authentication Methods**:
  - API Key (x-api-key header)
  - Bearer Token (Authorization header)
- **Custom Headers**: Add custom request headers in `key1:value1,key2:value2` format
- **Custom Endpoint**: Connect to any Anthropic-compatible API endpoint

## Configuration

### Required Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Model Name | The model identifier to use | - |
| API Key | Your API key for authentication | - |
| API Endpoint URL | The base URL for the API | https://api.anthropic.com |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Authentication Type | API Key or Bearer Token | API Key |
| Model Display Name | Custom display name for the model | - |
| Context Size | Maximum context length | 200000 |
| Max Output Tokens | Maximum tokens in response | 4096 |
| Custom Headers | Additional request headers | - |

## Custom Headers Format

Custom headers should be specified in the following format:
```
key1:value1,key2:value2
```

Example:
```
X-Custom-Header:custom-value,X-Another-Header:another-value
```

## Usage

1. Install the plugin in Dify
2. Configure your model with the required parameters
3. Start using your Anthropic-compatible model

## Supported Model Types

- LLM (Large Language Model)

## API Compatibility

This plugin is compatible with any API that follows the Anthropic Messages API format:
- Endpoint: `{base_url}/v1/messages`
- Request format: Anthropic Messages API
- Response format: Anthropic Messages API

## License

MIT License
