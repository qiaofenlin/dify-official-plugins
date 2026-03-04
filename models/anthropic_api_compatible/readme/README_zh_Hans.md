# Anthropic API 兼容模型供应商

该插件允许您在 Dify 中使用任何兼容 Anthropic API 的接口。

## 功能特性

- 支持 Anthropic Claude 系列模型
- 自定义 API 端点配置
- 支持自定义请求头
- 支持多模态模型的视觉功能
- 支持工具调用
- 支持流式响应

## 配置说明

### 必填参数

| 参数 | 描述 |
|------|------|
| API Key | 您的 Anthropic API 密钥 |
| 模型名称 | 要使用的模型（如：claude-3-opus-20240229） |

### 可选参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| API 端点 URL | 自定义 API 端点 | https://api.anthropic.com |
| 自定义请求头 | JSON 格式的自定义请求头 | {} |
| 上下文大小 | 最大上下文窗口 | 200000 |
| 最大输出 Tokens | 最大输出 token 数 | 4096 |
| 视觉支持 | 启用视觉能力 | 支持 |
| 工具调用支持 | 启用工具调用 | 支持 |
| 流式模式 | 启用流式输出 | 启用 |

## 自定义请求头

您可以以 JSON 格式设置自定义请求头：

```json
{
  "X-Custom-Header": "value",
  "X-Another-Header": "another-value"
}
```

## 支持的模型

- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku
- Claude 3.5 Sonnet
- 任何兼容 Anthropic API 的模型

## 使用方法

1. 在 Dify 中安装此插件
2. 配置模型供应商设置
3. 添加您的 API 密钥和端点
4. 在您的应用中开始使用 Anthropic 模型
