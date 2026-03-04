# Anthropic API Compatible Model Provider

このプラグインにより、DifyでAnthropic互換のAPIエンドポイントを使用できます。

## 機能

- Anthropic Claudeモデルのサポート
- カスタムAPIエンドポイント設定
- カスタムリクエストヘッダーのサポート
- マルチモーダルモデルのビジョンサポート
- ツール呼び出しのサポート
- ストリーミングレスポンスのサポート

## 設定

### 必須パラメータ

| パラメータ | 説明 |
|-----------|-------------|
| API Key | Anthropic APIキー |
| Model Name | 使用するモデル（例：claude-3-opus-20240229） |

### オプションパラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|-------------|---------|
| API Endpoint URL | カスタムAPIエンドポイント | https://api.anthropic.com |
| Custom Headers | JSON形式のカスタムヘッダー | {} |
| Context Size | 最大コンテキストウィンドウ | 200000 |
| Max Output Tokens | 最大出力トークン数 | 4096 |
| Vision Support | ビジョン機能を有効化 | サポート |
| Tool Call Support | ツール呼び出しを有効化 | サポート |
| Stream Mode | ストリーミングを有効化 | 有効 |

## カスタムヘッダー

JSON形式でカスタムリクエストヘッダーを設定できます：

```json
{
  "X-Custom-Header": "value",
  "X-Another-Header": "another-value"
}
```

## サポートされるモデル

- Claude 3 Opus
- Claude 3 Sonnet
- Claude 3 Haiku
- Claude 3.5 Sonnet
- Anthropic互換のモデル

## 使用方法

1. Difyにこのプラグインをインストール
2. モデルプロバイダー設定を構成
3. APIキーとエンドポイントを追加
4. アプリケーションでAnthropicモデルの使用を開始
