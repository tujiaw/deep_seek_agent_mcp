# DeepSeek Agent MCP

DeepSeek Agent MCP 是一个利用 DeepSeek AI 模型与 MCP（Model-Client-Protocol）框架创建多功能代理系统的项目。该项目包含一个天气助手，可以获取天气信息、执行 Shell 命令并提供当前时间信息。

![命令行界面截图](https://fibmocuqjpkyzrzoydzq.supabase.co/storage/v1/object/public/drop2/uploads/pasted-image-1744034159874-1744034162405.png)
## 功能特点

- 使用高德地图 API 获取天气信息
- Shell 命令执行能力
- 当前时间报告
- 大语言模型的流式响应
- 交互式聊天界面

## 系统要求

- Python 3.13 或更高版本
- OpenAI Agent 框架
- DeepSeek API 密钥
- 高德地图天气 API 密钥（用于天气功能）

## 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/tujiaw/deep_seek_agent_mcp.git
   cd deep_seek_agent_mcp
   ```

2. 设置虚拟环境：
   ```bash
   python -m venv .venv
   # Windows 系统
   .venv\Scripts\activate
   # Linux/Mac 系统
   source .venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install uv
   uv pip install -e .
   ```

4. 配置环境变量：
   - 基于提供的 `.env.example` 创建 `.env` 文件
   - 添加您的 DeepSeek API 密钥和高德地图天气 API 密钥

## 使用方法

### 启动 MCP 服务器

```bash
uv run python ./mcp_server.py
```

服务器默认将在 `0.0.0.0:8080` 上启动。您可以指定不同的主机和端口：

```bash
uv run python ./mcp_server.py --host 127.0.0.1 --port 9000
```

### 运行客户端应用

```bash
uv run python ./llm_client.py
```

这将启动一个交互式会话，您可以与助手进行对话。

## 配置说明

### 环境变量

- `API_KEY`：您的 DeepSeek API 密钥
- `BASE_URL`：DeepSeek API 基础 URL（默认：https://api.deepseek.com）
- `MODEL_NAME`：使用的大语言模型（默认：deepseek-chat）
- `OPENWEATHER_API_KEY`：您的高德地图天气 API 密钥

### MCP 配置

`mcp.json` 文件包含连接 MCP 服务器的配置：

```json
{
  "mcpServers": {
    "weather_sse": {
      "url": "http://localhost:8080/sse",
      "env": {
        "API_KEY": ""
      }
    }
  }
}
```

## 系统架构

系统由两个主要组件组成：

1. **MCP 服务器**（`mcp_server.py`）：通过 MCP 协议向大语言模型代理提供工具和功能
2. **LLM 客户端**（`llm_client.py`）：管理与 DeepSeek AI 模型和 MCP 服务器的通信

## 可用工具

- `get_weather`：获取指定城市的天气信息
- `power_shell`：执行 Windows shell 命令
- `now_time`：返回当前时间

