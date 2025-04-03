

# 加载 .env 文件中的环境变量

    
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from typing import Dict, Optional
from pydantic import BaseModel, Field
import os
import requests
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("weather")
NWS_API_BASE = "https://restapi.amap.com/v3/weather/weatherInfo?parameters"
USER_AGENT = "weather-app/1.0"


@mcp.tool()
async def get_weather(adcode: str) -> Dict:
    """
    获取指定城市的天气信息

    Args:
        adcode: 城市编码
        units: 温度单位 (metric: 摄氏度, imperial: 华氏度)
    """
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        params = {"city": adcode, "key": api_key}

        response = requests.get(NWS_API_BASE, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"获取天气信息失败: {str(e)}")
        raise


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provied mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)