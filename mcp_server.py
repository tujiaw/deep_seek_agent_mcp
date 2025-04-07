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
from datetime import datetime

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
        print(f"获取天气信息: {adcode}")
        api_key = os.getenv("OPENWEATHER_API_KEY")
        params = {"city": adcode, "key": api_key}

        response = requests.get(NWS_API_BASE, params=params)
        response.raise_for_status()
        print(response.json())
        return response.json()
    except Exception as e:
        print(f"获取天气信息失败: {str(e)}")
        raise


@mcp.tool()
async def power_shell(command: str) -> Dict:
    """
    执行windows命令工具

    Args:
        command: 执行的命令
    """
    try:
        import asyncio
        import subprocess

        print(f"执行命令: {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Try different encodings for Windows command output
        encodings = ['gbk', 'cp936', 'utf-8']
        stdout_text = ""
        stderr_text = ""
        
        for encoding in encodings:
            try:
                stdout_text = stdout.decode(encoding) if stdout else ""
                stderr_text = stderr.decode(encoding) if stderr else ""
                break
            except UnicodeDecodeError:
                continue
        
        return {
            "stdout": stdout_text,
            "stderr": stderr_text,
            "returncode": process.returncode
        }
    except Exception as e:
        print(f"执行命令失败: {str(e)}")
        raise


@mcp.tool()
async def now_time() -> Dict:
    """
    获取当前时间
    """
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


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