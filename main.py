import asyncio  # 异步IO支持
import os  # 系统操作
import traceback  # 异常堆栈追踪
import json  # 用于解析JSON文件
from typing import List, Optional, Union  # 类型提示

# 导入Agents SDK相关组件
from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
    ModelSettings,
)
from openai import AsyncOpenAI  # OpenAI异步API客户端
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent  # 响应事件类型

# MCP服务器连接组件
from agents.mcp import MCPServerSse, MCPServerSseParams, MCPServer, MCPServerStdio, MCPServerStdioParams

# 环境变量配置
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 禁用追踪，否则会报错：Tracing: request failed
set_tracing_disabled(True)

class Config:
    """配置管理类，集中管理所有配置信息"""
    
    # API配置
    API_KEY = os.environ["API_KEY"]
    BASE_URL = os.environ["BASE_URL"]
    MODEL_NAME = os.environ["MODEL_NAME"]
    
    # MCP服务器配置
    MCP_CONFIG_PATH = "mcp.json"
    
    # 模型配置
    MODEL_SETTINGS = ModelSettings(
        temperature=0.2,  # 较低温度提高回答精确性
        top_p=0.9,  # 保持适度词汇多样性
        max_tokens=4096,  # 输出长度限制
        tool_choice="auto",  # 自动选择最合适的工具
        parallel_tool_calls=True,  # 允许并行调用多个工具提高效率
        truncation="auto"  # 自动管理长文本
    )
    
    # Agent配置
    MAX_TURNS = 10  # 最大对话回合数

class DeepSeekModelProvider(ModelProvider):
    """DeepSeek模型提供器，通过OpenAI兼容接口连接DeepSeek API"""
    
    def __init__(self):
        """初始化模型提供器"""
        self.client = AsyncOpenAI(
            base_url=Config.BASE_URL,
            api_key=Config.API_KEY
        )
    
    def get_model(self, model_name: Optional[str] = None) -> Model:
        """获取模型实例
        
        Args:
            model_name: 模型名称，若为空则使用默认模型
            
        Returns:
            配置好的OpenAI兼容模型实例
        """
        return OpenAIChatCompletionsModel(
            model=model_name or Config.MODEL_NAME, 
            openai_client=self.client
        )

class MCPServerManager:
    """MCP服务器管理类，负责创建、连接和清理MCP服务器"""
    
    @staticmethod
    async def create_sse_server(name: str, url: str, cache_tools: bool = False, env: dict = None) -> MCPServer:
        """创建并连接MCP SSE服务器
        
        Args:
            name: 服务器名称
            url: 服务器URL
            cache_tools: 是否缓存工具列表
            env: 环境变量
            
        Returns:
            已连接的MCP服务器实例
        """
        print(f"正在初始化MCP SSE服务器: {name}...")
        
        # 设置环境变量
        if env:
            for key, value in env.items():
                os.environ[key] = value
        
        server = MCPServerSse(
            name=name,
            params=MCPServerSseParams(url=url),
            cache_tools_list=cache_tools
        )

        print(f"正在连接到MCP服务器: {name}...")
        await server.connect()
        print(f"MCP服务器 {name} 连接成功！")
        
        return server
    
    @staticmethod
    async def create_stdio_server(name: str, command: str, args: List[str], cache_tools: bool = False, env: dict = None) -> MCPServer:
        """创建并连接MCP STDIO服务器
        
        Args:
            name: 服务器名称
            command: 命令
            args: 命令参数
            cache_tools: 是否缓存工具列表
            env: 环境变量
            
        Returns:
            已连接的MCP服务器实例
        """
        print(f"正在初始化MCP STDIO服务器: {name}...")
        
        # 设置环境变量
        if env:
            for key, value in env.items():
                os.environ[key] = value
        
        server = MCPServerStdio(
            name=name,
            params=MCPServerStdioParams(command=command, args=args),
            cache_tools_list=cache_tools
        )

        print(f"正在连接到MCP服务器: {name}...")
        await server.connect()
        print(f"MCP服务器 {name} 连接成功！")
        
        return server
    
    @staticmethod
    async def cleanup_servers(servers: List[MCPServer]) -> None:
        """清理MCP服务器资源
        
        Args:
            servers: 要清理的MCP服务器列表
        """
        for server in servers:
            try:
                await server.cleanup()
                print(f"MCP服务器 {server.name} 资源清理成功！")
            except Exception as e:
                print(f"清理MCP服务器 {server.name} 资源时出错: {e}")
                traceback.print_exc()

class ResponseHandler:
    """响应处理类，处理模型响应数据"""
    
    @staticmethod
    async def handle_streaming_response(result) -> None:
        """处理流式响应
        
        Args:
            result: 流式响应结果
        """
        print("流式回复:", end="", flush=True)
        try:
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    # 处理文本增量事件 - 逐token输出
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        print(event.data.delta, end="", flush=True)
                    # 处理内容部分完成事件
                    elif isinstance(event.data, ResponseContentPartDoneEvent):
                        print(f"\n", end="", flush=True)
        except Exception as e:
            print(f"处理流式响应事件时发生错误: {e}", flush=True)
            traceback.print_exc()

class WeatherAssistant:
    """天气助手类，封装天气查询功能"""
    
    def __init__(self, model_provider: ModelProvider, servers: List[MCPServer]):
        """初始化天气助手
        
        Args:
            model_provider: 模型提供器实例
            servers: MCP服务器列表
        """
        self.model_provider = model_provider
        self.servers = servers
        self.agent = None
        self.response_handler = ResponseHandler()

    async def initialize(self) -> None:
        """初始化Agent，仅在首次使用时执行"""
        if self.agent:
            return
            
        print("正在初始化天气查询Agent...")
        self.agent = Agent(
            name="天气助手",
            instructions=(
                "你是一个专业的天气助手，可以帮助用户查询和分析天气信息。"
                "用户可能会询问天气状况、天气预报等信息，请根据用户的问题选择合适的工具进行查询。"
            ),
            mcp_servers=self.servers,
            model_settings=Config.MODEL_SETTINGS
        )

    def _create_run_config(self) -> RunConfig:
        """创建运行配置
        
        Returns:
            配置好的RunConfig实例
        """
        return RunConfig(
            model_provider=self.model_provider,
            trace_include_sensitive_data=True,
            handoff_input_filter=None
        )

    async def run_query(self, query: str, streaming: bool = True) -> str:
        """处理天气查询请求
        
        Args:
            query: 用户的自然语言查询
            streaming: 是否使用流式输出，默认为True
            
        Returns:
            查询结果的最终输出
        """
        # 确保初始化完成
        await self.initialize()
        print(f"\n正在处理查询：{query}\n")
        
        run_config = self._create_run_config()
        
        # 流式输出模式
        if streaming:
            result = Runner.run_streamed(
                self.agent,
                input=query,
                max_turns=Config.MAX_TURNS,
                run_config=run_config
            )
            await self.response_handler.handle_streaming_response(result)
        else:
            # 非流式模式
            print("使用非流式输出模式处理查询...")
            result = await Runner.run(
                self.agent,
                input=query,
                max_turns=Config.MAX_TURNS,
                run_config=run_config
            )
                        
        print("\n===== 完整天气信息 =====")
        print(result.final_output)
        return result.final_output

class WeatherApp:
    """天气应用类，负责整个应用的生命周期管理"""
    
    def __init__(self):
        """初始化天气应用"""
        self.model_provider = None
        self.servers = []
        self.assistant = None
        
    async def setup(self) -> None:
        """设置应用环境"""
        # 创建模型提供器
        self.model_provider = DeepSeekModelProvider()
        
        # 从配置文件加载MCP服务器配置
        try:
            with open(Config.MCP_CONFIG_PATH, 'r') as f:
                mcp_config = json.load(f)
                
            # 根据配置创建MCP服务器
            if 'mcpServers' in mcp_config:
                for server_name, server_config in mcp_config['mcpServers'].items():
                    server = None
                    
                    # 根据配置类型创建不同类型的服务器
                    if 'url' in server_config:
                        # 创建SSE服务器
                        server = await MCPServerManager.create_sse_server(
                            name=server_name,
                            url=server_config['url'],
                            cache_tools=False,
                            env=server_config.get('env')
                        )
                    elif 'command' in server_config and 'args' in server_config:
                        # 创建STDIO服务器
                        server = await MCPServerManager.create_stdio_server(
                            name=server_name,
                            command=server_config['command'],
                            args=server_config['args'],
                            cache_tools=False,
                            env=server_config.get('env')
                        )
                    
                    if server:
                        self.servers.append(server)
        except Exception as e:
            print(f"加载MCP配置文件时出错: {e}")
            traceback.print_exc()
        
        # 创建天气助手实例
        self.assistant = WeatherAssistant(self.model_provider, self.servers)
        await self.assistant.initialize()
    
    async def cleanup(self) -> None:
        """清理应用资源"""
        await MCPServerManager.cleanup_servers(self.servers)
        
    async def run_interactive(self) -> None:
        """运行交互式查询循环"""
        # 打印欢迎信息
        print("===== DeepSeek MCP 天气查询系统 =====")
        print("请输入自然语言查询，例如：")
        print(" - \"北京天气怎么样\"")
        print(" - \"查询上海未来5天天气预报\"")
        print("输入'quit'或'退出'结束程序")
        print("======================================\n")
        
        try:
            # 交互式查询循环
            while True:
                # 获取用户输入
                user_query = input("\n请输入您的天气查询(输入'quit'或'退出'结束程序): ").strip()

                # 检查退出条件
                if user_query.lower() in ["quit", "退出"]:
                    print("感谢使用DeepSeek MCP天气查询系统，再见！")
                    break
                
                # 验证输入
                if not user_query:
                    print("查询内容不能为空，请重新输入。")
                    continue
                
                # 处理查询（默认使用流式输出）
                await self.assistant.run_query(user_query, streaming=True)

        except KeyboardInterrupt:
            print("\n程序被用户中断，正在退出...")
        except Exception as e:
            print(f"程序运行时发生错误: {e}")
            traceback.print_exc()

async def main():
    """应用程序主入口"""
    app = WeatherApp()
    
    try:
        # 设置应用
        await app.setup()
        
        # 运行交互式查询循环
        await app.run_interactive()
    finally:
        # 退出前清理资源
        await app.cleanup()
        print("程序结束，所有资源已释放。")

# 程序入口
if __name__ == "__main__":
    asyncio.run(main())
        
    
    

