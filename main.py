import asyncio  # 异步IO支持
import os  # 系统操作
import traceback  # 异常堆栈追踪

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
from agents.mcp import MCPServerSse, MCPServerSseParams, MCPServer

# 环境变量配置
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.environ["API_KEY"]
BASE_URL = os.environ["BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]

# 禁用，否则会报错：Tracing: request failed
set_tracing_disabled(True)

class DeepSeekModelProvider(ModelProvider):
    """DeepSeek模型提供器，通过OpenAI兼容接口连接DeepSeek API"""
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    def get_model(self, model_name: str) -> Model:
        """获取模型实例
        
        Args:
            model_name: 模型名称，若为空则使用默认模型
            
        Returns:
            配置好的OpenAI兼容模型实例
        """
        return OpenAIChatCompletionsModel(model=model_name or MODEL_NAME, openai_client=self.client)

async def create_weather_server():
    """创建并连接MCP天气服务器
    
    Returns:
        已连接的MCP服务器实例
    """
    print("正在初始化DeepSeek-MCP天气服务器...")
    # 创建MCP服务器连接
    weather_server = MCPServerSse(
        name="weather_sse",
        params=MCPServerSseParams(
            url="http://127.0.0.1:8080/sse"
        ),
        cache_tools_list=False
    )

    # 连接到MCP服务器
    print("正在连接到MCP服务器...")
    await weather_server.connect()
    print("MCP服务器连接成功！")
    
    return weather_server

class WeatherAssistant:
    """天气助手类，封装天气查询功能，确保MCP服务器只初始化和连接一次"""
    
    def __init__(self, model_provider: ModelProvider, server_list: list[MCPServer]):
        """初始化天气助手
        
        Args:
            model_provider: 模型提供器实例
            server_list: MCP服务器列表
        """
        self.model_provider = model_provider
        self.server_list = server_list
        self.weather_agent = None   

    async def initialize(self):
        """初始化Agent，仅在首次使用时执行"""
        if self.weather_agent:
            return
            
        print("正在初始化DeepSeek-MCP天气查询agent...")

        # 创建天气助手Agent
        self.weather_agent = Agent(
            name="天气助手",
            instructions=(
                "你是一个专业的天气助手，可以帮助用户查询和分析天气信息。"
                "用户可能会询问天气状况、天气预报等信息，请根据用户的问题选择合适的工具进行查询。"
            ),
            mcp_servers=self.server_list,
            model_settings=ModelSettings(
                temperature=0.2,  # 较低温度提高回答精确性
                top_p=0.9,  # 保持适度词汇多样性
                max_tokens=4096,  # 输出长度限制
                tool_choice="auto",  # 自动选择最合适的工具
                parallel_tool_calls=True,  # 允许并行调用多个工具提高效率
                truncation="auto"  # 自动管理长文本
            )
        )

    async def run_query(self, query: str, streaming: bool = True) -> None:
        """处理天气查询请求
        
        Args:
            query: 用户的自然语言查询
            streaming: 是否使用流式输出，默认为True
        """
        # 确保初始化完成
        await self.initialize()

        print(f"\n正在处理查询：{query}\n")

        # 流式输出模式
        if streaming:
            # 流式调用大模型
            result = Runner.run_streamed(
                self.weather_agent,
                input=query,
                max_turns=10,  # 限制对话回合数
                run_config=RunConfig(
                    model_provider=self.model_provider,
                    trace_include_sensitive_data=True,
                    handoff_input_filter=None
                )
            )

            print("流式回复:", end="", flush=True)
            try:
                # 处理流式响应事件
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
        else:
            # 非流式模式
            print("使用非流式输出模式处理查询...")
            result = await Runner.run(
                self.weather_agent,
                input=query,
                max_turns=10,
                run_config=RunConfig(
                    model_provider=self.model_provider,
                    trace_include_sensitive_data=True,
                    handoff_input_filter=None
                )
            )
                        
        print("\n===== 完整天气信息 =====")
        print(result.final_output)

async def cleanup_servers(server_list: list[MCPServer]):
    """清理MCP服务器资源
    
    Args:
        server_list: 要清理的MCP服务器列表
    """
    for server in server_list:
        try:
            await server.cleanup()
            print(f"MCP服务器 {server.name} 资源清理成功！")
        except Exception as e:
            print(f"清理MCP服务器 {server.name} 资源时出错: {e}")
            traceback.print_exc()

async def main():
    """应用程序主入口 - 实现交互式天气查询循环"""
    # 打印欢迎信息
    print("===== DeepSeek MCP 天气查询系统 =====")
    print("请输入自然语言查询，例如：")
    print(" - \"北京天气怎么样\"")
    print(" - \"查询上海未来5天天气预报\"")
    print("输入'quit'或'退出'结束程序")
    print("======================================\n")

    # 创建模型提供器
    model_provider = DeepSeekModelProvider()
    
    # 创建并连接MCP服务器
    weather_server = await create_weather_server()
    server_list = [weather_server]
    
    # 创建天气助手实例
    assistant = WeatherAssistant(model_provider, server_list)
    
    try:
        # 初始化连接
        await assistant.initialize()
        
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
            
            # 默认使用流式输出
            streaming = True

            # 处理查询
            await assistant.run_query(user_query, streaming)

    except KeyboardInterrupt:
        print("\n程序被用户中断，正在退出...")
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        traceback.print_exc()
    finally:
        # 退出前清理资源
        await cleanup_servers(server_list)
        print("程序结束，所有资源已释放。")

# 程序入口
if __name__ == "__main__":
    asyncio.run(main())
        
    
    

