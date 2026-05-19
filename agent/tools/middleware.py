from typing import Callable
from utils.prompt_loader import load_system_prompts,load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from utils.logger_handler import logger

@wrap_tool_call       # 使函数能够拦截和处理工具调用
def monitor_tool(
        request: ToolCallRequest,       # 请求的数据封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command],       # 执行的函数本身（实际上这个去执行任务）
) -> ToolMessage | Command:
    """
    工具执行的监控
    """
    logger.info(f"[tool monitor]执行工具：{request.tool_call["name"]}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call["args"]}")

    try:
        # 执行函数
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            # 当工具名称为 fill_context_for_report 时，运行时上下文中标记结果为 True
            request.runtime.context["result"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model          # 在模型执行之前自动调用
def log_before_model(
        state: AgentState,    # 整个Agent智能体中的状态记录（包含所有消息的中间结果）
        runtime: Runtime,     # 记录了整个执行过程的上下文
):
    """
    在模型执行前输出日志
    """
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt        # 每次调用生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):
    """
    动态切换提示词
    """
    # 在文件夹中找到 request 标签，返回它的值
    is_report = request.runtime.context.get("report",False)
    if is_report:         # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    # 普通提示词
    return load_system_prompts()

