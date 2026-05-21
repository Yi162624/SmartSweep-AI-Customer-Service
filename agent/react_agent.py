"""
ReAct Agent 智能体模块 - 支持多轮对话记忆
修改日期：2026-05-20
修改目的：改造为用户名体系，支持用户登录和长期记忆持久化
"""
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summariesze,get_weather,get_user_location,get_user_id,
                                     get_current_month,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from typing import Optional, List, Dict
from agent.memory.memory_service import load_user_info

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,                       # 模型
            system_prompt=load_system_prompts(),    # 提示词（包含工具调用说明）
            tools=[rag_summariesze,get_weather,get_user_location,get_user_id,         # 工具列表
                   get_current_month,fetch_external_data,fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],         # 中间件（监控/日志/提示词切换）
        )

    def execute_stream(self, query: str, history: Optional[List[Dict[str, str]]] = None, username: Optional[str] = None):
        """
        流式执行Agent，支持多轮对话记忆 + 长期记忆加载

        Args:
            query: 用户当前输入的问题
            history: 历史对话记录列表，格式为 [{"role": "user/assistant", "content": "消息内容"}, ...]
            username: 用户名，用于加载长期记忆
        """
        user_memory = {}       # 初始化用户记忆字典
        if username:
            user_memory = load_user_info(username)       # 加载用户记忆

        # 构建用户上下文注入消息（放在历史之后、当前问题之前）
        context_injection = []
        if user_memory.get("device_model"):
            # 如果用户记忆中包含产品型号
            context_injection.append({
                "role": "system",
                "content": f"【用户记忆】该用户使用的产品型号是：{user_memory['device_model']}，回答时可优先参考对应型号的信息。"
            })

        # 将历史消息、上下文注入、当前消息合并，构建完整的对话上下文
        recent_history = (history or [])[-10:]  # 保留最近10条（5轮对话），避免token超限
        # 合并历史消息、上下文注入、当前消息，构建完整的对话上下文
        messages = recent_history + context_injection + [{"role": "user", "content": query}]    

        # 包装成 Agent 能理解的输入格式
        input_dict = {
            "messages": messages 
        }

        # 流式执行Agent，context传递报告生成标记
        full_answer = ""  # 收集完整回答，用于返回保存信息
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]            # 获取最新生成的消息
            if latest_message.content:                        # 确保消息内容不为空
                content = latest_message.content.strip() + "\n"
                full_answer += latest_message.content.strip()
                yield content                                  # 逐段流式输出

        # 流式结束后返回保存所需的元数据（通过 return 语句）
        return {
            "username": username,
            "question": query,
            "answer": full_answer,
            "device_model": user_memory.get("device_model")
        }


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)
