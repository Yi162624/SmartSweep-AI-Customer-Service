from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarsize,get_weather,get_user_location,get_user_id,
                                     get_current_month,fetch_external_data,fill_context_for_report)
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,                       # 模型
            system_prompt=load_system_prompts(),    # 提示词（这个提示词里面有一个工具会调用RAG但是不一定会使用）
            tools=[rag_summarsize,get_weather,get_user_location,get_user_id,         # 工具
                   get_current_month,fetch_external_data,fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],         # 中间件
        )

    def execute_stream(self,query: str):
        """
        流式执行Agent
        """
        # 将用户输入包装成 Agent 能理解的格式
        input_dict = {
            "messages":[
                {"role": "user","content": query}
            ]
        }

        # context就是上下文信息runtime中的信息，是否进行提示词切换的标记
        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
            latest_message = chunk["messages"][-1]            # 取最后一个（即新的信息）
            if latest_message.content:                        # 获取消息内容
                yield latest_message.content.strip() + "\n"   # 流式输出


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)