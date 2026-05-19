"""
streamlit run app.py
"""
from time import sleep
import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()     # 水平分割线

# st.session_state 是一个跨页面交互保持数据的记忆盒子，解决 Streamlit 每次刷新都会重置变量的问题。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# 遍历并显示所有历史聊天记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()


if prompt:
    # 在聊天界面右侧显示用户的消息（头像默认是"user"样式）
    st.chat_message("user").write(prompt)
    # 把用户的消息存入 session_state 的 message 列表中
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("智能客服思考中..."):
        # 流式输出
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)      # 保存到列表

                for chat in chunk:
                    sleep(0.01)
                    yield chat

        #  在屏幕上实时显示 AI 的回复（逐字输出）
        st.chat_message("assistant").write_stream(capture(res_stream,response_messages))
        # 把 AI 的完整回复保存到历史记录中
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()