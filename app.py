"""
智扫通机器人智能客服 - Streamlit 前端入口
修改日期：2026-05-20
修改目的：集成用户登录体系，支持用户注册、登录、登出功能
"""
from time import sleep
import streamlit as st
from agent.react_agent import ReactAgent
from agent.memory.memory_service import append_history
from utils.auth import login as auth_login, register as auth_register
from utils.database import DatabaseInitializer

# 设置页面配置
st.set_page_config(page_title="智扫通机器人智能客服", page_icon="🤖")


def init_database():
    """初始化数据库"""
    try:
        DatabaseInitializer.init_database()   # 初始化数据库
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")


def show_login_page():
    """显示登录/注册页面"""
    st.title("🤖 智扫通机器人智能客服")
    st.divider()   # 分隔线

    tab1, tab2 = st.tabs(["登录", "注册"])   # 登录/注册标签页

    # 登录标签页
    with tab1:
        st.subheader("用户登录")   # 登录标题
        login_username = st.text_input("用户名", key="login_username")   # 登录用户名输入框
        login_password = st.text_input("密码", type="password", key="login_password")   # 登录密码输入框

        if st.button("登录", type="primary", use_container_width=True):   # 登录按钮
            if not login_username or not login_password:
                st.error("请输入用户名和密码")
            else:
                # 判断用户信息是否正确
                success, msg = auth_login(login_username, login_password)
                if success:
                    # 登录成功，写入 session 状态
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = login_username    # 记录当前谁在的登录
                    st.session_state["message"] = []    # 登录成功后，清空消息记录
                    st.success("登录成功！")
                    st.rerun()    # 登录成功后，重新运行整个脚本
                else:
                    st.error(msg)

    # 注册标签页
    with tab2:
        st.subheader("用户注册")   # 注册标题
        reg_username = st.text_input("用户名（至少3个字符）", key="reg_username")     # 注册用户名输入框
        reg_password = st.text_input("密码（至少6个字符）", type="password", key="reg_password")     # 注册密码输入框
        reg_password_confirm = st.text_input("确认密码", type="password", key="reg_password_confirm")     # 注册确认密码输入框

        if st.button("注册", type="primary", use_container_width=True):     # 注册按钮
            # 校验输入合法性
            if not reg_username or not reg_password:
                st.error("请填写所有字段")
            elif len(reg_username) < 3:
                st.error("用户名至少需要2个字符")
            elif len(reg_password) < 6:
                st.error("密码至少需要6个字符")
            elif reg_password != reg_password_confirm:
                st.error("两次输入的密码不一致")
            else:
                success, msg = auth_register(reg_username, reg_password)    # 注册用户
                if success:
                    st.success("注册成功！请登录")
                    st.session_state["show_success"] = True      
                    st.rerun()
                else:
                    st.error(msg)


def show_chat_page():
    """显示聊天主页面"""
    st.title(f"🤖 智扫通机器人智能客服 - 欢迎，{st.session_state['username']}")

    # 登出按钮
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("退出登录", use_container_width=True)  :    # 退出登录按钮
            st.session_state["logged_in"] = False              # 退出登录后，清空登录状态
            st.session_state["username"] = None                # 退出登录后，清空用户名
            st.session_state["message"] = []                   # 退出登录后，清空消息记录
            st.rerun()

    st.divider()       # 分隔线

    # 展示历史对话记录
    for message in st.session_state["message"]:       # 遍历消息记录
        # message["role"] 决定气泡位置   .write(message["content"]) 写出文字
        st.chat_message(message["role"]).write(message["content"])

    prompt = st.chat_input("请输入您的问题...")     # 用户输入框

    # 用户输入了问题
    if prompt:
        st.chat_message("user").write(prompt)       # 显示到屏幕
        st.session_state["message"].append({"role": "user", "content": prompt})    # 保存用户输入的问题到 session_state

        response_messages = []
        with st.spinner("智能客服思考中..."):
            history = st.session_state["message"][:-1]      # 取出用户输入的问题之前的所有消息记录
            username = st.session_state["username"]         # 取出当前登录的用户名

            # 调用 Agent 流式生成回答
            res_stream = st.session_state["agent"].execute_stream(
                prompt,             # 用户输入的问题
                history=history,    # 历史对话记录
                username=username   # 当前登录的用户名
            )

            # 收集流式输出并逐字显示
            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)     # 加入缓存列表
                    for chat in chunk:
                        sleep(0.01)
                        yield chat       # 逐字显示

            # chat_message("assistant")：创建ai气泡，流式显示AI回答
            st.chat_message("assistant").write_stream(capture(res_stream, response_messages))

            # 保存 AI 回答到历史记录
            st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})

            # 持久化保存到数据库
            if response_messages and response_messages[-1]:
                # 保存用户输入的问题和 AI 回答到数据库
                append_history(username, prompt, response_messages[-1])

        st.rerun()


def main():
    """应用入口：初始化数据库和 session，然后根据登录状态显示对应页面"""
    init_database()

    # 初始化 Agent
    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()

    # 初始化登录状态
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False


    if st.session_state["logged_in"]:     # 如果用户已登录
        show_chat_page()     # 显示聊天主页面
    else:
        show_login_page()    # 显示登录页面


if __name__ == "__main__":
    main()
