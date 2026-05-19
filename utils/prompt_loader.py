"""
    提示词加载工具
"""
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    """
    读取系统提示词（System Prompt）的完整内容：定义 AI 的基本人设和回答规范。
    """
    try:
        # 获取系统提示词文件的绝对路径
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        # 重新抛出以捕获的异常
        raise e

    try:
        # 以只读的方式打开文件， .read()是读取整个文件的内容
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        return e


def load_rag_prompts():
    """
    加载RAG总结提示词：当用户提问时，指导 AI 如何利用检索到的资料回答问题，防止胡编乱造。
    """
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有main_prompt_path配置项")
        # 重新抛出以捕获的异常
        raise e

    try:
        # 打开文件
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"[load_rag_prompts]RAG总结提示词出错，{str(e)}")
        return e


def load_report_prompts():
    """
    加载报告生成提示词：当需要生成正式报告时，告诉 AI 应该按什么结构输出。
    """
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有main_prompt_path配置项")
        # 重新抛出以捕获的异常
        raise e

    try:
        # 打开文件
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except FileNotFoundError as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        return e
