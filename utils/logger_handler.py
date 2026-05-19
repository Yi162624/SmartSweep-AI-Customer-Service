"""
    日志器
"""
from datetime import datetime
import logging
from utils.path_tool import get_abs_path
import os

# 日志保存的根目录（存放在哪里）
LOG_ROOT = get_abs_path("log")

# 确保日志的目录存在（创建日志文件夹）
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置 error info debug （定义日志的书写格式）
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
    name: str = "agent",                 # 日志器名字
    console_level: int = logging.INFO,   # 控制台输出的级别（默认INFO），显示重要信息
    file_level: int = logging.DEBUG,     # 文件输出的级别（默认DEBUG），记录全部信息
    log_file: str = None,                # 日志文件路径（默认None，不写文件）
) -> logging.Logger:                     # 返回值类型：logger对象
    logger = logging.getLogger(name)   # 获取或创建一个 logger 实例（名字是 name）
    logger.setLevel(logging.DEBUG)     # 设置 logger 的全局最低级别为 DEBUG

    """
    如果重复，则直接返回logger，下面的代码不执行，日志不会添加到文件和控制台中
    解决重复输出问题
    """
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler()      # 创建一个控制台输出的处理器
    console_handler.setLevel(console_level)        # 设置这个 handler 的日志级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)    # 日志的配置

    logger.addHandler(console_handler)    # 把控制台输出功能，安装到 logger 上，还并没有真正的添加信息

    # 文件Handler
    if not log_file:  # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    # 	Python 内置的文件日志处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)    # 设置级别
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)  # 设置输出格式

    logger.addHandler(file_handler)      # 	把处理器绑定到 logger 上，还并没有真正的添加信息

    return logger

# 快捷获取日志器（单例模式）
logger = get_logger()

if __name__ == "__main__":
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
