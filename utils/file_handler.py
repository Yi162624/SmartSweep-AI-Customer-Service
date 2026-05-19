"""
    文件处理器
"""
import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 获取md5的十六进制字符串
def get_file_md5_hex(filepath: str):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return

    if not os.path.isfile(filepath):    # 是否是文件
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()       # md5哈希计算对象

    chunk_size = 4096      # 4KB分片，避免文件过的爆内存
    try:
        # 必须是二进制
        with open(filepath, "rb") as f:
            # :=  赋值+返回赋值的结果
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)     # 把数据块喂给 MD5 计算器

        md5_hex = md5_obj.hexdigest()   # 计算md5值
        return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


# 从指定文件夹中筛选出指定的文件类型
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []
    if not os.path.isdir(path):     # 是否是文件夹
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")

    for f in os.listdir(path):      # 列出文件夹 path 里的所有文件
        # 检查字符串 f 是否以指定的后缀结尾（.txt  .pdf ...）
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)


def pdf_loader(filepath: str,passwd=None) -> list[Document]:
    # 创建一个PDF加载器，传入文件路径和密码
    # .load()：执行加载操作，解析PDF文件
    return PyPDFLoader(filepath,passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    # 创建文本加载器对象
    # 执行加载，读取整个文本文件内容
    return TextLoader(filepath,encoding='utf-8').load()


