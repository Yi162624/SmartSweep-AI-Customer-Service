"""
    yaml
    k: v
"""

import yaml
from utils.path_tool  import get_abs_path


def load_rag_config(config_path: str=get_abs_path('config/rag.yml'),encoding: str='utf-8'):
    """
    取出rag的配置
    """
    with open(config_path, "r",encoding=encoding) as f:
        # 以 FullLoader 的方式把 YAML 的文本转化为 Python 对象
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(config_path: str=get_abs_path('config/chroma.yml'),encoding: str='utf-8'):
    """
    取出chroma的配置
    """
    with open(config_path, "r" ,encoding=encoding) as f:
        # 以 FullLoader 的方式把 YAML 的文本转化为 Python 对象
        return yaml.load(f,Loader=yaml.FullLoader)

def load_prompts_config(config_path: str=get_abs_path('config/prompts.yml'),encoding: str='utf-8'):
    """
    取出prompts的配置
    """
    with open(config_path, "r" ,encoding=encoding) as f:
        # 以 FullLoader 的方式把 YAML 的文本转化为 Python 对象
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(config_path: str=get_abs_path('config/agent.yml'),encoding: str='utf-8'):
    """
    取出agent的配置
    """
    with open(config_path, "r" ,encoding=encoding) as f:
        # 以 FullLoader 的方式把 YAML 的文本转化为 Python 对象
        return yaml.load(f,Loader=yaml.FullLoader)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
