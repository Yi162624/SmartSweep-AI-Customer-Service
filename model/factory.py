"""
模型工厂
"""
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from utils.config_handler import rag_conf

# ========== Ollama 模型（评估模式使用） ==========
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# ========== 原 DashScope 模型导入（保留备用） ==========
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_community.chat_models.tongyi import ChatTongyi


# 抽象类（无法创建实例）
class BaseModelFactory(ABC):
    @abstractmethod   # 抽象方法，子类必须重写
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

# ========== 聊天模型（Ollama 版本） ==========
# 原实现：ChatTongyi(model=rag_conf["chat_model_name"])
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 聊天模型 ChatOllama（本地 Ollama 服务）
        return ChatOllama(model=rag_conf["chat_model_name"])


# ========== Ollama 嵌入模型包装器 ==========
class OllamaEmbeddingsWrapper(Embeddings):
    """使用 Ollama 本地嵌入模型"""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        # 初始化 Ollama 嵌入模型
        self.embedding = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 构建知识库时，把切好的文档块批量转成向量存起来
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        # 检索时 ——把用户的问题转成向量，去 Milvus 里做相似度搜索
        return self.embedding.embed_query(text)


# ========== 嵌入模型工厂（Ollama 版本） ==========
# 原实现：CorrectV4Embeddings(model_name=rag_conf["embedding_model_name"])
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 使用 Ollama 本地嵌入模型
        return OllamaEmbeddingsWrapper(model_name=rag_conf["embedding_model_name"])



chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()