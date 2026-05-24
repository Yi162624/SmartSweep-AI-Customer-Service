"""
    RAG 问答的"在线处理"环节
    原流程：检索 → 重排序 → 生成（无条件，检索不到也强行回答）
    重构后：使用多层次 RAG Pipeline（LangGraph）→ 评分门控 → 查询重写 → 二次检索 → 生成
    新增：检索质量评估、查询重写（Step-back/HyDE）、降级处理
"""
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from RAG.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model

# ===== v1.1 新增：引入 LangGraph RAG Pipeline =====
from RAG.rag_pipeline import run_rag_graph, _rerank_docs
from utils.logger_handler import logger


# 打印提示词
def print_prompt(prompt):
    logger.info(f"提示词：{prompt}")
    return prompt

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

class RagSummarizerService(object):
    def __init__(self):
        # 向量数据库（Milvus Lite 或 Docker Milvus）
        self.vector_store = VectorStoreService()
        # 检索器（使用混合检索模式：BM25关键词 + Milvus向量）
        self.retriever = self.vector_store.get_retriever(mode="hybrid")
        # 提示词文本
        self.prompt_text = load_rag_prompts()
        # 提示词模板
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 模型
        self.model = chat_model
        # 链
        self.chain = self._init_chain()

    # 链
    def _init_chain(self):
        # 初始化链
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # 得到检索文档
    # Document = 文本内容 + 元数据（描述信息）
    def retrieve_docs(self,query: str,i: int) -> list[Document]:
        logger.info(query)
        logger.info("开始检索")
        # 获取检索文档
        context_docs = self.retriever.invoke(query)
        # 给检索文档重排序
        context_docs = self.reranking(query, context_docs) 
        logger.info(f"检索文档 {i} ：{context_docs}")
        return context_docs

    def reranking(self, query, chunks, top_k=4):
        if not chunks:
            return chunks
        # 对文档块重排序
        reranking_chunks = _rerank_docs(query, chunks, top_k)
        scores = [doc.metadata.get("score", 0) for doc in reranking_chunks]
        logger.info(f"文档块重排序得分: {scores}")
        return reranking_chunks


    # ===== v1.1 重构：rag_summarize 接入多层次 RAG Pipeline =====
    # 原逻辑：retrieve_docs → 拼接context → chain.invoke（无条件生成）
    # 新逻辑：run_rag_graph（含评分门控+查询重写+二次检索）→ 拼接context → chain.invoke
    def rag_summarize(self,query: str) -> str:
        # 使用 RAG Pipeline 进行检索（含评分门控和查询重写）
        logger.info(f"用户问题：{query}")
        # 运行 RAG Pipeline
        pipeline_result = run_rag_graph(query)

        context_docs = pipeline_result.get("docs", [])         # 得到检索到的文档
        route = pipeline_result.get("route", "N/A")            # 得到 Pipeline 路由路径
        rewrite_strategy = pipeline_result.get("rewrite_strategy", "N/A")     # 如果有的话 查询重写策略

        logger.info(f"Pipeline 路由路径：{route}")
        if rewrite_strategy and rewrite_strategy != "N/A":         # 如果有查询重写策略
            logger.info(f"查询重写策略：{rewrite_strategy}")

        # 如果最终没有检索到任何文档，返回提示信息而非强行生成
        if not context_docs:
            logger.warning("知识库中未找到相关信息")
            return "抱歉，知识库中暂时没有找到与您问题相关的信息。请尝试换个问法，或联系客服获取帮助。"

        # 拼接上下文
        context = ""
        counter = 0
        for doc in context_docs:         # 遍历检索到的文档
            counter += 1
            context += f"[参考资料]{counter}:参考资料: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,       # 问题
                "context": context,   # 参考资料
            }
        )

if __name__ == "__main__":
    rag = RagSummarizerService()
    question = [
        "小户型适合什么扫地机器人"
    ]
    m = 0
    for n in question:
        m += 1
        logger.info(f"{m} :")
        logger.info(rag.rag_summarize(n))
