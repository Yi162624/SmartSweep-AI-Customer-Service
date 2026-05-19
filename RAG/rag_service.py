"""
    RAG 问答的"在线处理"环节
"""
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from RAG.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from utils.path_tool import get_abs_path
from utils.config_handler import chroma_conf
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from FlagEmbedding import FlagReranker


# 打印提示词
def print_prompt(prompt):
    print("-"*20)
    print(prompt)
    print("-"*20)
    return prompt

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像
local_model_path = get_abs_path(chroma_conf["reranker_model_path"])    # 重排序模型

class RagSummarizerService(object):
    def __init__(self):
        # 向量数据库
        self.vector_store = VectorStoreService()
        # 检索器
        self.retriever = self.vector_store.get_retriever()
        # 提示词文本
        self.prompt_text = load_rag_prompts()
        # 提示词模板
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 模型
        self.model = chat_model
        # 链
        self.chain = self._init_chain()
        # 重排序模型（缓存，避免每次调用重复加载）
        self._reranker = FlagReranker(local_model_path, use_fp16=True)

    # 链
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # 得到检索文档
    # Document = 文本内容 + 元数据（描述信息）
    def retrieve_docs(self,query: str,i: int) -> list[Document]:
        print("-"*20)
        print(query)
        print("-"*20)
        print("开始检索")
        # 获取检索文档
        context_docs = self.retriever.invoke(query)
        # 给检索文档重排序
        context_docs = self.reranking(query, context_docs) 
        print(f"检索文档 {i} ：{context_docs}")
        return context_docs

    def reranking(self, query, chunks, top_k=4):
        # 空列表直接返回
        if not chunks:
            return chunks
        # chunks 是 Document 对象列表，需要转为 page_content 字符串
        input_pairs = [[query, chunk.page_content] for chunk in chunks]
        # 计算每个 chunk 与 query 的语义相似性得分
        scores = self._reranker.compute_score(input_pairs, normalize=True)
        print("文档块重排序得分:", scores)
        # 对得分进行排序并获取排名前 top_k 的 chunks
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranking_chunks = [chunks[i] for i in sorted_indices[:top_k]]
        # 打印前 top_k 个 score 对应的文档块
        for i in range(len(reranking_chunks)):
            print(f"重排序文档块{i + 1}: 相似度得分: {scores[sorted_indices[i]]}, 文档块信息: {reranking_chunks[i]}\n")
        return reranking_chunks


    # 总结提示词
    def rag_summarize(self,query: str) -> str:
        # 检索文档（返回list[Document]）
        context_docs = self.retrieve_docs(query,0)
        print(f"用户问题：{query}")
        print(f"总结提示词：{context_docs}")
        # 重排序（已由 retrieve_docs() 执行，此处避免双重重排序）
        # context_docs = self.reranking(query, context_docs)

        context = ""
        counter = 0
        for doc in context_docs:
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
        # "机器人拖地时突然停机怎么办？",
        # "机器人充电时拖地模组仍在工作怎么处理？",
        # "一次性拖布可以水洗后重复使用吗？",
        # "夏季地面有西瓜汁、冰淇淋渍怎么清理？",
        # "如何自定义扫拖的路线？",
        # "拖地时拖布支架摩擦地面，产生异响怎么办？",
        # "主刷滚刷仓内壁磨损，漏灰怎么办？",
        # "机器人无法恢复出厂设置，按钮无反应？",
        # "充电时指示灯闪烁，无法正常充电？",
        # "每次清扫完成后要如何保养？",
        # "普通家庭多久跟换一次主刷？",
        # "清水箱容量是多少？",
        # "机器人想要存放于车库、储物间等阴暗环境要如何存放",
        # "机器人清扫路线混乱，无规律",
        # "强力模式有什么用，适合处理什么",
        # "边刷的作用是什么？",
        # "拖布有哪些类型",
        # "扫地机器人真的省钱吗？",
        # "机器人工作时万向轮不转，影响转向",
        # "如何查询扫拖一体机器人的原装配件型号？",
        # "扫拖同时进行时吸力会降低吗？",
        "小户型适合什么扫地机器人"
    ]
    m = 0
    for n in question:
        m += 1
        print(f"{m} :")
        print(rag.rag_summarize(n))


