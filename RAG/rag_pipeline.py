"""
    基于 LangGraph 的多层次 RAG Pipeline
    流程：初次检索 → 评分门控 → (通过→直接生成) / (不通过→查询重写→二次检索→生成)
    参考 SuperMew 的 rag_pipeline.py 设计，使用 StateGraph 实现 DAG 编排
"""
from typing import Literal, Optional, List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from RAG.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import load_scoring_prompt, load_rewrite_prompt
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from FlagEmbedding import FlagReranker

# 重排序模型：优先使用本地路径，不存在则用 HuggingFace 模型名
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_local_model_path = get_abs_path(chroma_conf["reranker_model_path"])
if os.path.isdir(os.path.normpath(_local_model_path)):
    RERANKER_MODEL = os.path.normpath(_local_model_path)


# ========== Pydantic Schema（数据模板）（用于 with_structured_output） ==========

class GradeDocuments(BaseModel):
    """文档相关性评分：判断检索结果是否足以回答用户问题"""
    binary_score: str = Field(
        description="相关性评分：'yes' 表示相关通过，'no' 表示不相关需重写"
    )

class RewriteOutput(BaseModel):
    """查询重写输出：包含 Step-back 退步问题和 HyDE 假设文档"""
    step_back_question: str = Field(
        description="退步问题（Step-back）：将原始问题泛化为更通用的问题"
    )
    hypothetical_doc: str = Field(
        description="假设性文档（HyDE）：模拟的知识库文档片段"
    )


# ========== RAG State（流水线状态）（LangGraph 状态定义） ==========

class RAGState(TypedDict):
    """Pipeline 各阶段共享的状态"""
    question: str                       # 原始用户问题
    query: str                          # 当前查询（初始=原始问题，重写后=扩展查询）
    docs: List[Document]                # 检索到的文档列表
    context: str                        # 格式化后的上下文文本（供生成使用）
    route: Optional[str]                # 路由决策：generate_answer / rewrite_question
    rewrite_strategy: Optional[str]     # 重写策略（step_back_hyde / fallback）
    expanded_query: Optional[str]       # 扩展后的查询
    step_back_question: Optional[str]   # Step-back 退步问题
    hypothetical_doc: Optional[str]     # HyDE 假设文档


# ========== 工具函数 ==========

def _format_docs(docs: List[Document]) -> str:
    """
    将检索到的 Document 列表 格式化 为带有来源标记的字符串上下文
    """
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")        # 文档来源（原始文档或扩展文档）
        score = doc.metadata.get("score", "N/A")              # 文档相关性评分（FlagReranker 输出）
        text = doc.page_content                               # 文档内容（原始文本或扩展文本）
        chunks.append(f"[{i}] 来源：{source}（相关性：{score}）\n{text}")
    return "\n\n---\n\n".join(chunks)     # 格式化后的上下文文本（每个文档隔文档


def _rerank_docs(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """
    使用 FlagReranker 对文档进行语义重排序，返回前 top_k 个
    """
    if not docs:
        return docs
    reranker = _get_reranker()          # 获取重排序模型
    input_pairs = [[query, doc.page_content] for doc in docs]          # 输入对（查询 + 文档内容）
    scores = reranker.compute_score(input_pairs, normalize=True)       # 计算文档相关性评分（FlagReranker 输出）
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)      # 按评分排序（高到低）
    top_k = min(top_k, len(docs))           # 确保 top_k 不超过文档数量
    # 取出前 top_k 个文档的索引
    reranked = []
    for i in sorted_indices[:top_k]:             # 取前 top_k 个文档
        doc = docs[i]                            # 从原始文档列表中获取文档
        doc.metadata["score"] = float(scores[i])  # 把相关性评分也加入到文档元数据中
        reranked.append(doc)
    return reranked


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """
    基于 page_content 前 100 个字符对文档去重
    """
    seen = set()      # 用于存储已见的文档内容前 100 个字符（作为去重键）
    unique = []
    for doc in docs:
        key = doc.page_content[:100]        # 取文档内容前 100 个字符用来判断是否重复
        if key not in seen:
            seen.add(key)                     # 把去重键加入到已见集合中
            unique.append(doc)                # 把文档加入到去重列表中
    return unique


# ========== 全局单例（避免重复连接 Milvus 和加载模型） ==========
_vector_store = None
_reranker = None

def _get_vector_store():
    """获取 VectorStoreService 单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store

def _get_reranker():
    """获取 FlagReranker 单例（避免每次调用重复加载模型）"""
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


# ========== Pipeline 节点函数 ==========

def retrieve_initial(state: RAGState) -> dict:
    """
    【节点1】初次检索：使用混合检索（BM25关键词 + Milvus向量）获取候选文档
    检索后使用 FlagReranker 进行语义重排序，保留 top 4
    """
    logger.info(f"【阶段1-初次检索】查询：{state['question']}")

    vector_store = _get_vector_store()     # 获取向量存储服务
    retriever = vector_store.get_retriever(mode="hybrid")  # # 获取混合检索器（BM25关键词 + Milvus向量）

    # 执行混合检索
    docs = retriever.invoke(state["question"])
    logger.info(f"检索到 {len(docs)} 个候选文档块")

    # 重排序（保留 top 4）
    docs = _rerank_docs(state["question"], docs, top_k=4)
    logger.info(f"重排序后保留 {len(docs)} 个文档块")

    # 格式化为上下文文本
    context = _format_docs(docs)

    return {
        "query": state["question"],      # 原始查询
        "docs": docs,                    # 厙选文档块（重排序后 top 4）
        "context": context,              # 格式化后的上下文文本（每个文档隔文档）
    }


def grade_documents(state: RAGState) -> dict:
    """
    【节点2】文档相关性评分门控：使用 LLM（with_structured_output）判断检索结果是否足够
    评分通过（yes）→ 直接生成回答
    评分不通过（no）→ 触发查询重写
    LLM调用失败时降级处理：默认放行，避免系统崩溃
    """
    logger.info("【阶段2-评分门控】评估检索文档相关性...")

    # 如果根本没有检索到文档，直接走重写路径
    if not state.get("docs") or not state.get("context"):
        logger.warning("未检索到有效文档，触发查询重写")
        return {"route": "rewrite_question"}     # 触发查询重写节点

    try:
        # 加载评分提示词，判断检索到的文档是否足够回答用户问题
        scoring_prompt_text = load_scoring_prompt()
        # 把评分规则、用户问题和检索到的文档拼在一起，形成给 AI 的完整指令
        scoring_prompt = PromptTemplate.from_template(scoring_prompt_text)
        prompt_value = scoring_prompt.invoke({
            "question": state["question"],      # 原始查询（用户问题）
            "context": state["context"],        # 格式化后的上下文文本（每个文档隔文档）
        })

        # 使用 with_structured_output 让 LLM 返回结构化评分
        # 返回 GradeDocuments 对象
        grader = chat_model.with_structured_output(GradeDocuments)
        result = grader.invoke([{"role": "user", "content": prompt_value.text}])

        # result.binary_score：取出评分结果（yes/no）
        # .lower()：转换为小写
        score = (result.binary_score or "").strip().lower()
        logger.info(f"评分结果：{score}")

        if score == "yes":
            logger.info("文档相关性通过，直接生成回答")
            return {"route": "generate_answer"}
        else:
            logger.warning("文档相关性不足，触发查询重写")
            return {"route": "rewrite_question"}

    except Exception as e:
        # 降级处理：LLM 评分失败时默认放行，不阻塞回答
        logger.error(f"评分过程出错（{str(e)}），降级处理：直接生成回答")
        return {"route": "generate_answer"}


def rewrite_question(state: RAGState) -> dict:
    """
    【节点3】查询重写: 使用 Step-back + HyDE 双重策略扩展原始查询
    Step-back: 生成退步问题（更通用），用于检索背景知识
    HyDE: 生成假设文档，用于检索潜在相关内容
    LLM调用失败时回退到原始查询
    """
    logger.info("【阶段3-查询重写】扩展原始查询...")

    # 从状态中获取原始查询
    question = state["question"]

    try:
        # 加载重写提示词并构造 prompt
        rewrite_prompt_text = load_rewrite_prompt()
        rewrite_prompt = PromptTemplate.from_template(rewrite_prompt_text)
        prompt_value = rewrite_prompt.invoke({"question": question})

        # 使用 with_structured_output 让 LLM 指定格式回答
        rewriter = chat_model.with_structured_output(RewriteOutput)
        result = rewriter.invoke([{"role": "user", "content": prompt_value.text}])

        # result.step_back_question：退步问题（更通用），用于检索背景知识
        # result.hypothetical_doc：假设文档，用于检索潜在相关内容
        step_back_q = (result.step_back_question or "").strip()
        hypo_doc = (result.hypothetical_doc or "").strip()

        logger.info(f"退步问题（Step-back）：{step_back_q}")
        logger.info(f"假设文档（HyDE）：{hypo_doc[:80]}...")

        # 如果重写结果为空，回退到原始查询
        expanded_query = step_back_q or question

        return {
            "rewrite_strategy": "step_back_hyde",   # 用了什么重写方法
            "expanded_query": expanded_query,        # 扩展后的搜索词
            "step_back_question": step_back_q,       # 退步问题
            "hypothetical_doc": hypo_doc,           # 假设文档
        }

    except Exception as e:
        # 降级处理：重写失败时回退到原始查询
        logger.error(f"查询重写出错（{str(e)}），回退到原始查询")
        return {
            "rewrite_strategy": "fallback",
            "expanded_query": question,
            "step_back_question": "",
            "hypothetical_doc": "",
        }


def retrieve_expanded(state: RAGState) -> dict:
    """
    【节点4】二次检索：使用重写后的查询重新检索
    使用 Step-back 退步问题和 HyDE 假设文档分别检索，合并结果后去重 + 重排序
    如果重写结果为空，回退到原始问题检索
    """
    logger.info("【阶段4-二次检索】使用扩展查询重新检索...")

    vector_store = _get_vector_store()    # 获取向量数据库
    retriever = vector_store.get_retriever(mode="hybrid")    # 获取混合检索器

    all_docs = []

    # 0. 保留初次检索的文档，评分不通过也加入候选集参与重排序
    if state.get("docs"):
        logger.info(f"保留初次检索结果：{len(state['docs'])} 个文档块")
        all_docs.extend(state["docs"])

    # 1. 使用 Step-back 退步问题检索
    step_back_q = state.get("step_back_question", "")
    if step_back_q:
        logger.info(f"Step-back 检索：{step_back_q}")
        # 检索退步问题
        docs_step = retriever.invoke(step_back_q)
        logger.info(f"  → 检索到 {len(docs_step)} 个文档块")
        # 合并退步问题检索结果
        all_docs.extend(docs_step)

    # 2. 使用 HyDE 假设文档检索
    hypo_doc = state.get("hypothetical_doc", "")
    if hypo_doc:
        logger.info(f"HyDE 检索：{hypo_doc[:50]}...")
        # 检索假设文档
        docs_hyde = retriever.invoke(hypo_doc)
        logger.info(f"  → 检索到 {len(docs_hyde)} 个文档块")
        all_docs.extend(docs_hyde)

    # 3. 如果均无结果，回退到原始问题检索
    if not all_docs:
        logger.warning("扩展检索无结果，回退到原始问题检索")
        all_docs = retriever.invoke(state["question"])

    # 去重
    unique_docs = _dedup_docs(all_docs)
    logger.info(f"去重后共 {len(unique_docs)} 个文档块")

    # 重排序（会合并）
    unique_docs = _rerank_docs(state["question"], unique_docs, top_k=4)
    logger.info(f"重排序后保留 {len(unique_docs)} 个文档块")

    # 格式化为上下文文本
    context = _format_docs(unique_docs)

    return {
        "docs": unique_docs,      # 合并后的去重重排序文档
        "context": context,       # 格式化后的上下文文本
    }


# ========== 条件路由函数 ==========

def route_after_grade(state: RAGState) -> Literal["generate_answer", "rewrite_question"]:
    """
    评分门控后的路由决策
    根据 grade_documents 节点设置的 route 字段决定走哪条路径
    """
    return state.get("route", "generate_answer")


# ========== 构建 LangGraph ==========

def build_rag_graph():
    """
    构建 RAG Pipeline 的 LangGraph 有向无环图（DAG）
    节点：retrieve_initial → grade_documents → (generate_answer | rewrite_question → retrieve_expanded)
    """
    # ===== 步骤1：初始化图，指定任务单（RAGState）格式 =====
    graph = StateGraph(RAGState)

    # ===== 步骤2：添加4个节点（流水线工位） =====
    graph.add_node("retrieve_initial", retrieve_initial)     # 工位1：初次检索
    graph.add_node("grade_documents", grade_documents)       # 工位2：评分门控
    graph.add_node("rewrite_question", rewrite_question)     # 工位3：查询重写
    graph.add_node("retrieve_expanded", retrieve_expanded)   # 工位4：二次检索

    # ===== 步骤3：设置入口，指定流水线起点 =====
    graph.set_entry_point("retrieve_initial")

    # ===== 步骤4：固定边（没有岔路的固定路线） =====
    graph.add_edge("retrieve_initial", "grade_documents")   # 工位1 → 工位2（固定）

    # ===== 步骤5：条件边（岔路，根据评分结果走不同方向） =====
    graph.add_conditional_edges(
        "grade_documents",        # 从工位2（评分门控）出发
        route_after_grade,        # 看路标函数
        {                         # 路标映射表
            "generate_answer": END,                    # 返回 "generate_answer" → 直接结束
            "rewrite_question": "rewrite_question",    # 返回 "rewrite_question" → 去工位3（重写）
        },
    )

    # ===== 步骤4续：更多固定边 =====
    graph.add_edge("rewrite_question", "retrieve_expanded")  # 工位3 → 工位4（固定）
    graph.add_edge("retrieve_expanded", END)                 # 工位4 → 结束

    # ===== 步骤6：编译图，让蓝图变成可执行的流水线 =====
    return graph.compile()


# 编译好的图（全局单例，避免重复构建）
rag_graph = build_rag_graph()


# ========== 对外接口 ==========

def run_rag_graph(question: str) -> dict:
    """
    运行 RAG Pipeline 的对外接口
    输入：用户问题字符串
    输出：包含 docs、context、route 等字段的状态字典
    """
    initial_state = {
        "question": question,     # 原始问题
        "query": question,        # 初始查询
        "docs": [],
        "context": "",
        "route": None,
        "rewrite_strategy": None,
        "expanded_query": None,
        "step_back_question": None,
        "hypothetical_doc": None,
    }

    # 执行流水线
    result = rag_graph.invoke(initial_state)
    return result


if __name__ == "__main__":
    # 简单的功能测试
    test_questions = [
        "小户型适合什么扫地机器人",
        "机器人拖地时突然停机怎么办？",
    ]
    for q in test_questions:
        logger.info(f"测试问题：{q}")
        result = run_rag_graph(q)
        route = result.get("route", "N/A")
        doc_count = len(result.get("docs", []))
        strategy = result.get("rewrite_strategy", "N/A")
        logger.info(f"路由路径：{route} | 文档数量：{doc_count} | 重写策略：{strategy}")
