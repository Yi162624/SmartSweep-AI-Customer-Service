"""
    基于 LangGraph RAG 架构的评估脚本
    使用新的多层次 RAG Pipeline（评分门控 + Step-back/HyDE 查询重写 + 二次检索）进行评估
    注意：contexts 和 answers 来自同一次 Pipeline 调用，确保 RAGAS 评估数据一致性
"""
from datasets import Dataset
from langchain_community.embeddings import OllamaEmbeddings
from ragas import evaluate
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入新的 LangGraph RAG Pipeline
from RAG.rag_pipeline import run_rag_graph
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model


# 测试问题
question = [
    "机器人拖地时突然停机怎么办？",
    "机器人充电时拖地模组仍在工作怎么处理？",
    "一次性拖布可以水洗后重复使用吗？",
    "夏季地面有西瓜汁、冰淇淋渍怎么清理？",
    "如何自定义扫拖的路线？",
    "普通家庭多久跟换一次主刷？",
    "拖布有哪些类型",
    "强力模式有什么用，适合处理什么",
    "充电时指示灯闪烁，无法正常充电？",
    "每次清扫完成后要如何保养？",
    "清水箱容量是多少？",
    "机器人想要存放于车库、储物间等阴暗环境要如何存放？",
    "扫拖一体机器人可以只扫地不拖地吗？",
    "电动拖布不旋转怎么办？",
    "水箱长期不用如何保养？"
]

# 人工标注答案（ground truth）
ground_truth = [
    "检查剩余电量，电量不足会优先回充；电量充足则检查拖地模组是否卡顿，传感器是否被遮挡，清理后重试。",
    "重启机器人，确认是否因系统卡顿导致功能异常，若仍故障，在APP中恢复出厂设置，重新设置清扫参数。",
    "不建议，一次性拖布的吸水性和清洁性为单次设计，水洗后会变硬、掉毛，影响拖地效果，还可能堵塞拖地模组。",
    "立即用定点拖扫模式清理，加注清水配合少量果蔬清洁剂，拖完后用干拖模式吸干水分，避免污渍风干后难以清理。",
    " 支持“自定义路线”的机型，在APP中绘制扫拖路径，机器人将按绘制的路线进行清洁，适合重点区域的精准扫拖。",
    "普通家庭3-6个月更换，出现磨损、变形、开裂立即更换。",
    "一次性、可水洗、电动旋转拖布等多种选择。",
    "强力模式吸力提升50%以上，适合处理顽固污渍和深层灰尘。",
    "用干布擦拭充电触点，更换适配电源适配器，电池损坏则更换。",
    "每次清扫完成后，立即清理主刷上的毛发并清理尘盒，用干布擦拭机身和充电触点。",
    "清水箱≥300ml、污水箱≥250ml适配大户型，小户型可选择200ml左右容量，减少机身重量。",
    r"存放前将机器人充满电至80%-90%，全面清理机身和配件，排空水箱水分，放在阴凉干燥处存放。",
    "可以，在配套APP中进入清扫模式，关闭拖地功能，仅保留吸尘模式即可单独扫地。",
    "检查拖布是否被毛发、杂物缠绕，确认拖布电机卡扣是否安装到位，清理后重启机器人，仍不转联系售后检测电机。",
    "将水箱内剩余清水排空，用干布擦拭水箱内部和密封圈，放在阴凉干燥处存放，避免密封圈老化。"
]

# 初始化 Ollama 客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama的OpenAI兼容端点
    api_key="ollama",                      # Ollama不需要真实的API key，但必须提供
)

# 配置评估用的 LLM 和 Embeddings
custom_llm = llm_factory(
    model="qwen2.5:7b",
    client=client,
    temperature=0,     # 降低随机性，提高评估稳定性
)

custom_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 初始化 RAG 总结链（只初始化一次，避免重复加载提示词）
rag_prompt_text = load_rag_prompts()
rag_prompt_template = PromptTemplate.from_template(rag_prompt_text)
rag_chain = rag_prompt_template | chat_model | StrOutputParser()

# 存储评估数据
answers = []
contexts = []

# 遍历所有问题，使用新的 LangGraph RAG Pipeline
logger.info("========== 开始使用 LangGraph RAG 架构进行评估 ==========")
for idx, q in enumerate(question, 1):
    logger.info(f"\n【问题 {idx}/{len(question)}】{q}")
    
    # 只跑一次 Pipeline！（contexts 和 answer 都用同一批检索结果）
    pipeline_result = run_rag_graph(q)
    docs = pipeline_result.get("docs", [])
    route = pipeline_result.get("route", "N/A")
    rewrite_strategy = pipeline_result.get("rewrite_strategy", "N/A")
    
    logger.info(f"  - Pipeline 路由: {route}")
    if rewrite_strategy != "N/A":
        logger.info(f"  - 查询重写策略: {rewrite_strategy}")
    logger.info(f"  - 检索到文档数: {len(docs)}")
    
    # 提取上下文文本（用于 RAGAS 评估）
    context_texts = [doc.page_content for doc in docs]
    contexts.append(context_texts)
    
    # 如果没有检索到文档，返回提示信息
    if not docs:
        answer = "抱歉，知识库中暂时没有找到与您问题相关的信息。请尝试换个问法，或联系客服获取帮助。"
        answers.append(answer)
        logger.info(f"  - 生成答案: {answer[:100]}...")
        continue
    
    # 用同一次 Pipeline 的检索结果拼接上下文，生成答案（不再调 rag_summarize）
    context_str = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        context_str += f"[参考资料{i}]：{doc.page_content} | 参考元数据：{source}\n"
    
    answer = rag_chain.invoke({"input": q, "context": context_str})
    answers.append(answer)
    
    logger.info(f"  - 生成答案: {answer[:100]}...")

# 打印格式验证
logger.info("\n========== 评估数据格式验证 ==========")
logger.info(f"问题数: {len(question)}")
logger.info(f"答案数: {len(answers)}")
logger.info(f"上下文数: {len(contexts)}")

# 创建评估数据集
dataset = Dataset.from_dict({
    "question": question,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": ground_truth,
})

logger.info("\n========== 开始 RAGAS 评估 ==========")
result = evaluate(
    dataset,
    metrics=[
        faithfulness,        # 忠实度：答案是否基于检索上下文
        answer_relevancy,    # 答案相关性：答案与问题的相关程度
        context_precision,   # 上下文精度：检索到的上下文有多少与问题相关
        context_recall,      # 上下文召回率：真实答案的信息有多少被检索到
    ],
    llm=custom_llm,
    embeddings=custom_embeddings,
)

# 输出评估结果
logger.info("\n========== 评估结果 ==========")
print(result)
df = result.to_pandas()
print(df)

# # 计算综合得分
# faithfulness_score = result['faithfulness']
# answer_relevancy_score = result['answer_relevancy']
# context_precision_score = result['context_precision']
# context_recall_score = result['context_recall']

# # 加权综合得分（与之前评估报告一致的权重）
# overall_score = (context_recall_score * 0.3 + 
#                  context_precision_score * 0.2 + 
#                  faithfulness_score * 0.3 + 
#                  answer_relevancy_score * 0.2)

# logger.info(f"\n【综合得分】: {overall_score:.4f}")
# logger.info(f"  - context_recall: {context_recall_score:.4f} (权重 0.3)")
# logger.info(f"  - context_precision: {context_precision_score:.4f} (权重 0.2)")
# logger.info(f"  - faithfulness: {faithfulness_score:.4f} (权重 0.3)")
# logger.info(f"  - answer_relevancy: {answer_relevancy_score:.4f} (权重 0.2)")

# # 保存结果到 Markdown 文件
# result_file = "LangGraph_RAG_评估报告.md"
# with open(result_file, 'w', encoding='utf-8') as f:
#     f.write("# LangGraph RAG 架构评估报告\n\n")
#     f.write(f"**评估时间：** 2026-05-28\n\n")
#     f.write("---\n\n")
#     f.write("## 当前配置\n\n")
#     f.write("| 参数 | 值 |\n")
#     f.write("|------|:---:|\n")
#     f.write("| 架构 | LangGraph 多层次 RAG Pipeline |\n")
#     f.write("| 特性 | 评分门控 + Step-back/HyDE 查询重写 + 二次检索 |\n")
#     f.write("| 检索方式 | BM25 0.7 + 向量 0.3 |\n")
#     f.write("| chunk_overlap | 40 |\n")
#     f.write("| 评估 LLM | qwen2.5:7b（Ollama） |\n")
#     f.write("| 评估 Embeddings | nomic-embed-text（Ollama） |\n")
#     f.write("| 评估问题数 | 15 题 |\n\n")
#     f.write("## 评估指标结果\n\n")
#     f.write("| 指标 | 结果 |\n")
#     f.write("|------|:----:|\n")
#     f.write(f"| faithfulness（忠实度） | **{faithfulness_score:.4f}** |\n")
#     f.write(f"| answer_relevancy（答案相关性） | **{answer_relevancy_score:.4f}** |\n")
#     f.write(f"| context_precision（上下文精度） | **{context_precision_score:.4f}** |\n")
#     f.write(f"| context_recall（上下文召回率） | **{context_recall_score:.4f}** |\n")
#     f.write(f"| **综合得分**（加权） | **{overall_score:.4f}** |\n\n")
#     f.write("> 综合得分 = recall×0.3 + precision×0.2 + faithfulness×0.3 + relevancy×0.2\n\n")
#     f.write("---\n\n")
#     f.write("## 详细结果\n\n")
#     f.write(df.to_string())

# logger.info(f"\n评估结果已保存到: {result_file}")
