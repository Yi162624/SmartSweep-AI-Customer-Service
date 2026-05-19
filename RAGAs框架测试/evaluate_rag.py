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
from RAG.rag_service import RagSummarizerService

# 问题
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

# 人工标注答案
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

# 总结提示词
answer = []

# RAG检索到的 contexts(检索文档)
contexts =[]

rag = RagSummarizerService()

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama的OpenAI兼容端点
    api_key="ollama",                      # Ollama不需要真实的API key，但必须提供
)


# 更稳定的初始化方式
custom_llm = llm_factory(
    model="qwen2.5:7b",
    client=client,
    temperature=0,     # 降低随机性，提高评估稳定性（评估模型设为0）
)

custom_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # 或者 "llama2" 等其他支持 embedding 的模型
    base_url="http://localhost:11434"
)

i = 0
for q in question:
    """
    获取answer,contexts
    """
    i += 1
    # 返回list[Document] 需要变回 str
    # 一次检索
    docs = rag.retrieve_docs(q,i)

    contexts_text = [doc.page_content for doc in docs]

    contexts.append(contexts_text)
    # 两次检索
    answer.append(rag.rag_summarize(q))

# 打印看看格式对不对
print("Questions:", question)
print("Contexts", contexts)
print("Answers:", answer)

dataset = Dataset.from_dict({
    "question": question,
    "contexts": contexts,
    "answer": answer,
    "ground_truth": ground_truth,
})

print(dataset)

# 一键式评估
result = evaluate(
    dataset,
    metrics=[
        # 忠实度（幻觉） （生成答案是否严格基于检索到的上下文，没有编造。）
        # 返回的数据或知识库的问题（返回一堆不相关的内容，检索质量问题）  LLM的问题
        faithfulness,
        # 答案相关性（需要embedding模型） 
        # 检索出来的文档问题（太少，检索质量差）  LLM的问题
        answer_relevancy, 
        # 上下文精度/精确率    （检索到的上下文中，有多少片段是真正与问题相关的。） 
        # 切分文档有问题，嵌入模型效果差，没有优化索引（重排序，混合索引）
        context_precision,
        # 上下文召回率  （真实答案中的信息有多少被检索到的上下文覆盖）
        # 知识库有问题，是否漏掉关键文档（重排序裁剪掉了，top_k太小），嵌入模型效果差
        context_recall, 
    ],
    llm=custom_llm,  # 为所有指标统一设置LLM
    embeddings=custom_embeddings,  # 只有当 metrics 需要 embedding 时才需要
)


# 查看结果
print(result)
df = result.to_pandas()
print(df)
