# 智扫通 — 扫地机器人智能客服

> 基于 RAG（检索增强生成）的扫地/扫拖机器人智能问答系统，支持实时天气查询、用户定位、个性化使用报告生成。

---

## 功能特性

- **智能问答** — 基于 RAG 技术，从知识库中检索扫地机器人相关文档并生成专业回答
- **天气查询** — 通过高德地图 API 获取实时天气，提供环境适配建议
- **位置定位** — 自动获取用户所在城市，提供本地化建议
- **个性化报告** — 根据用户历史使用记录生成使用情况报告与保养建议
- **混合检索** — 结合 BM25 关键词检索 + 向量语义检索 + 重排序，提升检索精度
- **Streamlit 界面** — 友好的 Web 交互界面，支持流式输出

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端框架 | Streamlit |
| AI 框架 | LangChain |
| 大语言模型 | Ollama（qwen2.5:7b） |
| 向量数据库 | ChromaDB |
| 嵌入模型 | nomic-embed-text（Ollama） |
| 重排序模型 | BAAI/bge-reranker-v2-m3 |
| 评估框架 | RAGAS |
| 地理信息 | 高德地图 API |

## 项目结构

```
扫地机器人问答助手/
├── app.py                    # Streamlit 入口（启动文件）
├── agent/
│   ├── react_agent.py        # Agent 智能体（ReAct 模式）
│   └── tools/
│       ├── agent_tools.py    # 工具函数（天气、位置、RAG等）
│       └── middleware.py     # 中间件（日志、提示词切换）
├── RAG/
│   ├── vector_store.py       # 向量存储与检索
│   └── rag_service.py        # RAG 总结服务
├── model/
│   └── factory.py            # 模型工厂（LLM + 嵌入模型）
├── utils/
│   ├── config_handler.py     # YAML 配置加载
│   ├── path_tool.py          # 路径工具
│   ├── file_handler.py       # 文件处理（txt/pdf 加载、MD5）
│   ├── logger_handler.py     # 日志工具
│   └── prompt_loader.py      # 提示词加载
├── config/
│   ├── rag.yml               # RAG 模型配置
│   ├── chroma.yml            # 向量数据库配置
│   ├── agent.yml             # Agent 工具配置
│   └── prompts.yml           # 提示词路径配置
├── prompts/
│   ├── main_prompt.txt       # 系统提示词
│   ├── rag_summarize.txt     # RAG 总结提示词
│   └── report_prompt.txt     # 报告生成提示词
├── data/                     # 知识库数据目录
│   ├── 扫地机器人100问.pdf
│   ├── 选购指南.txt
│   ├── 维护保养.txt
│   └── ...
└── RAGAs框架测试/
    └── evaluate_rag.py       # RAGAS 评估脚本
```

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.ai/)（本地运行 LLM）
- 高德地图 API Key（可选，用于天气/定位功能）

## 安装与使用

### 1. 克隆项目

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装并启动 Ollama

```bash
# 安装 Ollama（详见 https://ollama.ai/）
# 下载所需模型
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 4. 配置高德 API Key（可选）

编辑 `config/agent.yml`，将 `gaode_key` 设置为你的高德地图 API Key：

```yaml
gaode_key: "你的高德 API Key"
```

> 建议通过环境变量设置，避免硬编码：`$env:GAODE_KEY = "你的Key"`

### 5. 构建知识库（首次运行）

```bash
python -c "from RAG.vector_store import VectorStoreService; VectorStoreService().load_document()"
```

### 6. 启动 Web 界面

```bash
streamlit run app.py
```

### 7. 运行 RAGAS 评估（可选）

```bash
python RAGAs框架测试/evaluate_rag.py
```

## 使用示例

启动后，在浏览器中打开 Streamlit 界面，你可以问以下类型的问题：

- **产品咨询** — "小户型适合什么扫地机器人？"
- **故障排查** — "机器人拖地时突然停机怎么办？"
- **维护保养** — "每次清扫完成后要如何保养？"
- **环境适配** — "夏季地面有西瓜汁渍怎么清理？"
- **使用报告** — "帮我生成上个月的使用报告"
- **综合问题** — "根据我所在的城市天气，告诉我机器人怎么保养"

## 许可证

本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。
