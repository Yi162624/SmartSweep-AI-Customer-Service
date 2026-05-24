"""
    向量存储: 向量数据库，文本分割器
"""
import os
import pickle
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, get_file_md5_hex, listdir_with_allowed_type
from utils.logger_handler import logger
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class VectorStoreService:
    def __init__(self):
        # 连接 Milvus 数据库（获取配置文件中的参数）
        milvus_config = chroma_conf.get('milvus', {})
        self.host = milvus_config.get('host', 'localhost')
        self.port = str(milvus_config.get('port', '19530'))
        self.collection_name = milvus_config.get('collection_name', 'rag_embeddings')
        self.metric_type = milvus_config.get('metric_type', 'IP')
        self.index_type = milvus_config.get('index_type', 'HNSW')
        self.dim = milvus_config.get('dim', 768)
        self.nlist = milvus_config.get('nlist', 1024)
        self.nprobe = milvus_config.get('nprobe', 16)

        # Milvus Lite 模式检查（用于区分 Docker Milvus 和 Milvus Lite）
        use_milvus_lite = milvus_config.get('use_milvus_lite', False)

        if use_milvus_lite:
            # Milvus Lite 模式：启动嵌入式服务器
            try:
                # milvus-lite 2.x 方式：单独导入 default_server
                from milvus import default_server
                default_server.start(host=self.host, port=int(self.port))
                logger.info(f"[Milvus Lite 2.x] 已启动嵌入式服务器 {self.host}:{self.port}")
                # 再通过 pymilvus 连接
                connections.connect("default", host=self.host, port=self.port)
            except ImportError:
                # milvus-lite 3.x 方式：通过 server_manager 启动嵌入式服务器
                from milvus_lite.server_manager import server_manager_instance
                db_path = get_abs_path(chroma_conf['persist_directory'])
                os.makedirs(db_path, exist_ok=True)
                lite_uri = server_manager_instance.start_and_get_uri(db_path)
                connections.connect("default", uri=lite_uri)
                logger.info(f"[Milvus Lite 3.x] 已启动嵌入式服务器: {lite_uri}")
        else:
            # Docker Milvus 模式：直接连接到 Docker Milvus 服务器
            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"[Milvus Docker] 已连接到 {self.host}:{self.port}")

        # 获取或创建集合
        self.collection = self._get_or_create_collection()

        # 文本分割器（递归字符文本分割器）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf['chunk_size'],           # 每一个文本块的最大长度
            chunk_overlap=chroma_conf['chunk_overlap'],     # 可重复的最大字符数
            separators=chroma_conf['separators'],           # 分隔符
            length_function=len,
        )

        # 存储bm25的文档模块（顺序有可能有问题）
        self.bm25_documents = []

        # 尝试加载已经存在的bm25文件
        self._load_bm25_data()


    def _get_or_create_collection(self):
        """
        获取或创建 Milvus 集合
        """
        # 如果集合已存在，直接返回
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)    # 创建一个操作入口
            collection.load()                                # 把集合的数据加载到内存中
            logger.info(f"[Milvus] 已加载现有集合: {self.collection_name}")        # 记录加载日志
            return collection     # 返回加载后的集合

        # 创建新的集合
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        ]
        # 创建集合模式（图纸）
        schema = CollectionSchema(fields, description="RAG 知识库向量集合")
        # 创建集合（实际建立起来的集合）
        collection = Collection(name=self.collection_name, schema=schema)

        # 创建 HNSW（高效的向量近邻搜索算法） 索引
        index_params = {
            "index_type": self.index_type,         # 索引类型
            "metric_type": self.metric_type,      # 距离度量类型
            "params": {"M": 8, "efConstruction": 64}  # HNSW算法参数
        }
        # embedding 字段创建索引
        collection.create_index(field_name="embedding", index_params=index_params)
        # 把索引加载到内存中
        collection.load()
        logger.info(f"[Milvus] 已创建新集合: {self.collection_name}，维度: {self.dim}")
        return collection     # 返回创建后的集合


    def bm25_retriever(self):
        """
        获取关键词检索器
        """
        if not self.bm25_documents:
            logger.warning("BM25文档为空，尝试重新加载")
            self._load_bm25_data()

        if not self.bm25_documents:
            logger.error("BM25文档仍然为空，无法创建检索器")
            # 返回一个空的检索器，避免程序崩溃
            return BM25Retriever.from_documents([], search_kwargs={"k": chroma_conf['k']})

        # 返回关键词检索器
        return BM25Retriever.from_documents(self.bm25_documents,search_kwargs={"k":chroma_conf['k']})


    def _save_bm25_data(self,documents):
        """
        保存bm25到磁盘
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(get_abs_path(chroma_conf["bm25_path"])), exist_ok=True)

        with open(get_abs_path(chroma_conf["bm25_path"]),"wb") as f:
            pickle.dump(documents, f)
        logger.info(f"[加载知识库] bm25数据已存入 {chroma_conf['bm25_path']}，共 {len(documents)} 个文档")


    def _load_bm25_data(self):
        """
        从磁盘中读取bm25
        """
        if not os.path.exists(get_abs_path(chroma_conf["bm25_path"])):
            logger.warning(f"[加载知识库] bm25数据文件不存在")
            return None

        try:
            with open(get_abs_path(chroma_conf["bm25_path"]),"rb") as f:
                # 读取出来后还原成 python 对象
                self.bm25_documents = pickle.load(f)
                logger.info(f"[加载知识库] bm25数据已从文件中加载，共 {len(self.bm25_documents)} 个文档")
                return True
        except Exception as e:
            logger.info(f"[加载知识库] bm25数据加载失败，原因：{str(e)}")
            return False


    def get_retriever(self, mode="hybrid"):
        """
        把向量数据库包装成一个检索器
        mode: 检索模式
            "hybrid" - BM25关键词 + Milvus向量混合检索（默认）
            "vector" - 仅向量检索
            "bm25" - 仅关键词检索
        """
        if mode == "vector":
            # 仅向量检索模式
            return MilvusRetriever(collection=self.collection, k=chroma_conf['k'], nprobe=self.nprobe)

        if mode == "bm25":
            # 仅关键词检索模式
            return self.bm25_retriever()

        # 默认混合检索模式（BM25 + Milvus 向量）
        bm25_retriever = self.bm25_retriever()
        vector_retriever = MilvusRetriever(collection=self.collection, k=chroma_conf['k'], nprobe=self.nprobe)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.7, 0.3],  # 调整权重（BM25 0.7 + 向量 0.3）
            search_kwargs={"k": chroma_conf['k']}
        )
        return ensemble_retriever

    def get_file_documents(self,read_path: str):
        """
        得到文件内容
        """
        if read_path.endswith(".txt"):  # 结尾是不是.txt
            return txt_loader(read_path)

        if read_path.endswith(".pdf"):
            return pdf_loader(read_path)
        return []


    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入的向量库中
        要计算文件的MD5做去重
        """

        def check_md5_hex(md5_for_check: str):
            """
            检查文件是否重复
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w" ,encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():       # 读取全部行
                    line = line.strip()          # 出去开头和结尾的空白字符
                    if line == md5_for_check:
                        return True              # 处理过了
                return False

        def save_md5_hex(md5_for_check: str):
            # 把已经处理的文件md5存入文件中
            with open(get_abs_path(chroma_conf["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        # 获取符合要求的原始数据源
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),             # 读取原始数据源（文件夹路径）
            tuple(chroma_conf["allow_knowledge_file_type"])     # 允许的文件后缀
        )

        for path in allowed_files_path:
            # 获取文件的md5值
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):    # 检查是否重复
                logger.info(f"[加载知识库]{path}内容已经存在在知识库中，跳过")
                continue     # 跳过当前循环

            try:
                documents: list[Document] = self.get_file_documents(path)
                if not documents:
                    # 文件里面没有内容
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 文本分割器（先每个文件切分在存入向量库）
                split_documents: list[Document] = self.spliter.split_documents(documents)

                if not split_documents:
                    # 判断分割后是否有有效内容（过度防御）
                    # 例：有可能原本文件里面是图片
                    logger.warning(f"[加载知识库]{path}分割后没有有效文本内容，跳过")
                    continue

                # 将内容存入 Milvus 向量库
                self._add_documents_to_milvus(split_documents)
                # 把bm25数据保存到全局列表中
                self.bm25_documents.extend(split_documents)


                # 记录这个已经处理的文件的md5，避免下次重复
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库] {path} 内容加载完成")
            except Exception as e:
                # exc_info为True会记录详细的报错，如果False仅记录报错信息本身
                logger.error(f"[加载数据库] {path} 加载失败：{str(e)}", exc_info=True)
                continue

            # 在循环结束后保存BM25数据
            if self.bm25_documents:
                self._save_bm25_data(self.bm25_documents)
                logger.info(f"BM25数据保存完成，共 {len(self.bm25_documents)} 个文档块")
            else:
                logger.warning("没有新文档添加到BM25")

    def _add_documents_to_milvus(self, documents: list[Document]):
        """
        将文档批量转换为向量并添加到 Milvus 向量数据库
        """
        texts = [doc.page_content for doc in documents]       # 提取文档内容
        sources = [doc.metadata.get('source', '') for doc in documents]  # 提取文档来源（文件路径）

        # 生成向量嵌入
        embeddings = embed_model.embed_documents(texts)

        # 准备数据
        data = [
            embeddings,     # 向量嵌入
            texts,          # 文档内容
            sources         # 文档来源（文件路径）
        ]

        # 插入 Milvus
        self.collection.insert(data)
        self.collection.flush()       # 刷新数据到磁盘
        logger.info(f"[Milvus] 已插入 {len(documents)} 个文档向量")

    def inspect_database(self):
        """
        查看向量数据库中存储的所有文档信息
        """
        try:
            # 查询所有数据
            results = self.collection.query(expr="id >= 0", output_fields=["id", "text", "source"])

            # 打印基本信息
            print(f"📊 Milvus 向量数据库统计:")
            print(f"  - 文档总数: {len(results)}")
            print(f"  - Collection名称: {self.collection_name}")
            print(f"  - Milvus连接: {self.host}:{self.port}")
            print("\n" + "=" * 60)

            # 按文件来源分组显示
            file_groups = {}
            for item in results:
                source = item.get('source', '未知来源')
                if source not in file_groups:
                    file_groups[source] = []
                file_groups[source].append({
                    'id': item.get('id'),
                    'content_preview': item['text'][:100] + '...' if len(item['text']) > 100 else item['text'],
                })

            # 打印分组信息
            for source, docs in file_groups.items():
                print(f"\n📁 文件: {source}")
                print(f"  文档块数量: {len(docs)}")
                print(f"  预览内容（前2个块）:")
                for i, doc in enumerate(docs[:2]):
                    print(f"    块{i + 1}: {doc['content_preview']}")
                print("-" * 40)

            return results

        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            return None


class MilvusRetriever(BaseRetriever):
    """
    封装 Milvus 向量检索逻辑，为上层提供统一的检索接口
    用于从 Milvus 集合中检索相似的文档
    """
    collection: Collection                # Milvus 集合对象
    k: int = 5                            # 返回的最相似文档数量
    nprobe: int = 16                      # 查询时搜索的聚类数量（精度与速度的权衡）


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """
        根据用户的查询文本，在 Milvus 向量数据库中检索 语义最相似 的文档
        query: 查询文本
        返回: Document 对象列表
        """
        # 生成查询文本的向量嵌入
        query_embedding = embed_model.embed_query(query)

        # 在 Milvus 中搜索
        search_params = {
            "metric_type": chroma_conf.get('milvus', {}).get('metric_type', 'IP'),  # 搜索指标类型（IP或L2）
            "params": {"nprobe": self.nprobe}     # 查询时搜索的聚类数量（精度与速度的权衡）
        }

        # 执行搜索（PyMilvus 库）
        results = self.collection.search(
            data=[query_embedding],     # 查询向量嵌入
            anns_field="embedding",     # 检索字段
            param=search_params,        # 搜索参数
            limit=self.k,               # 返回的最相似文档数量
            output_fields=["text", "source"]     # 除了找到内容，还要把找到的内容给我
        )

        # 将结果转换为 LangChain 的 Document 对象列表
        documents = []
        for hits in results:        # 遍历搜索结果
            for hit in hits:        # 遍历每个文档的搜索结果
                text = hit.entity.get("text", "")          # 提取文档内容
                source = hit.entity.get("source", "")      # 提取文档来源（文件路径）
                score = hit.score                          # 提取文档相似度分数
                documents.append(Document(
                    page_content=text,                     # 文档内容
                    metadata={"source": source, "score": score}  # 文档来源（文件路径）和相似度分数
                ))

        return documents


if __name__ == "__main__":
    vs = VectorStroreService()
    vs.load_document()
    retriever = vs.get_retriever()   # 检索器
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)    # 打印文档
        print("-"*20)
