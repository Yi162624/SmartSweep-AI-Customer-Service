"""
    向量存储: 向量数据库，文本分割器
"""
import os
import pickle
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import txt_loader, pdf_loader, get_file_md5_hex, listdir_with_allowed_type
from utils.logger_handler import logger
from langchain_community.retrievers import BM25Retriever


class VectorStoreService:
    def __init__(self):
        # 向量数据库
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],    # Chroma 向量数据库的集合名称
            embedding_function=embed_model,        # 嵌入模型（文本 -> 向量）
            persist_directory=chroma_conf['persist_directory'],    # 向量数据在磁盘上保存的路径(运行原始数据源之后 得到的 向量数据之后保存到文件中)
        )

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


    def get_retriever(self):
        """
        把向量数据库包装成一个检索器:  k:返回k个值
        """
        # # 关键词检索器
        bm25_retriever = self.bm25_retriever()
        # # 向量检索器
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k":chroma_conf['k']})
        # 混合检索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.8, 0.2],  # 调整权重（BM25 0.8 + 向量 0.2）
            search_kwargs={"k": chroma_conf['k']}
        )
        return ensemble_retriever
        # return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf['k']})

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

                # 将内容存入向量库（自动创建文件）
                self.vector_store.add_documents(split_documents)
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

    def inspect_database(self):
        """
        查看向量数据库中存储的所有文档信息
        """
        try:
            # 获取所有文档的元数据和内容
            all_data = self.vector_store.get()

            # 打印基本信息
            print(f"📊 向量数据库统计:")
            print(f"  - 文档总数: {len(all_data['ids'])}")
            print(f"  - Collection名称: {chroma_conf['collection_name']}")
            print(f"  - 持久化路径: {chroma_conf['persist_directory']}")
            print("\n" + "=" * 60)

            # 按文件来源分组显示
            file_groups = {}
            for idx, (doc_id, metadata, document) in enumerate(
                    zip(all_data['ids'], all_data['metadatas'], all_data['documents'])):
                source = metadata.get('source', '未知来源')
                if source not in file_groups:
                    file_groups[source] = []
                file_groups[source].append({
                    'id': doc_id,
                    'content_preview': document[:100] + '...' if len(document) > 100 else document,
                    'full_metadata': metadata
                })

            # 打印分组信息
            for source, docs in file_groups.items():
                print(f"\n📁 文件: {source}")
                print(f"  文档块数量: {len(docs)}")
                print(f"  预览内容（前2个块）:")
                for i, doc in enumerate(docs[:2]):
                    print(f"    块{i + 1}: {doc['content_preview']}")
                print(f"  元数据示例: {docs[0]['full_metadata']}")
                print("-" * 40)

            return all_data

        except Exception as e:
            print(f"❌ 查询失败: {str(e)}")
            return None

if __name__ == "__main__":
    # vs = VectorStroreService()
    # vs.load_document()
    # retriever = vs.get_retriever()   # 检索器
    # res = retriever.invoke("迷路")
    # for r in res:
    #     print(r.page_content)    # 打印文档
    #     print("-"*20)

    # # 测试单独的 BM25 检索器
    # vs = VectorStroreService()
    # bm25_only = vs.get_retriever("bm25")
    # docs = bm25_only.invoke("水箱加水后漏水怎么处理？")
    # print("仅 BM25 检索结果:")
    # for doc in docs:
    #     print(f"- {doc.page_content[:100]}...")

    # 测试单独的 vector
    vs = VectorStoreService()
    vs.load_document()
    vector_only = vs.get_retriever()
    docs = vector_only.invoke("小户型适合什么扫地机器人？")
    print("仅 vector 检索结果:")
    for doc in docs:
        print(f"- {doc.page_content[:100]}...")

    # vs = VectorStroreService()
    # vs.inspect_database()

