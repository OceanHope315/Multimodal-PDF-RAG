from llama_index.core.schema import ImageNode
import os
import shutil
import qdrant_client
from qdrant_client.http import models
from typing import List
import dashscope
from dashscope import MultiModalEmbedding as DashScopeMM
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.node_parser import SentenceSplitter

# --- 1. 自定义云端 DashScope Embedding 类 ---
# 手动包装官方 SDK，规避 llama_index 内部导入错误


class DashScopeCloudEmbedding(MultiModalEmbedding):
    def __init__(self, API_KEY: str, **kwargs):
        super().__init__(**kwargs)
        dashscope.api_key = API_KEY

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = DashScopeMM.call(
            model="multimodal-embedding-v1",
            input=[{'text': text}]
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"DashScope Text Error: {resp.message}")

    def _get_image_embedding(self, img_file_path: str) -> List[float]:
        # 获取绝对路径，DashScope 接口要求本地文件路径需明确
        abs_path = os.path.abspath(img_file_path)
        resp = DashScopeMM.call(
            model="multimodal-embedding-v1",
            input=[{'image': abs_path}]
        )
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            raise Exception(f"DashScope Image Error: {resp.message}")

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    def get_image_embedding_batch(self, img_file_paths: List[str], **kwargs) -> List[List[float]]:
        return [self._get_image_embedding(f) for f in img_file_paths]

    async def _aget_query_embedding(
        self, query: str) -> List[float]: return self._get_query_embedding(query)

    async def _aget_text_embedding(
        self, text: str) -> List[float]: return self._get_text_embedding(text)

    async def _aget_image_embedding(
        self, img_file_path: str) -> List[float]: return self._get_image_embedding(img_file_path)

# --- 核心逻辑：多模态索引构建 ---


def run_build_index():
    # 1. 准备 Qdrant 客户端
    client = qdrant_client.QdrantClient(path="./qdrant_db")

    # 2. 重新创建集合 (确保维度是 1024)
    vector_params = models.VectorParams(
        size=1024, distance=models.Distance.COSINE)
    client.recreate_collection(
        collection_name="texts", vectors_config=vector_params)
    client.recreate_collection(
        collection_name="images", vectors_config=vector_params)

    # 3. 初始化多模态组件
    embed_model = DashScopeCloudEmbedding(
        API_KEY="")
    storage_context = StorageContext.from_defaults(
        vector_store=QdrantVectorStore(client=client, collection_name="texts"),
        image_store=QdrantVectorStore(client=client, collection_name="images")
    )

    # 4. 加载数据
    # OCR 出来的文本
    text_documents = SimpleDirectoryReader(
        "./extracted_data/texts").load_data()
    # 渲染出来的图片
    image_documents = SimpleDirectoryReader(
        "./extracted_data/images").load_data()

    # 5. 构建索引 (这一步会自动处理文本向量化)
    index = MultiModalVectorStoreIndex.from_documents(
        text_documents,
        storage_context=storage_context,
        embed_model=embed_model,
        image_embed_model=embed_model,
    )

    # 6. 【核心任务】将图片作为 ImageNode 显式插入
    # 只有这样，图片才会被 DashScope 多模态 Embedding 处理
    print(f"正在上传图片向量 (共 {len(image_documents)} 张)...")
    for img_doc in image_documents:
        # 使用 ImageNode 包装，确保进入 image_store
        node = ImageNode(
            image_path=img_doc.metadata.get('file_path'),
            metadata=img_doc.metadata
        )
        index.insert_nodes([node])

    print("🚀 恭喜！多模态索引构建完成。")
    client.close()


if __name__ == "__main__":
    run_build_index()
