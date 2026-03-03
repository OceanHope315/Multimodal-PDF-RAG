import os
import re
import qdrant_client
import dashscope
from typing import List
from dashscope import MultiModalEmbedding as DashScopeMM
from dashscope import MultiModalConversation
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageNode
from llama_index.core.embeddings import MultiModalEmbedding

# 禁用代理防止连接阿里云报错
os.environ['NO_PROXY'] = 'dashscope.aliyuncs.com'
API_KEY = ""

# --- 1. 自定义 Embedding 类 ---


class DashScopeCloudEmbedding(MultiModalEmbedding):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        dashscope.api_key = api_key

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = DashScopeMM.call(
            model="multimodal-embedding-v1", input=[{'text': text}])
        return resp.output['embeddings'][0]['embedding'] if resp.status_code == 200 else []

    def _get_image_embedding(self, img_file_path: str) -> List[float]:
        resp = DashScopeMM.call(model="multimodal-embedding-v1",
                                input=[{'image': os.path.abspath(img_file_path)}])
        return resp.output['embeddings'][0]['embedding'] if resp.status_code == 200 else []

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_query_embedding(
        self, query: str) -> List[float]: return self._get_query_embedding(query)

    async def _aget_text_embedding(
        self, text: str) -> List[float]: return self._get_text_embedding(text)
    async def _aget_image_embedding(
        self, img_file_path: str) -> List[float]: return self._get_image_embedding(img_file_path)

# --- 2. 辅助工具函数 ---


def cn_to_num(text: str) -> str:
    """将中文数字转换为阿拉伯数字，支持检索校准"""
    cn_num = {'一': '1', '二': '2', '三': '3', '四': '4',
              '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
    for k, v in cn_num.items():
        text = text.replace(k, v)
    return text


def get_best_images(query, results, top_n=2):
    """自动化校准：根据题号关键词强制重排图片"""
    query_norm = cn_to_num(query)
    match = re.search(r'(\d+)', query_norm)
    target_num = match.group(1) if match else None

    scored_paths = []
    for res in results:
        path = None
        if isinstance(res.node, ImageNode):
            path = os.path.abspath(res.node.image_path)
        else:
            # 从文本节点元数据关联图片路径
            source_file = res.node.metadata.get('file_name', '')
            if 'page_' in source_file:
                p_num = source_file.split('_')[1].split('.')[0]
                path = os.path.abspath(
                    f"./extracted_data/images/page_{p_num}.png")

        if path and path not in [p for p, s in scored_paths]:
            score = res.score or 0
            if target_num:
                page_id = os.path.basename(path).split('_')[1].split('.')[0]
                md_path = f"./extracted_data/texts/page_{page_id}.md"
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()

                        # 精准匹配题号标志
                        patterns = [
                            f"\n{target_num}", f" {target_num}.",
                            f"{target_num} (", f"{target_num}（",
                            f"第{target_num}题",
                            f"八、" if target_num == "8" else "___NON___"
                        ]
                        if any(p in text_content for p in patterns):
                            print(
                                f"🎯 关键匹配成功：在 page_{page_id} 发现了题号 {target_num}")
                            score += 50.0  # 给予极高权重置顶
                        elif target_num in text_content:
                            score += 2.0   # 普通包含数字稍微加分

            scored_paths.append((path, score))

    # 按校准后的分数排序
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in scored_paths[:top_n]]

# --- 3. 主逻辑 ---


def main():
    embed_model = DashScopeCloudEmbedding(api_key=API_KEY)

    # 连接 Qdrant
    try:
        client = qdrant_client.QdrantClient(path="./qdrant_db")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    index = MultiModalVectorStoreIndex.from_vector_store(
        vector_store=QdrantVectorStore(client=client, collection_name="texts"),
        image_store=QdrantVectorStore(client=client, collection_name="images"),
        image_embed_model=embed_model,
        embed_model=embed_model
    )

    # 用户输入
    query_str = input("请输入您的问题: ")
    if not query_str.strip():
        return

    print(f"🔍 正在检索与校准页面...")
    retriever = index.as_retriever(
        similarity_top_k=8, image_similarity_top_k=5)
    results = retriever.retrieve(query_str)

    # 获取重排后的图片
    final_imgs = get_best_images(query_str, results, top_n=2)

    if not final_imgs:
        print("⚠️ 未找到匹配图片，尝试第一页兜底")
        final_imgs = [os.path.abspath("./extracted_data/images/page_1.png")]

    # 调用 Qwen-VL
    print(f"🤖 正在分析图片: {[os.path.basename(p) for p in final_imgs]}")
    content_list = []
    for img_p in final_imgs:
        content_list.append({'image': f"file://{img_p}"})

    content_list.append({
        'text': f"你是一个专业的助教。请根据提供的试卷图片，精准找到并回答：{query_str}。如果图片中有多道题，请只回答要求的这一道，并给出详细的步骤。"
    })

    response = MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=[{'role': 'user', 'content': content_list}]
    )

    if response.status_code == 200:
        print("\n" + "="*25 + " AI 精准回答 " + "="*25)
        print(response.output.choices[0].message.content[0]['text'])
        print("="*59)
    else:
        print(f"❌ 模型调用失败: {response.message}")

    client.close()


if __name__ == "__main__":
    main()
