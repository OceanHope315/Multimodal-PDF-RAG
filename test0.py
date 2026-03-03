import os
import re
import time
import qdrant_client
import dashscope
from typing import List
from dashscope import MultiModalEmbedding as DashScopeMM
from dashscope import MultiModalConversation
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageNode
from llama_index.core.embeddings import MultiModalEmbedding

# --- 1. 基础配置 ---
os.environ['NO_PROXY'] = 'dashscope.aliyuncs.com'
API_KEY = "sk-71edcc7dce2e449f8539fe2d9edfae94"
MODEL_NAME = 'qwen-vl-plus'  # 若追求极致精度可更换为 'qwen-vl-max'

# --- 2. 自定义 Embedding 类 ---


# --- 1. 自定义 Embedding 类 (修复异步抽象方法问题) ---
class DashScopeCloudEmbedding(MultiModalEmbedding):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        dashscope.api_key = api_key

    # --- 同步方法 ---
    def _get_text_embedding(self, text: str) -> List[float]:
        resp = DashScopeMM.call(
            model="multimodal-embedding-v1", input=[{'text': text}])
        return resp.output['embeddings'][0]['embedding'] if resp.status_code == 200 else [0.0]*1024

    def _get_image_embedding(self, img_file_path: str) -> List[float]:
        resp = DashScopeMM.call(model="multimodal-embedding-v1",
                                input=[{'image': os.path.abspath(img_file_path)}])
        return resp.output['embeddings'][0]['embedding'] if resp.status_code == 200 else [0.0]*1024

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def get_text_embedding_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    # --- 异步方法 (必须实现，否则会报 TypeError) ---
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_image_embedding(self, img_file_path: str) -> List[float]:
        return self._get_image_embedding(img_file_path)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

# --- 3. 核心辅助工具 ---


def cn_to_num(text: str) -> str:
    """中文数字转阿拉伯数字"""
    cn_num = {'一': '1', '二': '2', '三': '3', '四': '4',
              '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'}
    for k, v in cn_num.items():
        text = text.replace(k, v)
    return text


def get_best_images(query, results, top_n=2):
    """自动化校准：根据题号关键词强制置顶图片"""
    query_norm = cn_to_num(query)
    match = re.search(r'(\d+)', query_norm)
    target_num = match.group(1) if match else None
    scored_paths = []

    for res in results:
        path = None
        if isinstance(res.node, ImageNode):
            path = os.path.abspath(res.node.image_path)
        else:
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
                        patterns = [
                            f"\n{target_num}. (",         # 匹配行首大题号
                            f" {target_num}. (",          # 匹配带空格的大题号
                            f"第{target_num}题",           # 匹配中文习惯
                        ]
                        if any(p in text_content for p in patterns):
                            print(
                                f"🎯 关键匹配：在 page_{page_id} 发现了题号 {target_num}")
                            score += 50.0
            scored_paths.append((path, score))
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    return [p for p, s in scored_paths[:top_n]]

# --- 4. 主程序逻辑 ---


def main():
    embed_model = DashScopeCloudEmbedding(api_key=API_KEY)

    try:
        client = qdrant_client.QdrantClient(path="./qdrant_db")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    # 加载多模态索引
    index = MultiModalVectorStoreIndex.from_vector_store(
        vector_store=QdrantVectorStore(client=client, collection_name="texts"),
        image_store=QdrantVectorStore(client=client, collection_name="images"),
        image_embed_model=embed_model, embed_model=embed_model
    )

    query_str = input("请输入您的问题: ")
    if not query_str.strip():
        return

    print(f"🔍 正在从数据库检索相关文本与图片...")
    retriever = index.as_retriever(
        similarity_top_k=5, image_similarity_top_k=3)
    results = retriever.retrieve(query_str)

    # 1. 提取 OCR 文本作为参考上下文
    context_texts = []
    for res in results:
        if not isinstance(res.node, ImageNode):
            context_texts.append(res.node.get_content())
    reference_ocr = "\n---\n".join(context_texts[:3])  # 取前3个相关文本片段

    # 2. 获取经过校准后的图片
    final_imgs = get_best_images(query_str, results, top_n=2)
    if not final_imgs:
        final_imgs = [os.path.abspath("./extracted_data/images/page_1.png")]

    print(f"🤖 正在调用 {MODEL_NAME} 进行图文混合推理...")

    # 3. 构造图文混合的 Prompt
    content_list = []
    for img_p in final_imgs:
        content_list.append({'image': f"file://{img_p}"})

    prompt = f"""
你是一个专业的数学助教。请结合提供的【试卷图片】视觉信息和以下【参考 OCR 文本】，回答用户问题。

【参考 OCR 文本】（可能存在识别误差）：
{reference_ocr}

【用户问题】：
{query_str}

【回答要求】：
1. 精准定位题目，如果图片中有多道题，请只回答要求的这一道。
2. 给出详细的解题步骤和计算过程。
3. 数学公式必须使用 LaTeX 格式书写。
"""
    content_list.append({'text': prompt})

    # 4. 请求模型
    response = MultiModalConversation.call(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': content_list}],
        request_timeout=120  # 增加超时时间防止卡住
    )

    if response.status_code == 200:
        print("\n" + "="*20 + " AI 精准回答 " + "="*20)
        print(response.output.choices[0].message.content[0]['text'])
        print("="*53)
    else:
        print(f"❌ 模型调用失败: {response.message}")

    client.close()


if __name__ == "__main__":
    main()
