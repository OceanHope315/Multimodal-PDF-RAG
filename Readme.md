# Multimodal-PDF-RAG: 基于 Qwen-VL 与 LlamaIndex 的多模态 PDF 问答系统 （半成品）

这是一个利用多模态大语言模型（MLLM）和向量检索技术，对复杂 PDF（如含有大量公式、图表的试卷）进行精准识别与回答的系统。

## 🌟 核心特性
- **多模态提取**：结合 PyMuPDF 渲染页面与 EasyOCR 文本识别，完整保留文档视觉信息。
- **自定义嵌入**：包装了阿里云 DashScope 的多模态 Embedding 接口，实现图文跨模态检索。
- **智能校准**：内置题号校准逻辑，通过正则匹配 OCR 文本，强力提升图片检索的准确率。
- **精准回答**：采用 `qwen-vl-plus` 模型，支持复杂数学公式解析（LaTeX 输出）。

## 🛠️ 技术栈
- **框架**: [LlamaIndex](https://www.llamaindex.ai/)
- **模型**: 阿里云通义千问 Qwen-VL (Vision-Language)
- **数据库**: Qdrant (本地模式)
- **OCR**: EasyOCR + PyMuPDF

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone [https://github.com/OceanHope315/Multimodal-PDF-RAG.git](https://github.com/OceanHope315/Multimodal-PDF-RAG.git)
cd Multimodal-PDF-RAG
```
### 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. 配置 API Key
在build_index和test(test0)脚本中替换你的阿里云 DashScope API Key：
```python
API_KEY = "你的_DASHSCOPE_API_KEY"
```
### 4.运行流程
解析 PDF: 提取图片与文本
```bash
python scripts/pdf.py
```
构建索引: 向量化并存入 Qdrant
```bash
python scripts/build_index.py # 如果无法生成qdrant_db，请手动运行
```
开始提问:
```bash
python scripts/test.py # 该测试仅上传图片
(python scripts/test0.py) # 该测试仅上传图片和文本块
```
## 输入题目示例
jidaoA: 请回答SECTION VI的题目 \
2022-2023A: 请回答第八题 \
示例输出的结果在output_example中