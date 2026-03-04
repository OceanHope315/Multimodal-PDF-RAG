import fitz  # PyMuPDF
import os
import easyocr


def parse_multimodal_pdf(pdf_path, output_dir="extracted_data"):
    # 初始化 EasyOCR，支持中英文识别
    print("正在初始化 EasyOCR 引擎 (首次运行会自动下载模型)...")
    reader = easyocr.Reader(['ch_sim', 'en'])

    text_dir = os.path.join(output_dir, "texts")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    print(f"🚀 开始解析 PDF，总计 {len(doc)} 页...")

    for page_index in range(len(doc)):
        page = doc[page_index]

        # 渲染页面为图片
        zoom = 2
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_path = os.path.join(image_dir, f"page_{page_index + 1}.png")
        pix.save(img_path)

        try:
            # 执行识别 (detail=0 直接返回纯文本列表)
            result = reader.readtext(img_path, detail=0)
            page_text = "\n".join(result)

            # 保存为文本
            with open(os.path.join(text_dir, f"page_{page_index+1}.md"), "w", encoding="utf-8") as f:
                f.write(page_text)

            print(f"✅ 第 {page_index + 1} 页 OCR 提取完成")
        except Exception as e:
            print(f"❌ 第 {page_index + 1} 页识别失败: {e}")

    print(f"\n🎉 预处理完成！文本在 {text_dir}，图片在 {image_dir}")


if __name__ == "__main__":
    parse_multimodal_pdf(r".\data\jidaoA.pdf")
    # 更换文件后删除之前的 extracted_data 与 qdrant_db 文件夹以避免干扰。
    # parse_multimodal_pdf(r".\data\2022-2023A.pdf")
