import fitz  # PyMuPDF

# 将单个 PDF 文件中的多个页面拼接为一个页面，并允许手动设置每个页面的位置和缩放比例
def concatenate_single_pdf_with_custom_positions_and_scales(pdf_path, positions, scales):
    # 打开 PDF 文档

    pdf_document = fitz.open(pdf_path)
    final_pdf = fitz.open()

    # 读取所有页面
    pages = [pdf_document.load_page(page_num) for page_num in range(len(pdf_document))]

    # 确保提供的位置和缩放因子的数量与页面数量一致
    if len(pages) != len(positions) or len(pages) != len(scales):
        raise ValueError("The number of positions and scales must match the number of pages in the PDF.")

    # 计算新画布的尺寸，以确保所有页面能够放下
    canvas_width = max([x + page.rect.width * scale_x for page, (x, y), (scale_x, scale_y) in zip(pages, positions, scales)])
    canvas_height = max([y + page.rect.height * scale_y for page, (x, y), (scale_x, scale_y) in zip(pages, positions, scales)])

    # 创建新的拼接页面（画布）
    new_page = final_pdf.new_page(width=canvas_width, height=canvas_height)

    # 将每个页面插入到拼接后的页面中，按照自定义的位置和缩放比例
    for page, (x, y), (scale_x, scale_y) in zip(pages, positions, scales):
        # 创建缩放矩阵
        matrix = fitz.Matrix(scale_x, scale_y)

        # 计算每个页面在缩放后的宽度和高度
        width = page.rect.width * scale_x
        height = page.rect.height * scale_y

        # 将页面按缩放比例插入到指定的位置
        new_page.show_pdf_page(fitz.Rect(x, y, x + width, y + height),
                               page.parent, page.number)

    pdf_document.close()
    return final_pdf

# PDF 文件路径
pdf_path = "cytometry figure binder 2.pdf"

# 自定义每个 PDF 页面在新 PDF 中的位置 (x, y) 坐标
positions = [
    (0, 0),         # 第一页的位置
    (200, -28),     # 第二页的位置
    (0, 170),         # 第一页的位置
    (150, 170),
    (300, 170)
]

# 自定义每个页面的缩放比例 (scale_x, scale_y)
scales = [
    (0.48, 0.48),     # 第一页的缩放比例 (不缩放)
    (0.45, 0.45),     # 第二页缩小一半
    (0.3, 0.3),  # 第一页的缩放比例 (不缩放)
    (0.3, 0.3),
    (0.3, 0.3)
]

# 拼接 PDF 页面并指定位置和缩放比例
final_pdf = concatenate_single_pdf_with_custom_positions_and_scales(pdf_path, positions, scales)

# 保存拼接后的 PDF
final_pdf.save("cytometry figure 2.pdf")
final_pdf.close()
