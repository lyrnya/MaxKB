# coding=utf-8
"""
@project: maxkb
@Author：虎
@file： text_split_handle.py
@date：2024/3/27 18:19
@desc:
"""

import os
import re
import tempfile
import time
import traceback
from typing import List

import pdfplumber
from django.utils.translation import gettext_lazy as _
from pypdf import PdfReader
from pypdf.generic import Destination

from common.handle.base_split_handle import BaseSplitHandle
from common.handle.impl.text.ocr_handle import ocr_image
from common.utils.heading import MAX_HEADING_LEVEL, detect_heading_level
from common.utils.logger import maxkb_logger
from common.utils.split_model import SplitModel, smart_split_paragraph

# 页面文字量低于该阈值且含图片时,判定为扫描页/图片页,触发 OCR
OCR_SPARSE_TEXT_THRESHOLD = 30

# PDF 提取出的是视觉行,正文行占满整行宽度,标题行明显更短。取最长行宽的该比例作为标题行宽上限
HEADING_LINE_WIDTH_RATIO = 0.7

# markdown 表格骨架和标题标记,统计页面真实文字量时需要剔除
MARKDOWN_SCAFFOLD_PATTERN = re.compile(r"[|\-#\s]")

# pdfplumber 默认的固定字距阈值会在中英文/数字交界处插入多余空格,改成按字号比例判断
X_TOLERANCE_RATIO = 0.3

default_pattern_list = [
    re.compile("(?<=^)# .*|(?<=\\n)# .*"),
    re.compile("(?<=\\n)(?<!#)## (?!#).*|(?<=^)(?<!#)## (?!#).*"),
    re.compile("(?<=\\n)(?<!#)### (?!#).*|(?<=^)(?<!#)### (?!#).*"),
    re.compile("(?<=\\n)(?<!#)#### (?!#).*|(?<=^)(?<!#)#### (?!#).*"),
    re.compile("(?<=\\n)(?<!#)##### (?!#).*|(?<=^)(?<!#)##### (?!#).*"),
    re.compile("(?<=\\n)(?<!#)###### (?!#).*|(?<=^)(?<!#)###### (?!#).*"),
    re.compile("(?<!\n)\n\n+"),
]


def check_links_in_pdf(doc):
    for page in doc.pages:
        if PdfSplitHandle.get_internal_links(doc, page):
            return True
    return False


def get_pdf_object(value):
    if hasattr(value, "get_object"):
        return value.get_object()
    return value


class PdfSplitHandle(BaseSplitHandle):
    def handle(
        self,
        file,
        pattern_list: List,
        with_filter: bool,
        limit: int,
        get_buffer,
        save_image,
    ):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # 将上传的文件保存到临时文件中
            for chunk in file.chunks():
                temp_file.write(chunk)
            # 获取临时文件的路径
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as pdf_file:
                pdf_document = PdfReader(pdf_file)
                if type(limit) is str:
                    limit = int(limit)
                if type(with_filter) is str:
                    with_filter = with_filter.lower() == "true"
                # 标题统一转成 markdown 标记后交给 SplitModel,分段标识/分段长度才能生效
                content = self.handle_pdf_content(file, pdf_document, temp_file_path)

                if pattern_list is not None and len(pattern_list) > 0:
                    split_model = SplitModel(pattern_list, with_filter, limit)
                else:
                    split_model = SplitModel(default_pattern_list, with_filter=with_filter, limit=limit)
        except BaseException as e:
            maxkb_logger.error(f"File: {file.name}, error: {e}, {traceback.format_exc()}")
            return {"name": file.name, "content": []}
        finally:
            # 处理完后可以删除临时文件
            os.remove(temp_file_path)

        return {"name": file.name, "content": split_model.parse(content)}

    @staticmethod
    def handle_pdf_content(file, pdf_document, pdf_path=None):
        # 第一步:按版面把每页拆成有序的 block(文字行 / 表格),pdfplumber 不可用时退回 pypdf 视觉行
        pages = PdfSplitHandle.extract_pages_by_layout(pdf_path) if pdf_path else None
        if pages is None:
            pages = PdfSplitHandle.extract_pages_by_text(pdf_document)

        # 计算正文字体大小(众数)
        font_sizes = [
            font_size
            for blocks in pages
            for kind, _value, font_size, _width in blocks
            if kind == "line" and font_size > 0
        ]
        if not font_sizes:
            body_font_size = 12
        else:
            from collections import Counter

            body_font_size = Counter(font_sizes).most_common(1)[0][0]

        # 标题识别:书签优先,没有书签时退回正文编号模式
        heading_titles = PdfSplitHandle.get_outline_headings(pdf_document)
        line_widths = [width for blocks in pages for kind, _value, _font_size, width in blocks if kind == "line"]
        heading_max_width = 0 if heading_titles else PdfSplitHandle.get_heading_max_width(line_widths)

        # 第二步:提取内容
        content = ""
        for page_num, blocks in enumerate(pages):
            start_time = time.time()
            page_content = ""

            for kind, value, font_size, width in blocks:
                if kind == "table":
                    table_md = PdfSplitHandle.table_to_md(value)
                    if table_md:
                        page_content += f"\n{table_md}\n\n"
                    continue

                heading_level = PdfSplitHandle.get_line_heading_level(
                    value, font_size, body_font_size, heading_titles, heading_max_width, width
                )
                if heading_level is not None:
                    page_content += f"{'#' * heading_level} {value}\n\n"
                else:
                    page_content += f"{value}\n"

            page = pdf_document.pages[page_num]
            page_image_count = PdfSplitHandle.get_page_image_count(page)
            # 页面文字量过低但含图片,判定为扫描页/图片页,对该页图片做 OCR 并输出识别文字
            if PdfSplitHandle.get_real_text_length(page_content) < OCR_SPARSE_TEXT_THRESHOLD and page_image_count > 0:
                page_content = PdfSplitHandle.ocr_page_images(page, page_content)

            page_content = page_content.replace("\0", "")
            content += page_content

            elapsed_time = time.time() - start_time
            maxkb_logger.debug(f"File: {file.name}, Page: {page_num + 1}, Time: {elapsed_time:.3f}s")

        return content

    @staticmethod
    def extract_pages_by_layout(pdf_path):
        """
        用 pdfplumber 按版面提取:表格还原成 markdown,表格以外的文字按视觉行输出。
        PDF 里表格只是文字块加线段,没有行列语义,必须靠矢量线反推,否则窄列单元格的折行会被拆成独立文本行。
        :return: 每页的 block 列表,pdfplumber 不可用时返回 None,交给 pypdf 兜底
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return [PdfSplitHandle.extract_layout_blocks(page) for page in pdf.pages]
        except BaseException as e:
            maxkb_logger.warning(f"pdfplumber layout extract failed, fallback to pypdf: {e}")
            return None

    @staticmethod
    def extract_layout_blocks(page):
        """把一页拆成有序 block,表格块和表格外的文字行统一按纵坐标排序,还原阅读顺序"""
        try:
            tables = page.find_tables()
        except BaseException:
            tables = []

        blocks = []
        for table in tables:
            try:
                blocks.append((table.bbox[1], ("table", table.extract(), 0, 0)))
            except BaseException as e:
                maxkb_logger.warning(f"pdfplumber table extract failed: {e}")

        # 表格区域的文字已经在表格里,只取表格以外的部分,避免重复
        region = page
        for table in tables:
            region = region.outside_bbox(table.bbox)
        try:
            text_lines = region.extract_text_lines(x_tolerance_ratio=X_TOLERANCE_RATIO)
        except BaseException:
            text_lines = []

        for text_line in text_lines:
            text = text_line["text"].strip()
            if not text:
                continue
            chars = text_line.get("chars") or []
            font_size = max((char.get("size") or 0 for char in chars), default=0)
            # 有真实坐标,行宽直接用宽度,比字符数更准
            blocks.append((text_line["top"], ("line", text, float(font_size), text_line["x1"] - text_line["x0"])))

        blocks.sort(key=lambda block: block[0])
        return [block for _top, block in blocks]

    @staticmethod
    def extract_pages_by_text(pdf_document):
        """pypdf 兜底:只能拿到视觉行,没有表格结构,行宽用字符数近似"""
        pages = []
        for page in pdf_document.pages:
            blocks = [
                ("line", text, font_size, len(text))
                for text, font_size in PdfSplitHandle.extract_page_lines(page)
                if text
            ]
            pages.append(blocks)
        return pages

    @staticmethod
    def table_to_md(rows):
        """
        表格转 markdown。合并单元格会在网格里留下全空的行列,直接输出全是噪声,入库前丢掉
        """
        rows = [[PdfSplitHandle.normalize_cell(cell) for cell in row] for row in rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        columns = [index for index in range(width) if any(row[index] for row in rows)]
        if not columns:
            return ""
        rows = [[row[index] for index in columns] for row in rows]

        md_table = "| " + " | ".join(rows[0]) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(rows[0])) + " |"
        for row in rows[1:]:
            md_table += "\n| " + " | ".join(row) + " |"
        return md_table

    @staticmethod
    def normalize_cell(cell):
        """单元格里的折行是列宽造成的,拼回一行;竖线会破坏 markdown 表格结构,需要转义"""
        return (cell or "").replace("\n", "").replace("|", "\\|").strip()

    @staticmethod
    def get_real_text_length(page_content):
        """
        统计页面真实文字量。空白表单会输出只有骨架的 markdown 表格(9 列空表骨架就有 80 多个字符),
        直接按长度判断会让扫描页躲过 OCR,所以先把骨架剔掉
        """
        return len(MARKDOWN_SCAFFOLD_PATTERN.sub("", page_content))

    @staticmethod
    def get_outline_headings(doc):
        """
        从书签中挑出真正的标题。书签常把编号段落(如“（一）……”)也挂进来,
        只保留最外层且命中标题编号模式的条目。
        :return: 标题文本集合,没有可用书签时返回空集合
        """
        toc = PdfSplitHandle.get_toc(doc)
        if not toc:
            return set()
        top_level = min(level for level, _title, _page_number, _top in toc)
        return {
            title.strip()
            for level, title, _page_number, _top in toc
            if level == top_level and detect_heading_level(title) is not None
        }

    @staticmethod
    def get_heading_max_width(line_widths):
        """
        正文行会占满整行宽度,标题行明显更短,用行宽把标题和折行的正文区分开
        :return: 标题行宽上限,无法判断时返回 0
        """
        if not line_widths:
            return 0
        return int(max(line_widths) * HEADING_LINE_WIDTH_RATIO)

    @staticmethod
    def get_line_heading_level(text, font_size, body_font_size, heading_titles, heading_max_width, line_width):
        """
        判断一行文本的标题层级,返回 None 表示正文
        """
        # 明显大于正文的字号,作为文档/篇章大标题
        if font_size - body_font_size > 2:
            return 1

        text = text.strip()
        if heading_titles:
            return 2 if text in heading_titles else None

        if heading_max_width > 0 and line_width <= heading_max_width:
            level = detect_heading_level(text)
            if level is not None:
                return min(level + 1, MAX_HEADING_LEVEL)

        # 略大于正文的字号
        if font_size - body_font_size > 0.5:
            return 2
        return None

    @staticmethod
    def ocr_page_images(page, page_content):
        """对一页中的图片逐一 OCR,把识别文字追加到正文(不保留图片)"""
        try:
            images = page.images
        except BaseException:
            return page_content
        for image in images:
            try:
                recognized_text = ocr_image(image.data)
                if recognized_text:
                    page_content += f"\n{recognized_text}\n"
            except BaseException as e:
                maxkb_logger.error(f"OCR page image error: {e}, {traceback.format_exc()}")
        return page_content

    @staticmethod
    def extract_page_lines(page):
        lines = []
        current_text = []
        current_sizes = []

        def flush_line():
            text = "".join(current_text).strip()
            if text:
                font_size = current_sizes[0] if current_sizes else 0
                lines.append((text, font_size))
            current_text.clear()
            current_sizes.clear()

        def visitor_text(text, cm, tm, font_dict, font_size):
            if text is None:
                return
            parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for index, part in enumerate(parts):
                current_text.append(part)
                if part.strip() and font_size:
                    current_sizes.append(float(font_size))
                if index < len(parts) - 1:
                    flush_line()

        try:
            page.extract_text(visitor_text=visitor_text)
        except BaseException:
            text = PdfSplitHandle.extract_page_text(page)
            return [(line.strip(), 0) for line in text.splitlines() if line.strip()]
        flush_line()
        if lines:
            return lines

        text = page.extract_text() or ""
        return [(line.strip(), 0) for line in text.splitlines() if line.strip()]

    @staticmethod
    def get_page_image_count(page):
        try:
            return len(page.images)
        except BaseException:
            return 0

    @staticmethod
    def extract_page_text(page):
        return (page.extract_text() or "").replace("\0", "")

    @staticmethod
    def get_toc(doc):
        try:
            outline = doc.outline
        except BaseException:
            return []

        toc = []
        PdfSplitHandle.collect_toc(doc, outline, 1, toc)
        return toc

    @staticmethod
    def collect_toc(doc, outline, level, toc):
        for item in outline:
            if isinstance(item, list):
                PdfSplitHandle.collect_toc(doc, item, level + 1, toc)
                continue

            page_number = PdfSplitHandle.get_destination_page_number(doc, item)
            if page_number is None:
                continue

            title = getattr(item, "title", None)
            if title is None and hasattr(item, "get"):
                title = item.get("/Title")
            if title is None:
                title = str(item)
            toc.append(
                (
                    level,
                    str(title).replace("\0", ""),
                    page_number,
                    PdfSplitHandle.get_destination_top(item),
                )
            )

    @staticmethod
    def get_destination_top(destination):
        top = getattr(destination, "top", None)
        try:
            return float(top)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def extract_page_text_by_position(page, top=None, bottom=None):
        if top is None and bottom is None:
            return PdfSplitHandle.extract_page_text(page)

        text_parts = []

        def visitor_text(text, cm, tm, font_dict, font_size):
            if not text:
                return

            # Text matrix coordinates can be relative to a page-level transform.
            # Convert the text origin to PDF user-space coordinates before comparing
            # it with the outline destination's /Top value.
            x = tm[4] if len(tm) > 4 else 0
            y = tm[5] if len(tm) > 5 else 0
            if len(cm) > 5:
                y = x * cm[1] + y * cm[3] + cm[5]

            if top is not None and y > top:
                return
            if bottom is not None and y <= bottom:
                return
            text_parts.append(text)

        try:
            page.extract_text(visitor_text=visitor_text)
        except BaseException:
            return PdfSplitHandle.extract_page_text(page)
        return "".join(text_parts).replace("\0", "")

    @staticmethod
    def remove_leading_title(text, *titles):
        for title in titles:
            title = title.strip()
            if not title:
                continue
            pattern = r"^\s*" + r"\s*".join(re.escape(char) for char in title)
            stripped_text, count = re.subn(pattern, "", text, count=1)
            if count:
                return stripped_text
        return text

    @staticmethod
    def discard_ambiguous_destination_tops(toc):
        position_counts = {}
        for _level, _title, page_number, top in toc:
            if top is not None:
                position = (page_number, top)
                position_counts[position] = position_counts.get(position, 0) + 1

        ambiguous_tops = {top for (_page_number, top), count in position_counts.items() if count > 1}

        return [
            (level, title, page_number, None if top in ambiguous_tops else top)
            for level, title, page_number, top in toc
        ]

    @staticmethod
    def handle_toc(doc, limit):
        # 找到目录
        toc = PdfSplitHandle.get_toc(doc)
        if toc is None or len(toc) == 0:
            return None
        # Some PDF generators assign the same default position to every bookmark
        # on a page. Such coordinates cannot define chapter boundaries, so preserve
        # the title-based behavior for those entries.
        toc = PdfSplitHandle.discard_ambiguous_destination_tops(toc)

        # 创建存储章节内容的数组
        chapters = []

        # 遍历目录并按章节提取文本
        for i, entry in enumerate(toc):
            level, title, start_page, start_top = entry
            chapter_title = title
            # 确定结束页码，如果是最后一个章节则到文档末尾
            if i + 1 < len(toc):
                _next_level, next_title, next_start_page, next_top = toc[i + 1]
                # A positioned bookmark can start partway down a page. Include that
                # page and keep only the text above the next bookmark for this chapter.
                end_page = next_start_page if next_top is not None else next_start_page - 1
            else:
                end_page = len(doc.pages) - 1
                next_title = None
                next_start_page = None
                next_top = None
            end_page = max(start_page, end_page)

            # 去掉标题中的符号
            title = PdfSplitHandle.handle_chapter_title(title)

            # 提取该章节的文本内容
            chapter_text = ""
            for page_num in range(start_page, end_page + 1):
                page_top = start_top if page_num == start_page else None
                page_bottom = next_top if page_num == next_start_page else None
                text = PdfSplitHandle.extract_page_text_by_position(doc.pages[page_num], page_top, page_bottom)
                text = re.sub(r"(?<!。)\n+", "", text)
                text = re.sub(r"(?<!.)\n+", "", text)

                if page_num == start_page:
                    if start_top is not None:
                        text = PdfSplitHandle.remove_leading_title(text, chapter_title, title)
                    else:
                        idx = text.find(title)
                        if idx > -1:
                            text = text[idx + len(title) :]

                if next_title is not None and next_top is None:
                    handled_next_title = PdfSplitHandle.handle_chapter_title(next_title)
                    idx = text.find(handled_next_title)
                    if idx > -1:
                        text = text[:idx]

                chapter_text += text  # 提取文本

            # Null characters are not allowed.
            chapter_text = chapter_text.replace("\0", "")
            # 限制标题长度
            real_chapter_title = chapter_title[:256]
            # 限制章节内容长度
            if 0 < limit < len(chapter_text):
                split_text = smart_split_paragraph(chapter_text, limit)
                for text in split_text:
                    chapters.append(
                        {"title": real_chapter_title, "content": text.encode("utf-8", "ignore").decode("utf-8")}
                    )
            else:
                chapters.append(
                    {
                        "title": real_chapter_title,
                        "content": (chapter_text if chapter_text else real_chapter_title)
                        .encode("utf-8", "ignore")
                        .decode("utf-8"),
                    }
                )
            # 保存章节内容和章节标题
        return chapters

    @staticmethod
    def handle_links(doc, pattern_list, with_filter, limit):
        # 检查文档是否包含内部链接
        if not check_links_in_pdf(doc):
            return
        # 创建存储章节内容的数组
        chapters = []
        toc_start_page = -1
        page_content = ""
        handle_pre_toc = True
        # 遍历 PDF 的每一页，查找带有目录链接的页
        for page_num, page in enumerate(doc.pages):
            links = PdfSplitHandle.get_internal_links(doc, page)
            # 如果目录开始页码未设置，则设置为当前页码
            if len(links) > 0 and toc_start_page < 0:
                toc_start_page = page_num
            if toc_start_page < 0:
                page_content += PdfSplitHandle.extract_page_text(page)
            # 检查该页是否包含内部链接（即指向文档内部的页面）
            for num in range(len(links)):
                link = links[num]
                # 获取链接目标的页面
                dest_page = link["page"]
                rect = link["from"]  # 获取链接的矩形区域
                # 如果目录开始页码包括前言部分，则不处理前言部分
                if dest_page < toc_start_page:
                    handle_pre_toc = False

                # 提取链接区域的文本作为标题
                link_title = PdfSplitHandle.extract_link_title(page, rect)
                if not link_title:
                    link_title = PdfSplitHandle.extract_first_line(doc.pages[dest_page])
                # 提取目标页面内容作为章节开始
                start_page = dest_page
                end_page = dest_page
                # 下一个link
                next_link = links[num + 1] if num + 1 < len(links) else None
                next_link_title = None
                if next_link is not None:
                    next_link_title = PdfSplitHandle.extract_link_title(page, next_link["from"])
                    if not next_link_title:
                        next_link_title = PdfSplitHandle.extract_first_line(doc.pages[next_link["page"]])
                    end_page = next_link["page"]

                # 提取章节内容
                chapter_text = ""
                for p_num in range(start_page, min(end_page, len(doc.pages) - 1) + 1):
                    text = PdfSplitHandle.extract_page_text(doc.pages[p_num])
                    text = re.sub(r"(?<!。)\n+", "", text)
                    text = re.sub(r"(?<!.)\n+", "", text)

                    idx = text.find(link_title)
                    if idx > -1:
                        text = text[idx + len(link_title) :]

                    if next_link_title is not None:
                        idx = text.find(next_link_title)
                        if idx > -1:
                            text = text[:idx]
                    chapter_text += text

                # Null characters are not allowed.
                chapter_text = chapter_text.replace("\0", "")

                # 限制章节内容长度
                if 0 < limit < len(chapter_text):
                    split_text = smart_split_paragraph(chapter_text, limit)
                    for text in split_text:
                        chapters.append({"title": link_title, "content": text})
                else:
                    # 保存章节信息
                    chapters.append({"title": link_title, "content": chapter_text})

        # 目录中没有前言部分，手动处理
        if handle_pre_toc:
            pre_toc = []
            lines = page_content.strip().split("\n")
            try:
                for line in lines:
                    if re.match(r"^前\s*言", line):
                        pre_toc.append({"title": line, "content": ""})
                    else:
                        pre_toc[-1]["content"] += line
                for i in range(len(pre_toc)):
                    pre_toc[i]["content"] = re.sub(r"(?<!。)\n+", "", pre_toc[i]["content"])
                    pre_toc[i]["content"] = re.sub(r"(?<!.)\n+", "", pre_toc[i]["content"])
            except BaseException as e:
                maxkb_logger.error(_("This document has no preface and is treated as ordinary text: {e}").format(e=e))
                if pattern_list is not None and len(pattern_list) > 0:
                    split_model = SplitModel(pattern_list, with_filter, limit)
                else:
                    split_model = SplitModel(default_pattern_list, with_filter=with_filter, limit=limit)
                # 插入目录前的部分
                page_content = re.sub(r"(?<!。)\n+", "", page_content)
                page_content = re.sub(r"(?<!.)\n+", "", page_content)
                page_content = page_content.strip()
                pre_toc = split_model.parse(page_content)
            chapters = pre_toc + chapters
        return chapters

    @staticmethod
    def get_internal_links(doc, page):
        links = []
        annotations = getattr(page, "annotations", None) or []
        for annotation in annotations:
            annotation = get_pdf_object(annotation)
            if not hasattr(annotation, "get"):
                continue
            if annotation.get("/Subtype") != "/Link":
                continue
            dest_page = PdfSplitHandle.get_annotation_destination_page_number(doc, annotation)
            if dest_page is None or dest_page < 0 or dest_page >= len(doc.pages):
                continue
            rect = annotation.get("/Rect")
            links.append({"page": dest_page, "from": PdfSplitHandle.normalize_rect(rect)})
        return links

    @staticmethod
    def get_annotation_destination_page_number(doc, annotation):
        destination = annotation.get("/Dest")
        if destination is None:
            action = get_pdf_object(annotation.get("/A"))
            if hasattr(action, "get") and action.get("/S") == "/GoTo":
                destination = action.get("/D")
        return PdfSplitHandle.get_destination_page_number(doc, destination)

    @staticmethod
    def get_destination_page_number(doc, destination):
        if destination is None:
            return None

        destination = get_pdf_object(destination)

        if isinstance(destination, bytes):
            destination = destination.decode(errors="ignore")

        if isinstance(destination, str):
            destination = doc.named_destinations.get(destination)
            if destination is None:
                return None

        if isinstance(destination, Destination):
            try:
                page_number = doc.get_destination_page_number(destination)
                return page_number if page_number >= 0 else None
            except BaseException:
                return None

        if isinstance(destination, (list, tuple)) and len(destination) > 0:
            return PdfSplitHandle.get_page_number_by_reference(doc, destination[0])

        if hasattr(destination, "get") and destination.get("/D") is not None:
            return PdfSplitHandle.get_destination_page_number(doc, destination.get("/D"))

        return None

    @staticmethod
    def get_page_number_by_reference(doc, page_reference):
        try:
            page_number = int(page_reference)
            if 0 <= page_number < len(doc.pages):
                return page_number
        except BaseException:
            pass

        try:
            page = get_pdf_object(page_reference)
            page_number = doc.get_page_number(page)
            return page_number if page_number >= 0 else None
        except BaseException:
            return None

    @staticmethod
    def normalize_rect(rect):
        if rect is None or len(rect) < 4:
            return None
        left, bottom, right, top = [float(value) for value in rect[:4]]
        return min(left, right), min(bottom, top), max(left, right), max(bottom, top)

    @staticmethod
    def extract_link_title(page, rect):
        if rect is None:
            return ""

        left, bottom, right, top = rect
        tolerance = 2
        text_parts = []

        def visitor_text(text, cm, tm, font_dict, font_size):
            if not text:
                return
            x = tm[4] if len(tm) > 4 else 0
            y = tm[5] if len(tm) > 5 else 0
            text_top = y + (float(font_size) if font_size else 0)
            in_horizontal_range = left - tolerance <= x <= right + tolerance
            in_vertical_range = (
                bottom - tolerance <= y <= top + tolerance or bottom - tolerance <= text_top <= top + tolerance
            )
            if in_horizontal_range and in_vertical_range:
                text_parts.append(text)

        try:
            page.extract_text(visitor_text=visitor_text)
        except BaseException:
            return ""

        return "".join(text_parts).replace("\0", "").strip().split("\n")[0].replace(".", "").strip()

    @staticmethod
    def extract_first_line(page):
        text = PdfSplitHandle.extract_page_text(page).strip()
        return text.split("\n")[0].replace(".", "").strip() if text else ""

    @staticmethod
    def handle_chapter_title(title):
        title = title.replace("\0", "")
        title = re.sub(r"[一二三四五六七八九十\s*]、\s*", "", title)
        title = re.sub(r"第[一二三四五六七八九十]章\s*", "", title)
        return title

    def support(self, file, get_buffer):
        file_name: str = file.name.lower()
        if file_name.endswith(".pdf") or file_name.endswith(".PDF"):
            return True
        return False

    def get_content(self, file, save_image):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # 将上传的文件保存到临时文件中
            temp_file.write(file.read())
            # 获取临时文件的路径
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, "rb") as pdf_file:
                pdf_document = PdfReader(pdf_file)
                return self.handle_pdf_content(file, pdf_document, temp_file_path)
        except BaseException as e:
            traceback.print_exception(e)
            return f"{e}"
        finally:
            os.remove(temp_file_path)
