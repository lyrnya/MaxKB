# coding=utf-8
"""
@project: maxkb
@Author：虎
@file： heading.py
@date：2026/8/26 10:30
@desc: 文档标题识别,按编号模式判断标题层级
"""

import re

# 标题编号模式,按层级从高到低排列,下标 + 1 即标题层级
HEADING_NUMBER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千]+[章篇部]"),
    re.compile(r"^第[一二三四五六七八九十百千]+[节条]"),
    re.compile(r"^[一二三四五六七八九十百]+[、.．]"),
    re.compile(r"^[（(][一二三四五六七八九十百]+[)）]"),
    re.compile(r"^\d+(?:\.\d+)*(?:[、.．]|\s)"),
]

# 标题字数上限,超过则视为正文
MAX_HEADING_LENGTH = 40

# markdown 标题最大层级
MAX_HEADING_LEVEL = 6


def detect_heading_level(text: str, max_length: int = MAX_HEADING_LENGTH):
    """
    根据编号模式判断一行文本的标题层级
    :param text:       单行文本
    :param max_length: 标题字数上限
    :return: 标题层级(从 1 开始),不是标题则返回 None
    """
    if text is None:
        return None
    text = text.strip()
    if not text or len(text) > max_length:
        return None
    for index, pattern in enumerate(HEADING_NUMBER_PATTERNS):
        match = pattern.match(text)
        if match is None:
            continue
        # 1.1 / 1.1.1 这类多级编号,按编号深度继续下沉
        level = index + 1 + match.group().count(".") - (1 if match.group().endswith(".") else 0)
        return min(max(level, 1), MAX_HEADING_LEVEL)
    return None
