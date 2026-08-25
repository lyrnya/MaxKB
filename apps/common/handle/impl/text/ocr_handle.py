# coding=utf-8
"""
    @project: maxkb
    @Author：lyrnya
    @file： ocr_handle.py
    @date：2026/8/25
    @desc: 基于 RapidOCR(onnxruntime)的图片文字识别,懒加载单例,按需调用
"""
import io
import threading
import traceback

from common.utils.logger import maxkb_logger

_engine = None
_engine_lock = threading.Lock()
_init_lock = threading.Lock()


def _get_engine():
    global _engine
    if _engine is None:
        with _init_lock:
            if _engine is None:
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    _engine = RapidOCR()
                    maxkb_logger.info("RapidOCR engine loaded")
                except Exception as e:
                    maxkb_logger.error(f"Failed to load RapidOCR: {e}, {traceback.format_exc()}")
                    raise
    return _engine


def ocr_image(image_bytes: bytes) -> str:
    """识别图片中的文字,失败时返回空字符串,不影响文档解析主流程"""
    if not image_bytes:
        return ""
    try:
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        array = np.array(image)
        # onnxruntime session 对并发推理线程安全,加锁避免共享引擎状态竞争
        with _engine_lock:
            result, _elapse = _get_engine()(array)
        if not result:
            return ""
        return "\n".join(line[1] for line in result).strip()
    except Exception as e:
        maxkb_logger.error(f"OCR error: {e}, {traceback.format_exc()}")
        return ""
