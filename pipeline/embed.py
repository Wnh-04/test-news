"""向量嵌入封装：fastembed + ONNX（无需 torch）。
模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（384 维，中英对齐）。
国内网络下 huggingface.co 不可达时自动切 hf-mirror.com。
懒加载 + 进程级单例（模型加载 ~2s，别放在模块 import 时）。
"""
import logging
import os

log = logging.getLogger("embed")

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384
_instance = None


class Embedder:
    def __init__(self):
        self._model = None

    def _ensure(self):
        global _instance
        if self._model is None:
            if _instance is not None:      # 复用已加载模型（serve 与 pipeline 同进程场景）
                self._model = _instance
                return
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from fastembed import TextEmbedding
            log.info("加载嵌入模型 %s (HF_ENDPOINT=%s)…", MODEL, os.environ["HF_ENDPOINT"])
            self._model = TextEmbedding(model_name=MODEL)
            _instance = self._model
            log.info("模型加载完成")

    def embed(self, texts):
        self._ensure()
        import numpy as np
        out = list(self._model.embed(texts, batch_size=32))
        return np.array([e / (np.linalg.norm(e) + 1e-12) for e in out])  # L2 归一化

    def embed_one(self, text: str):
        return self.embed([text])[0]
