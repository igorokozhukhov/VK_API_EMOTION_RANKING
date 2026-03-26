from __future__ import annotations

import os
from typing import Any

import numpy as np

_MAX_POSTS = int(os.environ.get("RUBERT_MAX_POSTS", "24"))
_MODEL_NAME = os.environ.get("RUBERT_MODEL_NAME", "cointegrated/rubert-tiny2")

_POS = (
    "Я счастлив, доволен жизнью, полон энергии и позитива. "
    "Радуюсь общению с друзьями и новым впечатлениям."
)
_NEG = (
    "Мне грустно, тревожно и безнадёжно. Устал от всего, "
    "ничего не радует, только уныние и раздражение."
)


class RuBERTAnchorSentiment:

    def __init__(self) -> None:
        self._tok = None
        self._model = None
        self._pos_emb: Any = None
        self._neg_emb: Any = None
        self._device = None

    def _lazy_init(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModel.from_pretrained(_MODEL_NAME)
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            pe = self._encode_batch([_POS])
            ne = self._encode_batch([_NEG])
            self._pos_emb = self._l2n(pe[0])
            self._neg_emb = self._l2n(ne[0])
        return True

    def _l2n(self, v):
        import torch
        return v / (v.norm(dim=-1, keepdim=True) + 1e-9)

    def _encode_batch(self, texts: list[str]):
        import torch
        if not texts:
            return torch.zeros(0, 0, device=self._device)
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(out.shape).float()
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return self._l2n(summed / counts)

    def score_posts(self, posts: dict[str, Any]) -> float | None:
        texts = [(p.get("text") or "").strip() for p in posts.values() if (p.get("text") or "").strip()]
        texts = texts[:_MAX_POSTS]
        if not texts:
            return 0.5
        if not self._lazy_init():
            return None
        import torch
        embs = self._encode_batch(texts)
        cos_p = (embs * self._pos_emb).sum(dim=-1)
        cos_n = (embs * self._neg_emb).sum(dim=-1)
        scores = torch.sigmoid(3.0 * (cos_p - cos_n))
        return float(scores.mean().cpu().numpy())


_singleton: RuBERTAnchorSentiment | None = None
_rubert_unavailable: bool = False


def get_rubert_scorer() -> RuBERTAnchorSentiment | None:
    global _singleton, _rubert_unavailable
    if _rubert_unavailable:
        return None
    if _singleton is not None:
        return _singleton
    s = RuBERTAnchorSentiment()
    if not s._lazy_init():
        _rubert_unavailable = True
        return None
    _singleton = s
    return s


def rubert_mean_score(posts: dict[str, Any]) -> float | None:
    sc = get_rubert_scorer()
    if sc is None:
        return None
    return sc.score_posts(posts)
