from __future__ import annotations

import os
import sys
from typing import Any

_MAX_POSTS = int(os.environ.get("DEEPPAVLOV_MAX_POSTS", "40"))
_HF_MODEL = os.environ.get(
    "HF_RUSSIAN_SENTIMENT_MODEL",
    "blanchefort/rubert-base-cased-sentiment-kaggle",
)


def _debug(msg: str) -> None:
    if os.environ.get("DEBUG_ML", "").strip() in ("1", "true", "yes"):
        print(f"[deeppavlov_tone] {msg}", file=sys.stderr)


def _label_to_01(label: Any) -> float:
    s = str(label).lower()
    if "pos" in s or "позит" in s:
        return 1.0
    if "neg" in s or "негат" in s:
        return 0.0
    return 0.5


def _load_deeppavlov_model():
    try:
        import deeppavlov
        from deeppavlov import build_model
    except ImportError as e:
        _debug(f"deeppavlov import: {e}")
        return None
    candidates: list = []
    try:
        from deeppavlov import configs
        candidates.append(configs.classifiers.rusentiment.rusentiment_cnn)
    except Exception as e:
        _debug(f"configs rusentiment_cnn: {e}")
    try:
        from deeppavlov import configs
        candidates.append(configs.classifiers.rusentiment.rusentiment_bigru)
    except Exception as e:
        _debug(f"configs rusentiment_bigru: {e}")
    try:
        root = os.path.dirname(deeppavlov.__file__)
        for rel in (
            "configs/classifiers/rusentiment/rusentiment_cnn.json",
            "configs/classifiers/rusentiment/rusentiment_bigru.json",
        ):
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                candidates.append(p)
    except Exception as e:
        _debug(f"path configs: {e}")
    for cfg in candidates:
        try:
            return build_model(cfg, download=True)
        except Exception as e:
            _debug(f"build_model failed: {e}")
            continue
    return None


_dp_model = None
_hf_pipe = None
_hf_failed = False


def _get_dp_model():
    global _dp_model
    if _dp_model is False:
        return None
    if _dp_model is None:
        _dp_model = _load_deeppavlov_model()
        if _dp_model is None:
            _dp_model = False
            _debug("DeepPavlov недоступен, будет запасной HuggingFace.")
    return _dp_model if _dp_model is not False else None


def _get_hf_pipe():
    global _hf_pipe, _hf_failed
    if _hf_failed:
        return None
    if _hf_pipe is not None:
        return _hf_pipe
    try:
        from transformers import pipeline
    except ImportError as e:
        _debug(f"transformers import: {e}")
        _hf_failed = True
        return None
    try:
        _hf_pipe = pipeline(
            "sentiment-analysis",
            model=_HF_MODEL,
            tokenizer=_HF_MODEL,
        )
        _debug(f"HuggingFace pipeline загружен: {_HF_MODEL}")
    except Exception as e:
        _debug(f"HuggingFace pipeline error: {e}")
        _hf_failed = True
        return None
    return _hf_pipe


def _hf_texts_to_mean_01(texts: list[str]) -> float | None:
    pipe = _get_hf_pipe()
    if pipe is None:
        return None
    scores: list[float] = []
    try:
        batch = 8
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            outs = pipe(chunk, truncation=True, max_length=256)
            if isinstance(outs, dict):
                outs = [outs]
            for item in outs:
                lab = str(item.get("label", "")).lower()
                p = float(item.get("score", 0.5))
                if "neg" in lab:
                    scores.append(1.0 - p)
                elif "pos" in lab:
                    scores.append(p)
                else:
                    scores.append(0.5)
    except Exception as e:
        _debug(f"HF predict error: {e}")
        return None
    if not scores:
        return None
    return sum(scores) / len(scores)


def _dp_mean_score(texts: list[str]) -> float | None:
    m = _get_dp_model()
    if m is None:
        return None
    try:
        out = m(texts)
    except Exception as e:
        _debug(f"DeepPavlov predict error: {e}")
        return None
    if not out:
        return None
    scores: list[float] = []
    if isinstance(out[0], (list, tuple)):
        for row in out:
            if isinstance(row, (list, tuple)) and row:
                scores.append(_label_to_01(row[0]))
            else:
                scores.append(_label_to_01(row))
    else:
        for row in out:
            scores.append(_label_to_01(row))
    if not scores:
        return None
    return sum(scores) / len(scores)


def deeppavlov_mean_score(posts: dict[str, Any]) -> float | None:
    texts = [(p.get("text") or "").strip() for p in posts.values() if (p.get("text") or "").strip()]
    texts = texts[:_MAX_POSTS]
    if not texts:
        return None
    dp = _dp_mean_score(texts)
    if dp is not None:
        return dp
    return _hf_texts_to_mean_01(texts)
