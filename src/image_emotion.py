from __future__ import annotations

import os
import tempfile
from io import BytesIO
import numpy as np
import requests


def _pil_contrast_fallback(content: bytes) -> float | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(BytesIO(content)).convert("L")
        a = np.asarray(im, dtype=np.float64) / 255.0
        if a.size < 16:
            return None
        std = float(np.std(a))
        t = (std - 0.06) / 0.28
        return float(np.clip(0.2 + 0.6 * t, 0.1, 0.95))
    except Exception:
        return None


def score_profile_photo_url(url: str) -> float | None:
    if os.environ.get("SKIP_PROFILE_IMAGE_EMOTION", "").strip() in ("1", "true", "yes"):
        return None
    if not url or not url.startswith("http"):
        return None
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research)"})
        r.raise_for_status()
        content = r.content
    except Exception:
        return None

    deepface_ok = False
    try:
        from deepface import DeepFace
        deepface_ok = True
    except Exception:
        pass

    if deepface_ok:
        suf = ".jpg" if "jpeg" in (r.headers.get("content-type") or "").lower() or "jpg" in url.lower() else ".png"
        if "png" in url.lower():
            suf = ".png"
        try:
            with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
                tmp.write(content)
                path = tmp.name
            try:
                res = DeepFace.analyze(
                    path,
                    actions=["emotion"],
                    enforce_detection=False,
                )
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            if res:
                row = res[0] if isinstance(res, list) else res
                em = row.get("emotion") or {}
                if em:
                    return _emotions_dict_to_01(em)
                dom = row.get("dominant_emotion") or ""
                return _dominant_to_01(str(dom))
        except Exception:
            pass

    return _pil_contrast_fallback(content)


_EMOTION_POS = ("happy", "surprise")
_EMOTION_NEG = ("sad", "angry", "fear", "disgust")


def _dominant_to_01(dominant: str) -> float:
    d = (dominant or "").lower()
    if d in _EMOTION_POS:
        return 0.85 if d == "happy" else 0.65
    if d in _EMOTION_NEG:
        return 0.25
    return 0.5


def _emotions_dict_to_01(em: dict[str, float | int]) -> float:
    h = float(em.get("happy", 0)) + 0.5 * float(em.get("surprise", 0))
    n = (
        float(em.get("sad", 0))
        + float(em.get("angry", 0))
        + float(em.get("fear", 0))
        + float(em.get("disgust", 0))
    )
    t = h + n + 1e-6
    return max(0.0, min(1.0, h / t))
