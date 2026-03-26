from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.config import (
    KEYWORDS_EDUCATION,
    KEYWORDS_HOBBY,
    KEYWORDS_TOXIC,
    PERCENTILE_HIGH,
    PERCENTILE_LOW,
    WEIGHT_ACTIVITY,
    WEIGHT_GROUPS,
    WEIGHT_MUSIC,
    WEIGHT_SENTIMENT_MERGED,
    WEIGHT_SOCIAL,
    WEIGHT_VISUAL,
)
from src.deeppavlov_tone import deeppavlov_mean_score
from src.image_emotion import score_profile_photo_url
from src.music_mood import compute_music_mood
from src.rubert_sentiment import rubert_mean_score


def _log1p_sum(*vals: float) -> float:
    return float(sum(math.log1p(max(0.0, v)) for v in vals))


def robust_minmax_01(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    lo = np.percentile(v, PERCENTILE_LOW)
    hi = np.percentile(v, PERCENTILE_HIGH)
    if hi <= lo:
        return np.full_like(v, 0.5, dtype=float)
    clipped = np.clip(v, lo, hi)
    return np.clip((clipped - lo) / (hi - lo), 0.0, 1.0)


def _norm_optional_masked(values: np.ndarray) -> np.ndarray | None:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    med = float(np.nanmedian(values))
    filled = np.nan_to_num(values, nan=med)
    return robust_minmax_01(filled)


def group_theme_score(groups: dict[str, Any]) -> float:
    if not groups:
        return 0.5
    hobby = edu = toxic = 0
    total = len(groups)
    for g in groups.values():
        name = (g.get("name") or "").lower()
        desc = (g.get("description") or "").lower()
        blob = f"{name} {desc}"
        if any(k in blob for k in KEYWORDS_TOXIC):
            toxic += 1
        if any(k in blob for k in KEYWORDS_HOBBY):
            hobby += 1
        if any(k in blob for k in KEYWORDS_EDUCATION):
            edu += 1
    raw = (hobby + edu) / max(1, total) - 0.6 * (toxic / max(1, total))
    return float(np.clip(raw, 0.0, 1.0))


def _model_bin_candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("DOSTOEVSKY_MODEL_PATH", "").strip()
    if env:
        out.append(Path(env))
    out.append(_PROJECT_ROOT / "models" / "fasttext-social-network-model.bin")
    try:
        from dostoevsky.data import DATA_BASE_PATH
        out.append(Path(DATA_BASE_PATH) / "models" / "fasttext-social-network-model.bin")
    except Exception:
        pass
    return out


def _existing_model_bin() -> Path | None:
    for p in _model_bin_candidates():
        try:
            if p.is_file() and p.stat().st_size > 1000:
                return p.resolve()
        except OSError:
            continue
    return None


def _ensure_dostoevsky_weights() -> None:
    import os

    if _existing_model_bin() is not None:
        return
    from dostoevsky.data import AVAILABLE_FILES, DATA_BASE_PATH, DataDownloader

    key = "fasttext-social-network-model"
    bin_path = os.path.join(DATA_BASE_PATH, "models", "fasttext-social-network-model.bin")
    if os.path.isfile(bin_path) and os.path.getsize(bin_path) > 1000:
        return
    try:
        downloader = DataDownloader()
        source, destination = AVAILABLE_FILES[key]
        dest_path = os.path.join(DATA_BASE_PATH, destination)
        if not os.path.isfile(dest_path):
            downloader.download(source=source, destination=destination)
    except Exception:
        pass


def _positive_share_from_dostoevsky_dict(item: dict) -> float:
    lower = {str(k).lower(): float(v) for k, v in item.items()}
    pos = lower.get("positive", lower.get("pos", 0.0))
    neg = lower.get("negative", lower.get("neg", 0.0))
    neu = lower.get("neutral", lower.get("neu", 0.0))
    s = pos + neg + neu
    if s > 1e-9:
        return pos / s
    if lower:
        return max(0.0, min(1.0, float(next(iter(lower.values())))))
    return 0.5


def build_sentiment_predictor():
    try:
        _ensure_dostoevsky_weights()
        from dostoevsky.models import FastTextSocialNetworkModel
        from dostoevsky.tokenization import RegexTokenizer

        bin_path = _existing_model_bin()
        if bin_path is not None:
            FastTextSocialNetworkModel.MODEL_PATH = str(bin_path)
        tokenizer = RegexTokenizer()
        model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    except Exception:
        return None, _fallback_sentiment_ru

    def predict_proba_positive(text: str) -> float:
        if not text or not text.strip():
            return 0.5
        try:
            item: dict = model.predict([text])[0]
            return _positive_share_from_dostoevsky_dict(item)
        except Exception:
            return _fallback_sentiment_ru(text)

    return model, predict_proba_positive


def _fallback_sentiment_ru(text: str) -> float:
    t = text.lower()
    pos = len(re.findall(r"рад|счастл|люблю|хорош|класс|супер|отличн|😊|❤️", t))
    neg = len(re.findall(r"устал|бесит|плох|ненавиж|груст|депресс|😢|😠", t))
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


def user_sentiment_dostoevsky_only(
    posts: dict[str, Any],
    model: Any,
    predict_proba,
) -> float:
    texts = [(p.get("text") or "").strip() for p in posts.values() if (p.get("text") or "").strip()]
    if not texts:
        return 0.5
    texts = texts[:80]
    if model is not None:
        try:
            batch = model.predict(texts)
            scores = [_positive_share_from_dostoevsky_dict(d) for d in batch]
            return float(np.mean(scores))
        except Exception:
            pass
    scores = [predict_proba(t) for t in texts]
    return float(np.mean(scores))


def compute_happiness_for_corpus(users: list[dict]) -> list[dict]:
    model_d, predict_proba = build_sentiment_predictor()

    n = len(users)
    raw_social: list[float] = []
    raw_activity: list[float] = []
    raw_dost: list[float] = []
    raw_rubert: list[float] = []
    raw_dp: list[float] = []
    raw_groups: list[float] = []
    raw_visual: list[float] = []
    raw_music: list[float] = []

    for u in users:
        friends = float(u.get("friends_num") or 0)
        followers = float(u.get("followers_num") or 0)
        groups_n = float(u.get("groups_num") or 0)
        posts_n = float(u.get("self_posts_num") or 0)
        likes = float(u.get("likes_received_num") or 0)
        comments = float(u.get("comments_received_num") or 0)

        raw_social.append(_log1p_sum(friends, followers, groups_n))
        raw_activity.append(_log1p_sum(posts_n, likes, comments))

        posts = u.get("posts") or {}
        raw_dost.append(user_sentiment_dostoevsky_only(posts, model_d, predict_proba))

        rv = rubert_mean_score(posts)
        raw_rubert.append(rv if rv is not None else np.nan)

        dpv = deeppavlov_mean_score(posts)
        raw_dp.append(dpv if dpv is not None else np.nan)

        raw_groups.append(group_theme_score(u.get("groups") or {}))

        url = (u.get("photo_200") or "").strip()
        if os.environ.get("SKIP_PROFILE_IMAGE_EMOTION", "").strip() in ("1", "true", "yes"):
            raw_visual.append(0.5)
        else:
            img = score_profile_photo_url(url) if url else None
            raw_visual.append(float(img) if img is not None else 0.5)

        raw_music.append(compute_music_mood(u))

    arr_s = np.array(raw_social)
    arr_a = np.array(raw_activity)
    arr_g = np.array(raw_groups)
    arr_v = np.array(raw_visual)
    arr_m = np.array(raw_music)

    s_s = robust_minmax_01(arr_s)
    s_a = robust_minmax_01(arr_a)
    s_g = np.clip(robust_minmax_01(arr_g), 0.0, 1.0)
    s_v = np.clip(robust_minmax_01(arr_v), 0.0, 1.0)
    s_m = np.clip(robust_minmax_01(arr_m), 0.0, 1.0)

    arr_d = np.array(raw_dost)
    arr_r = np.array(raw_rubert)
    arr_p = np.array(raw_dp)

    n_d = robust_minmax_01(arr_d)
    n_r = _norm_optional_masked(arr_r)
    n_p = _norm_optional_masked(arr_p)

    parts = [n_d]
    if n_r is not None:
        parts.append(n_r)
    if n_p is not None:
        parts.append(n_p)
    s_sent = np.mean(np.stack(parts, axis=0), axis=0)

    wsum = (WEIGHT_SOCIAL + WEIGHT_ACTIVITY + WEIGHT_SENTIMENT_MERGED
            + WEIGHT_GROUPS + WEIGHT_VISUAL + WEIGHT_MUSIC)
    out: list[dict] = []
    for i, row in enumerate(users):
        H = (
            WEIGHT_SOCIAL * s_s[i]
            + WEIGHT_ACTIVITY * s_a[i]
            + WEIGHT_SENTIMENT_MERGED * s_sent[i]
            + WEIGHT_GROUPS * s_g[i]
            + WEIGHT_VISUAL * s_v[i]
            + WEIGHT_MUSIC * s_m[i]
        ) / wsum if wsum else 0.0
        row = dict(row)
        row["happiness_index"] = round(float(H), 4)
        row["components"] = {
            "social_01": round(float(s_s[i]), 4),
            "activity_01": round(float(s_a[i]), 4),
            "sentiment_merged_01": round(float(s_sent[i]), 4),
            "sentiment_dostoevsky_01": round(float(n_d[i]), 4),
            "sentiment_rubert_01": round(float(n_r[i]), 4) if n_r is not None else None,
            "sentiment_deeppavlov_01": round(float(n_p[i]), 4) if n_p is not None else None,
            "groups_theme_01": round(float(s_g[i]), 4),
            "visual_emotion_01": round(float(s_v[i]), 4),
            "music_mood_01": round(float(s_m[i]), 4),
        }
        out.append(row)
    return out
