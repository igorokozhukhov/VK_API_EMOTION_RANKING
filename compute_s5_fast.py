#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "data.json"
DOWNLOAD_THREADS = 30
TIMEOUT = 8
MODEL_NAME = "trpakov/vit-face-expression"


def download_avatar(url: str) -> Image.Image | None:
    if not url or not url.startswith("http"):
        return None
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers={"User-Agent": "Mozilla/5.0 (research)"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except Exception:
        return None


def emotions_to_score(result: dict) -> float:
    scores = {r["label"].lower(): r["score"] for r in result}

    pos = scores.get("happy", 0) + 0.5 * scores.get("surprise", 0)
    neg = (scores.get("sad", 0) + scores.get("angry", 0)
           + scores.get("fear", 0) + scores.get("disgust", 0))
    neu = scores.get("neutral", 0)

    total = pos + neg + neu + 1e-9
    return float(np.clip(pos / total * 0.85 + neu / total * 0.5 + neg / total * 0.15, 0.05, 0.95))


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        users = json.load(f)

    n = len(users)
    print(f"Пользователей: {n}", file=sys.stderr, flush=True)

    print(f"Скачиваем аватары ({DOWNLOAD_THREADS} потоков)...", file=sys.stderr, flush=True)
    t0 = time.time()
    images: list[Image.Image | None] = [None] * n
    urls = [(i, (u.get("photo_200") or "").strip()) for i, u in enumerate(users)]

    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        futures = {pool.submit(download_avatar, url): idx for idx, url in urls}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                images[idx] = future.result()
            except Exception:
                pass
            done += 1
            if done % 1000 == 0 or done == n:
                print(f"  Скачано: {done}/{n} ({time.time()-t0:.0f}с)",
                      file=sys.stderr, flush=True)

    downloaded = sum(1 for img in images if img is not None)
    print(f"Скачано аватаров: {downloaded}/{n} ({time.time()-t0:.0f}с)",
          file=sys.stderr, flush=True)

    print(f"Загружаем модель {MODEL_NAME}...", file=sys.stderr, flush=True)

    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Устройство: {device}", file=sys.stderr, flush=True)

    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=device,
        top_k=7,
    )

    SCORES_CACHE = ROOT / "data" / "s5_scores_cache.json"
    scores = [None] * n
    if SCORES_CACHE.is_file():
        try:
            with open(SCORES_CACHE, encoding="utf-8") as fc:
                cached = json.load(fc)
            for i, v in enumerate(cached):
                if i < n and v is not None:
                    scores[i] = v
            cached_count = sum(1 for s in scores if s is not None)
            print(f"Загружено из кэша: {cached_count} скоров", file=sys.stderr, flush=True)
        except Exception:
            pass

    batch_size = 32
    MAX_PHOTOS = int(os.environ.get("S5_MAX_PHOTOS", 0)) or None
    valid_indices = [i for i in range(n) if images[i] is not None and scores[i] is None]
    if MAX_PHOTOS is not None:
        already_done_count = sum(1 for s in scores if s is not None)
        remaining_to_do = max(0, MAX_PHOTOS - already_done_count)
        if remaining_to_do < len(valid_indices):
            valid_indices = valid_indices[:remaining_to_do]
            print(f"Лимит: обработаем ещё {remaining_to_do} фото (до {MAX_PHOTOS} всего)",
                  file=sys.stderr, flush=True)

    already_done = sum(1 for s in scores if s is not None)
    print(f"Инференс: {len(valid_indices)} новых изображений (уже готово: {already_done}), батч={batch_size}...",
          file=sys.stderr, flush=True)
    t1 = time.time()

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_idx = valid_indices[batch_start:batch_start + batch_size]
        batch_images = [images[i] for i in batch_idx]

        try:
            results = classifier(batch_images)
            for idx, result in zip(batch_idx, results):
                scores[idx] = emotions_to_score(result)
        except Exception as e:
            for idx in batch_idx:
                try:
                    result = classifier(images[idx])
                    scores[idx] = emotions_to_score(result[0] if isinstance(result, list) and isinstance(result[0], list) else result)
                except Exception:
                    pass

        processed = min(batch_start + batch_size, len(valid_indices))

        if processed % 200 < batch_size or processed == len(valid_indices):
            with open(str(SCORES_CACHE) + ".tmp", "w") as fc:
                json.dump(scores, fc)
            os.replace(str(SCORES_CACHE) + ".tmp", str(SCORES_CACHE))

        if processed % 100 < batch_size or processed == len(valid_indices):
            elapsed = time.time() - t1
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(valid_indices) - processed) / rate if rate > 0 else 0
            total_done = already_done + processed
            print(f"  [{total_done}/{n}] {rate:.0f} фото/с, ETA: {eta:.0f}с",
                  file=sys.stderr, flush=True)

    valid_scores = [s for s in scores if s is not None]
    print(f"\nS5 рассчитан: {len(valid_scores)}/{n}",
          file=sys.stderr, flush=True)
    if valid_scores:
        print(f"  Среднее: {np.mean(valid_scores):.4f}, "
              f"мин: {min(valid_scores):.4f}, макс: {max(valid_scores):.4f}",
              file=sys.stderr, flush=True)

    from src.config import (
        WEIGHT_SOCIAL, WEIGHT_ACTIVITY, WEIGHT_SENTIMENT_MERGED,
        WEIGHT_GROUPS, WEIGHT_VISUAL, WEIGHT_MUSIC,
        PERCENTILE_LOW, PERCENTILE_HIGH,
    )

    median_score = float(np.median(valid_scores)) if valid_scores else 0.5
    missing = sum(1 for s in scores if s is None)
    if missing > 0:
        print(f"  Подставляем медиану ({median_score:.4f}) для {missing} необработанных",
              file=sys.stderr, flush=True)

    raw_visual = np.array([s if s is not None else median_score for s in scores])
    lo = np.percentile(raw_visual, PERCENTILE_LOW)
    hi = np.percentile(raw_visual, PERCENTILE_HIGH)
    if hi > lo:
        s5_norm = np.clip((np.clip(raw_visual, lo, hi) - lo) / (hi - lo), 0.0, 1.0)
    else:
        s5_norm = np.full(n, 0.5)

    wsum = (WEIGHT_SOCIAL + WEIGHT_ACTIVITY + WEIGHT_SENTIMENT_MERGED
            + WEIGHT_GROUPS + WEIGHT_VISUAL + WEIGHT_MUSIC)

    for i, u in enumerate(users):
        comp = u.get("components", {})
        comp["visual_emotion_01"] = round(float(s5_norm[i]), 4)
        u["components"] = comp

        H = (
            WEIGHT_SOCIAL * comp.get("social_01", 0.5)
            + WEIGHT_ACTIVITY * comp.get("activity_01", 0.5)
            + WEIGHT_SENTIMENT_MERGED * comp.get("sentiment_merged_01", 0.5)
            + WEIGHT_GROUPS * comp.get("groups_theme_01", 0.5)
            + WEIGHT_VISUAL * s5_norm[i]
            + WEIGHT_MUSIC * comp.get("music_mood_01", 0.5)
        ) / wsum
        u["happiness_index"] = round(float(H), 4)

    tmp = str(DATA_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False)
    os.replace(tmp, str(DATA_PATH))

    total_time = time.time() - t0
    indices = [u["happiness_index"] for u in users]
    print(f"\nГотово за {total_time:.0f}с ({total_time/60:.1f} мин)!", file=sys.stderr, flush=True)
    print(f"  Обработано фото: {len(valid_scores)}/{n} ({len(valid_scores)/n*100:.0f}%)",
          file=sys.stderr, flush=True)
    print(f"  H: среднее={np.mean(indices):.4f}, мин={min(indices):.4f}, макс={max(indices):.4f}",
          file=sys.stderr, flush=True)
    print(f"  S5: среднее={np.mean(s5_norm):.4f}, std={np.std(s5_norm):.4f}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
