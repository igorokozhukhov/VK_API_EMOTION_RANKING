from __future__ import annotations

import math
from typing import Any


def _nums(users: list[dict[str, Any]], path: str) -> list[float]:
    out: list[float] = []
    for u in users:
        if path == "happiness_index":
            v = u.get("happiness_index")
        else:
            v = (u.get("components") or {}).get(path)
        if v is None:
            continue
        try:
            x = float(v)
            if not math.isnan(x):
                out.append(x)
        except (TypeError, ValueError):
            continue
    return out


def _line(name: str, vals: list[float]) -> str:
    if not vals:
        return f"{name}: нет числовых данных."
    a = vals
    n = len(a)
    mu = sum(a) / n
    var = sum((x - mu) ** 2 for x in a) / max(n - 1, 1)
    sd = math.sqrt(var)
    mn, mx = min(a), max(a)
    sa = sorted(a)
    p25 = sa[n // 4] if n else 0.0
    p75 = sa[(3 * n) // 4] if n else 0.0
    return (
        f"{name}: n={n}, среднее={mu:.3f}, σ={sd:.3f}, min={mn:.3f}, max={mx:.3f}, "
        f"p25={p25:.3f}, p75={p75:.3f}"
    )


def stats_paragraphs(users: list[dict[str, Any]]) -> list[str]:
    blocks = [
        ("Итоговый индекс H", "happiness_index"),
        ("S1 социальный объём (норм.)", "social_01"),
        ("S2 активность (норм.)", "activity_01"),
        ("S3 тональность, ансамбль (норм.)", "sentiment_merged_01"),
        ("  — Dostoevsky", "sentiment_dostoevsky_01"),
        ("  — RuBERT", "sentiment_rubert_01"),
        ("  — DeepPavlov / запасной HF RuBERT", "sentiment_deeppavlov_01"),
        ("S4 тематика групп (норм.)", "groups_theme_01"),
        ("S5 эмоции по аватару (норм.)", "visual_emotion_01"),
        ("S6 музыкальное настроение (норм.)", "music_mood_01"),
    ]
    return [_line(lbl, _nums(users, key)) for lbl, key in blocks]


def insight_paragraphs(users: list[dict[str, Any]]) -> list[str]:
    h = _nums(users, "happiness_index")
    if len(h) < 3:
        return ["Выборка слишком мала для устойчивых выводов."]

    comp_keys = [
        ("социальный объём S1", "social_01"),
        ("активность S2", "activity_01"),
        ("тональность S3", "sentiment_merged_01"),
        ("тематика групп S4", "groups_theme_01"),
        ("визуал S5", "visual_emotion_01"),
        ("музыка S6", "music_mood_01"),
    ]
    means: list[tuple[str, float]] = []
    for name, key in comp_keys:
        v = _nums(users, key)
        if v:
            means.append((name, sum(v) / len(v)))
    means.sort(key=lambda x: -x[1])
    if not means:
        return ["Недостаточно данных по компонентам для выводов."]
    top = ", ".join(f"{a} ({b:.2f})" for a, b in means[:2])
    low = ", ".join(f"{a} ({b:.2f})" for a, b in means[-2:]) if len(means) >= 2 else ""

    def corr(x: list[float], y: list[float]) -> float | None:
        if len(x) != len(y) or len(x) < 3:
            return None
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if dx < 1e-12 or dy < 1e-12:
            return None
        return num / (dx * dy)

    best: tuple[str, float] | None = None
    for name, key in comp_keys:
        xs, ys = [], []
        for u in users:
            hi = u.get("happiness_index")
            cv = (u.get("components") or {}).get(key)
            if hi is None or cv is None:
                continue
            try:
                xf, yf = float(cv), float(hi)
                if not (math.isnan(xf) or math.isnan(yf)):
                    xs.append(xf)
                    ys.append(yf)
            except (TypeError, ValueError):
                continue
        r = corr(xs, ys)
        if r is not None and (best is None or abs(r) > abs(best[1])):
            best = (name, r)

    lines = [
        f"По средним нормированным значениям компонент выше всего: {top}. "
        f"Ниже остальных (в среднем по выборке): {low}." if low else f"Средние по компонентам: {top}.",
    ]
    if best:
        lines.append(
            f"Наибольшая по модулю линейная связь с итоговым индексом H: «{best[0]}» "
            f"(коэффициент корреляции Пирсона r ≈ {best[1]:.3f}). Направление связи учитывать "
            f"вместе со знаком r."
        )
    lines.append(
        "Ансамбль тональности объединяет Dostoevsky, RuBERT и DeepPavlov; если часть моделей "
        "отключена, S3 опирается только на доступные источники — см. гистограммы fig6."
    )
    lines.append(
        "Визуальный компонент S5 шумный (один кадр аватара); низкая корреляция с H не означает "
        "бесполезность блока, а отражает слабый сигнал относительно текстов."
    )
    return lines
