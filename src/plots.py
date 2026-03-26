from __future__ import annotations

import os
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _components_frame(users: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for u in users:
        row: dict[str, Any] = {"happiness_index": u.get("happiness_index")}
        c = u.get("components") or {}
        for k in (
            "social_01",
            "activity_01",
            "sentiment_merged_01",
            "sentiment_dostoevsky_01",
            "sentiment_rubert_01",
            "sentiment_deeppavlov_01",
            "groups_theme_01",
            "visual_emotion_01",
            "music_mood_01",
        ):
            row[k] = c.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def make_all_plots(users: list[dict[str, Any]], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    df = pd.DataFrame(users)
    if "happiness_index" not in df.columns:
        raise ValueError("Ожидается поле happiness_index — сначала запустите расчёт индекса")

    cdf = _components_frame(users)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["happiness_index"].dropna(), bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Индекс цифрового благополучия")
    ax.set_ylabel("Число пользователей")
    ax.set_title("Распределение индекса по выборке")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig1_happiness_dist.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    if "gender" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for g, sub in df.groupby("gender"):
            ax.hist(
                sub["happiness_index"].dropna(),
                bins=15,
                alpha=0.5,
                label=str(g),
                density=True,
            )
        ax.set_xlabel("Индекс")
        ax.set_ylabel("Плотность")
        ax.legend()
        ax.set_title("Распределение индекса по полу")
        fig.tight_layout()
        p2 = os.path.join(out_dir, "fig2_by_gender.png")
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        paths.append(p2)

    if "age" in df.columns:
        df2 = df.dropna(subset=["age"]).copy()
        df2["age_bucket"] = pd.cut(
            df2["age"].astype(float),
            bins=[0, 22, 30, 45, 100],
            labels=["≤22", "23–30", "31–45", "46+"],
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        df2.groupby("age_bucket", observed=True)["happiness_index"].mean().plot(kind="bar", ax=ax, color="teal")
        ax.set_xlabel("Возрастная группа")
        ax.set_ylabel("Средний индекс")
        ax.set_title("Средний индекс по возрастным группам")
        fig.tight_layout()
        p3 = os.path.join(out_dir, "fig3_by_age.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        paths.append(p3)

    act = (
        df.get("self_posts_num", 0).fillna(0)
        + df.get("likes_received_num", 0).fillna(0) / 50.0
    )
    act_threshold = float(np.percentile(act, 99))
    mask = act <= act_threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(act[mask], df.loc[mask, "happiness_index"], alpha=0.35, s=20, c="darkgreen")
    ax.set_xlabel("Условная активность (посты + лайки/50)")
    ax.set_ylabel("Индекс")
    ax.set_title("Связь индекса с активностью")
    fig.tight_layout()
    p4 = os.path.join(out_dir, "fig4_activity_vs_index.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths.append(p4)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()
    pairs = [
        ("happiness_index", "Итоговый индекс H"),
        ("social_01", "Соц. объём S1"),
        ("activity_01", "Активность S2"),
        ("sentiment_merged_01", "Тональность (ансамбль) S3"),
        ("groups_theme_01", "Тематика групп S4"),
        ("visual_emotion_01", "Эмоции по аватару S5"),
    ]
    for ax, (col, title) in zip(axes, pairs):
        s = cdf[col].dropna() if col in cdf.columns else pd.Series(dtype=float)
        if col == "happiness_index":
            s = df["happiness_index"].dropna()
        if len(s) > 0:
            ax.hist(s.astype(float), bins=18, color="slategray", edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Значение [0,1]")
        ax.set_ylabel("N")
    fig.suptitle("Распределения итогового индекса и компонент формулы", fontsize=12, y=1.02)
    fig.tight_layout()
    p5 = os.path.join(out_dir, "fig5_components_distributions.png")
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p5)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    src = [
        ("sentiment_dostoevsky_01", "Dostoevsky (FastText)"),
        ("sentiment_rubert_01", "RuBERT (эмбеддинги)"),
        ("sentiment_deeppavlov_01", "DeepPavlov"),
    ]
    for ax, (col, label) in zip(axes, src):
        s = pd.to_numeric(cdf[col], errors="coerce").dropna()
        if len(s) > 0:
            ax.hist(s, bins=16, color="indianred", edgecolor="white", alpha=0.85)
            ax.axvline(float(s.mean()), color="navy", linestyle="--", linewidth=1, label=f"μ={s.mean():.2f}")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "нет данных\n(модель не использовалась)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("[0,1]")
        ax.set_ylabel("N")
    fig.suptitle(
        "Тональность: Dostoevsky / RuBERT-эмбеддинги / DeepPavlov или запасной RuBERT (HuggingFace)",
        fontsize=10,
        y=1.06,
    )
    fig.tight_layout()
    p6 = os.path.join(out_dir, "fig6_sentiment_models.png")
    fig.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p6)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bar_cols = [
        "social_01",
        "activity_01",
        "sentiment_merged_01",
        "groups_theme_01",
        "visual_emotion_01",
        "music_mood_01",
    ]
    bar_colors = ["#0D7C66", "#5DB8A2", "#1B9E5A", "#E08A0A", "#7B2D8E", "#2E86C1"]
    labels = ["S1\nсоц.", "S2\nакт.", "S3\nтон.", "S4\nгруппы", "S5\nфото", "S6\nмузыка"]
    means = []
    medians = []
    stds = []
    used_labels = []
    used_colors = []
    for c, lbl, clr in zip(bar_cols, labels, bar_colors):
        if c in cdf.columns:
            vals = cdf[c].dropna().astype(float).values
            if len(vals) > 0:
                means.append(float(np.mean(vals)))
                medians.append(float(np.median(vals)))
                stds.append(float(np.std(vals)))
                used_labels.append(lbl)
                used_colors.append(clr)
    if means:
        x = np.arange(len(means))
        width = 0.35
        bars_mean = ax.bar(x - width / 2, means, width, color=used_colors,
                           alpha=0.85, label="Среднее", edgecolor="white", linewidth=0.8)
        bars_med = ax.bar(x + width / 2, medians, width, color=used_colors,
                          alpha=0.45, label="Медиана", edgecolor="white", linewidth=0.8,
                          hatch="//")
        ax.errorbar(x - width / 2, means, yerr=stds, fmt="none",
                    ecolor="#333333", elinewidth=1.2, capsize=4, capthick=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(used_labels, fontsize=9)
        for i, (m, md) in enumerate(zip(means, medians)):
            ax.text(i - width / 2, m + stds[i] + 0.02, f"{m:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
            ax.text(i + width / 2, md + 0.02, f"{md:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#666666")
        ax.legend(fontsize=9, loc="upper right")
    ax.set_ylabel("Нормированное значение [0, 1]")
    ax.set_title("Статистика компонент индекса: среднее, медиана и стандартное отклонение")
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    p7 = os.path.join(out_dir, "fig7_components_boxplot.png")
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    paths.append(p7)

    cols = [
        "happiness_index",
        "social_01",
        "activity_01",
        "sentiment_merged_01",
        "groups_theme_01",
        "visual_emotion_01",
    ]
    mat = cdf.copy()
    mat["happiness_index"] = df["happiness_index"].values
    sub = mat[[c for c in cols if c in mat.columns]].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson")
    cvals = corr.values.astype(float)
    plot_vals = np.nan_to_num(cvals, nan=0.0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(plot_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    short = ["H", "S1", "S2", "S3", "S4", "S5"]
    ax.set_xticklabels(short[: len(corr.columns)], rotation=45, ha="right")
    ax.set_yticklabels(short[: len(corr.columns)])
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            v = cvals[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    ax.set_title("Корреляции: индекс и компоненты формулы")
    fig.tight_layout()
    p8 = os.path.join(out_dir, "fig8_correlation_matrix.png")
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    paths.append(p8)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sm = pd.to_numeric(cdf["sentiment_merged_01"], errors="coerce")
    axes[0].scatter(sm, df["happiness_index"], alpha=0.3, s=18, c="purple")
    axes[0].set_xlabel("S3 тональность (ансамбль)")
    axes[0].set_ylabel("Индекс H")
    axes[0].set_title("H vs тональность текстов")
    ve = pd.to_numeric(cdf["visual_emotion_01"], errors="coerce")
    axes[1].scatter(ve, df["happiness_index"], alpha=0.3, s=18, c="darkorange")
    axes[1].set_xlabel("S5 эмоции по аватару")
    axes[1].set_ylabel("Индекс H")
    axes[1].set_title("H vs визуальный компонент")
    fig.suptitle("Связь итогового индекса с новыми блоками признаков", fontsize=11, y=1.02)
    fig.tight_layout()
    p9 = os.path.join(out_dir, "fig9_index_vs_new_features.png")
    fig.savefig(p9, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p9)

    return paths


def sorted_figure_filenames(figures_dir: str) -> list[str]:
    names = [f for f in os.listdir(figures_dir) if f.endswith(".png")]

    def sort_key(name: str) -> tuple[int, str]:
        m = re.match(r"fig(\d+)_", name)
        return (int(m.group(1)), name) if m else (9999, name)

    return sorted(names, key=sort_key)
