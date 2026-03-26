from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fpdf import FPDF

from src.config import (
    WEIGHT_ACTIVITY,
    WEIGHT_GROUPS,
    WEIGHT_MUSIC,
    WEIGHT_SENTIMENT_MERGED,
    WEIGHT_SOCIAL,
    WEIGHT_VISUAL,
)
from src.plots import sorted_figure_filenames
from src.report_stats import insight_paragraphs, stats_paragraphs


def _font_path() -> str:
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if p.is_file():
            return str(p)
    except Exception:
        pass
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError("Не найден TTF-шрифт для кириллицы (нужен DejaVu или Arial Unicode).")


def build_pdf_report(
    users: list[dict[str, Any]],
    figures_dir: str,
    out_pdf: str,
    project_root: Path,
) -> None:
    font = _font_path()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("RU", "", font)
    pdf.add_font("RU", "B", font)
    pdf.set_font("RU", size=11)
    pdf.add_page()

    def p(txt: str) -> None:
        if not txt.strip():
            pdf.ln(4)
            return
        pdf.multi_cell(pdf.epw, 6, txt)
    pdf.set_font("RU", "B", 16)
    pdf.cell(0, 10, "Отчёт: индекс цифрового благополучия (VK, РТ)", ln=True)
    pdf.set_font("RU", size=11)

    p(
        "1. Этический протокол. Используются только открытые данные VK API: публичные счётчики, "
        "стена с публичным доступом, открытые группы, список друзей там, где он доступен. "
        "Персональные сообщения и закрытые профили не запрашиваются. Результаты носят "
        "агрегированный характер; идентификаторы в учебном корпусе обезличивать при публикации "
        "рекомендуется (в выгрузке для платформы — по требованию задания)."
    )
    p("")

    p(
        "2. Сбор данных. Пользователи отбирались по городам Республики Татарстан (идентификаторы "
        "городов VK). Для каждого профиля сохранялись счётчики, друзья, группы (название и описание), "
        "посты на стене за последний год (текст и дата), агрегированные лайки и комментарии к постам."
    )
    p("")

    p(
        "3. Предобработка. Тексты постов очищаются от пустых строк; тональность оценивается ансамблем: "
        "Dostoevsky (FastText), RuBERT-эмбеддинги с якорными фразами (transformers), при наличии пакета — "
        "DeepPavlov при возможности, иначе запасной RuBERT из HuggingFace (тот же слот в формуле). "
        "По URL аватара: ViT-модель (trpakov/vit-face-expression) для распознавания 7 классов эмоций "
        "(happy, sad, angry, fear, disgust, surprise, neutral); для необработанных аватаров подставляется медиана. "
        "Музыкальное настроение (S6): анализ названий музыкальных сообществ в подписках по ключевым словам "
        "(позитивные/негативные жанры и настроения) + счётчик аудиозаписей как косвенный индикатор. "
        "Компоненты объёма связей и активности: log1p и робастная нормализация (5–95 перцентили); при константном столбце — нейтраль 0.5."
    )
    p("")

    pdf.set_font("RU", "B", 12)
    pdf.cell(0, 8, "4. Математическая модель", ln=True)
    pdf.set_font("RU", size=11)
    p(
        "Сырые признаки: S1* = log(1+friends)+log(1+followers)+log(1+groups); "
        "S2* = log(1+posts)+log(1+likes)+log(1+comments); S3* — ансамбль тональности (нормируется по выборке); "
        "S4 — тематика групп; S5* — эмоции по аватару (ViT-модель trpakov/vit-face-expression); "
        "S6* — музыкальное настроение (анализ музыкальных групп в подписках и счётчика аудиозаписей)."
    )
    p(
        "Нормализация: S1 = RobScale(S1*), S2 = RobScale(S2*), компоненты S3–S6 приведены к [0,1]. "
        f"Итог: H = {WEIGHT_SOCIAL}·S1 + {WEIGHT_ACTIVITY}·S2 + {WEIGHT_SENTIMENT_MERGED}·S3 + "
        f"{WEIGHT_GROUPS}·S4 + {WEIGHT_VISUAL}·S5 + {WEIGHT_MUSIC}·S6."
    )
    p("")

    n = len(users)
    hi = [float(u["happiness_index"]) for u in users if u.get("happiness_index") is not None]
    mean_h = sum(hi) / len(hi) if hi else float("nan")
    pdf.set_font("RU", "B", 12)
    pdf.cell(0, 8, "5. Распределения и сводная статистика", ln=True)
    pdf.set_font("RU", size=11)
    p(
        f"Выборка: n={n} пользователей. Среднее значение индекса H ≈ {mean_h:.3f} (шкала [0,1]). "
        "Ниже — описательные показатели по H и по каждой компоненте после нормировки; для подмоделей "
        "тональности n может быть меньше, если RuBERT/DeepPavlov не использовались."
    )
    p("")
    for line in stats_paragraphs(users):
        p(line)
    p("")

    pdf.set_font("RU", "B", 12)
    pdf.cell(0, 8, "6. Краткие выводы по графикам и компонентам", ln=True)
    pdf.set_font("RU", size=11)
    for line in insight_paragraphs(users):
        p(line)
    p("")
    p(
        "Ограничения: закрытые аккаунты недоступны; тональность по тексту не равна эмоциям в жизни; "
        "аватар не отражает реальное настроение; VK Audio API закрыт с 2017 года, поэтому аудио-вложения "
        "из постов недоступны — S6 опирается на музыкальные сообщества и счётчик аудиозаписей. "
        "64% пользователей не имеют публичных постов (закрытая стена), для них S3 = 0.5 (нейтраль). "
        "Рисунки fig1–fig9 показывают распределения индекса и компонент, сравнение по полу и возрасту, "
        "вклад моделей тональности, статистику компонент, матрицу корреляций и связь H с тональностью/визуалом."
    )

    for name in sorted_figure_filenames(figures_dir):
        pdf.add_page()
        pdf.set_font("RU", "B", 12)
        pdf.cell(0, 8, f"Рис. {name}", ln=True)
        pdf.set_font("RU", size=11)
        img_path = os.path.join(figures_dir, name)
        pdf.image(img_path, w=180)

    pdf.add_page()
    pdf.set_font("RU", "B", 12)
    pdf.cell(0, 8, "Приложение: фрагменты исходного кода", ln=True)
    pdf.set_font("RU", size=7)
    for rel in (
        "src/config.py",
        "src/vk_client.py",
        "src/collect_vk.py",
        "src/happiness_index.py",
        "src/rubert_sentiment.py",
        "src/deeppavlov_tone.py",
        "src/image_emotion.py",
        "src/music_mood.py",
        "src/report_stats.py",
        "src/plots.py",
        "run_analysis.py",
    ):
        path = project_root / rel
        if not path.is_file():
            continue
        pdf.set_font("RU", "B", 9)
        pdf.cell(0, 5, rel, ln=True)
        pdf.set_font("RU", size=6)
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 12000:
            text = text[:12000] + "\n... [обрезано для PDF] ..."
        for line in text.splitlines():
            pdf.multi_cell(pdf.epw, 3, line[:120])
        pdf.ln(2)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    pdf.output(out_pdf)
