#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.json")
    ap.add_argument("--output", default="data/data.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--pdf", default="report_task2.pdf")
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    if args.skip_images:
        os.environ["SKIP_PROFILE_IMAGE_EMOTION"] = "1"

    from src.build_report import build_pdf_report
    from src.happiness_index import compute_happiness_for_corpus
    from src.plots import make_all_plots

    data_path = ROOT / args.data
    out_path = ROOT / args.output
    fig_dir = ROOT / args.figures
    pdf_path = ROOT / args.pdf

    with open(data_path, encoding="utf-8") as f:
        users = json.load(f)

    users = compute_happiness_for_corpus(users)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

    fig_dir.mkdir(parents=True, exist_ok=True)
    make_all_plots(users, str(fig_dir))
    build_pdf_report(users, str(fig_dir), str(pdf_path), ROOT)
    print(f"Готово: {out_path}, {pdf_path}")


if __name__ == "__main__":
    main()
