#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import time
from datetime import datetime

from src.config import CITY_IDS_TATARSTAN
from src.vk_client import age_from_bdate, call_with_backoff, city_title_en

SAVE_EVERY = 100

AGE_RANGES = [
    (14, 18),
    (19, 22),
    (23, 27),
    (28, 33),
    (34, 40),
    (41, 50),
    (51, 65),
    (66, 80),
]

SEX_VALUES = [1, 2]


def _ts_to_str(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%d.%m.%Y %H:%M:%S")


def _save_json(records: list[dict], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    os.replace(tmp, path)


def search_user_ids_segmented(
    token: str, city_id: int, need: int, seen: set[int]
) -> list[int]:
    out: list[int] = []
    for age_from, age_to in AGE_RANGES:
        for sex in SEX_VALUES:
            if len(out) >= need:
                return out[:need]
            offset = 0
            empty_pages = 0
            while len(out) < need and offset < 1000:
                chunk = min(1000, need - len(out) + 50)
                try:
                    resp = call_with_backoff(
                        "users.search",
                        token,
                        {
                            "city": city_id,
                            "country": 1,
                            "count": chunk,
                            "offset": offset,
                            "fields": "city",
                            "age_from": age_from,
                            "age_to": age_to,
                            "sex": sex,
                        },
                    )
                except RuntimeError:
                    break
                if isinstance(resp, list):
                    items = resp
                else:
                    items = resp.get("items", [])
                if not items:
                    break
                new_in_page = 0
                for u in items:
                    uid = u.get("id")
                    if uid and uid not in seen:
                        seen.add(uid)
                        out.append(uid)
                        new_in_page += 1
                        if len(out) >= need:
                            break
                offset += len(items)
                if new_in_page == 0:
                    empty_pages += 1
                    if empty_pages >= 2:
                        break
                else:
                    empty_pages = 0
                time.sleep(0.34)
    return out[:need]


def fetch_user_record(token: str, user_id: int) -> dict | None:
    fields = (
        "bdate,city,sex,photo_200,domain,followers_count,counters,"
        "can_access_closed,is_closed,universities,schools"
    )
    users = call_with_backoff("users.get", token, {"user_ids": user_id, "fields": fields})
    if not users:
        return None
    u = users[0]
    if u.get("is_closed") and not u.get("can_access_closed"):
        return None

    city_obj = u.get("city") or {}
    city_id = city_obj.get("id")
    city_name = city_title_en(city_id, city_obj.get("title"))

    counters = u.get("counters") or {}
    sex = u.get("sex")
    gender = "female" if sex == 1 else "male" if sex == 2 else "unknown"

    uni = ""
    unis = u.get("universities") or []
    if unis:
        uni = (unis[0].get("name") or "")[:200]

    base = {
        "id": str(u.get("id")),
        "city": city_name,
        "age": age_from_bdate(u.get("bdate")),
        "gender": gender,
        "university": uni,
        "photos_num": int(counters.get("albums") or 0),
        "videos_num": int(counters.get("videos") or 0),
        "audio_num": int(counters.get("audios") or 0),
        "self_posts_num": 0,
        "reposts_num": 0,
        "followers_num": int(u.get("followers_count") or counters.get("followers") or 0),
        "groups_num": 0,
        "friends_num": int(counters.get("friends") or 0),
        "likes_received_num": 0,
        "comments_received_num": 0,
        "reposts_received_num": 0,
        "friends_to_followers_ratio": None,
        "photo_200": (u.get("photo_200") or "")[:500],
        "posts": {},
        "groups": {},
    }

    time.sleep(0.34)
    try:
        fr = call_with_backoff("friends.get", token, {"user_id": user_id, "count": 1})
        base["friends_num"] = int(fr.get("count", 0)) or len(fr.get("items", []))
    except RuntimeError:
        pass

    time.sleep(0.34)
    try:
        gr = call_with_backoff(
            "groups.get",
            token,
            {"user_id": user_id, "extended": 1, "fields": "description", "count": 200},
        )
        items = gr.get("items") or []
        base["groups_num"] = len(items)
        for g in items:
            gid = str(g.get("id"))
            base["groups"][gid] = {
                "name": (g.get("name") or "")[:300],
                "description": (g.get("description") or "")[:500],
            }
    except RuntimeError:
        pass

    time.sleep(0.34)
    wall_likes = 0
    wall_comments = 0
    reposts = 0
    self_posts = 0
    posts_map: dict[str, dict] = {}
    audio_attachments: list[dict] = []
    try:
        wall = call_with_backoff(
            "wall.get",
            token,
            {"owner_id": user_id, "count": 100, "filter": "owner"},
        )
        witems = wall.get("items") or []
        one_year_ago = int(time.time()) - 365 * 86400
        for p in witems:
            if int(p.get("date", 0)) < one_year_ago:
                continue
            pid = str(p.get("id"))
            txt = p.get("text") or ""
            posts_map[pid] = {"text": txt, "date": _ts_to_str(int(p.get("date", 0)))}
            self_posts += 1
            likes = (p.get("likes") or {}).get("count") or 0
            comments = (p.get("comments") or {}).get("count") or 0
            rep = (p.get("reposts") or {}).get("count") or 0
            wall_likes += int(likes)
            wall_comments += int(comments)
            reposts += int(rep)
            for att in p.get("attachments") or []:
                if att.get("type") == "audio":
                    audio = att.get("audio") or {}
                    artist = (audio.get("artist") or "").strip()
                    title = (audio.get("title") or "").strip()
                    if artist or title:
                        audio_attachments.append({
                            "artist": artist[:200],
                            "title": title[:200],
                        })
    except RuntimeError:
        pass

    base["self_posts_num"] = self_posts
    base["reposts_num"] = reposts
    base["likes_received_num"] = wall_likes
    base["comments_received_num"] = wall_comments
    base["reposts_received_num"] = reposts
    base["posts"] = posts_map
    base["audio_attachments"] = audio_attachments

    ff = base["followers_num"]
    if ff and ff > 0:
        base["friends_to_followers_ratio"] = round(base["friends_num"] / ff, 4)

    return base


def _load_existing(path: str) -> tuple[list[dict], set[int]]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        seen = {int(r["id"]) for r in records if r.get("id")}
        print(f"Resume: загружено {len(records)} ранее собранных профилей.", file=sys.stderr, flush=True)
        return records, seen
    except (json.JSONDecodeError, KeyError):
        return [], set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/data.json")
    ap.add_argument("--target", type=int, default=10000)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    token = os.environ.get("VK_TOKEN", "").strip()
    if not token:
        print("Задайте переменную окружения VK_TOKEN (access_token пользователя).", file=sys.stderr)
        sys.exit(1)

    if args.no_resume:
        records: list[dict] = []
        seen: set[int] = set()
    else:
        records, seen = _load_existing(out_path)

    if len(records) >= args.target:
        print(f"Уже собрано {len(records)} >= {args.target}. Нечего делать.", file=sys.stderr)
        return

    remaining = args.target - len(records)
    per_city = max(remaining // len(CITY_IDS_TATARSTAN) + 100, 200)

    print(
        f"Цель: {args.target} пользователей (уже есть: {len(records)}, осталось: {remaining}).\n"
        f"Поиск id: {len(CITY_IDS_TATARSTAN)} городов × {len(AGE_RANGES)} возрастных групп × 2 пола...",
        file=sys.stderr,
        flush=True,
    )

    ids: list[int] = []
    for cid in CITY_IDS_TATARSTAN:
        part = search_user_ids_segmented(token, cid, per_city, seen)
        ids.extend(part)
        print(
            f"  город {cid}: +{len(part)} id, всего новых id: {len(ids)}",
            file=sys.stderr, flush=True,
        )
        if len(ids) >= remaining:
            break

    ids = ids[:remaining]
    print(
        f"Найдено {len(ids)} новых id. Загрузка профилей (~4 запроса/чел, "
        f"~{len(ids) * 1.4 / 60:.0f} мин)...",
        file=sys.stderr,
        flush=True,
    )

    fetched_in_session = 0
    skipped = 0
    t_start = time.time()

    for i, uid in enumerate(ids, start=1):
        try:
            rec = fetch_user_record(token, uid)
            if rec:
                records.append(rec)
                fetched_in_session += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"  skip {uid}: {e}", file=sys.stderr)

        if i == 1 or i % 50 == 0 or i == len(ids):
            elapsed = time.time() - t_start
            rate = fetched_in_session / elapsed * 3600 if elapsed > 0 else 0
            eta_h = (len(ids) - i) / (fetched_in_session / elapsed) / 3600 if fetched_in_session > 0 else 0
            print(
                f"  [{i}/{len(ids)}] собрано: {len(records)} | "
                f"пропущено: {skipped} | "
                f"скорость: {rate:.0f}/час | "
                f"ETA: {eta_h:.1f}ч",
                file=sys.stderr,
                flush=True,
            )

        if fetched_in_session > 0 and fetched_in_session % SAVE_EVERY == 0:
            _save_json(records, out_path)
            print(f"  [checkpoint] сохранено {len(records)} записей.", file=sys.stderr, flush=True)

        time.sleep(0.15)

        if len(records) >= args.target:
            break

    _save_json(records, out_path)
    elapsed_total = time.time() - t_start
    print(
        f"\nГотово: {out_path}\n"
        f"  Всего записей: {len(records)}\n"
        f"  Собрано в этой сессии: {fetched_in_session}\n"
        f"  Пропущено (закрытые/ошибки): {skipped}\n"
        f"  Время: {elapsed_total / 60:.1f} мин",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
