from __future__ import annotations

import time
from typing import Any

import requests

API_VER = "5.199"
BASE = "https://api.vk.com/method"


def call(method: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {"access_token": token, "v": API_VER}
    if params:
        p.update(params)
    r = requests.get(f"{BASE}/{method}", params=p, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"VK API error: {data['error']}")
    return data["response"]


def call_with_backoff(method: str, token: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    delay = 0.35
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return call(method, token, params)
        except RuntimeError as e:
            last = e
            err = str(e)
            if "Too many requests" in err or "6" in err:
                time.sleep(1.0 + attempt)
                continue
            raise
        except requests.RequestException as e:
            last = e
            time.sleep(delay * (attempt + 1))
    if last:
        raise last
    raise RuntimeError("VK call failed")


def city_title_en(city_id: int | None, title_ru: str | None) -> str:
    mapping = {
        73: "Kazan",
        204: "Naberezhnye Chelny",
        99: "Almetyevsk",
        1192: "Nizhnekamsk",
        1187: "Zelenodolsk",
        11168: "Yelabuga",
    }
    if city_id and city_id in mapping:
        return mapping[city_id]
    if title_ru:
        return title_ru
    return "unknown"


def age_from_bdate(bdate: str | None) -> int | None:
    if not bdate:
        return None
    parts = bdate.split(".")
    if len(parts) != 3:
        return None
    try:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    from datetime import date

    today = date.today()
    age = today.year - year - ((today.month, today.day) < (month, day))
    return max(0, age) if 0 <= age <= 120 else None
