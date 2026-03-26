from __future__ import annotations

from typing import Any

from src.config import (
    MUSIC_GROUP_KEYWORDS,
    MUSIC_NEGATIVE_KEYWORDS,
    MUSIC_POSITIVE_KEYWORDS,
)


def _classify_track(artist: str, title: str) -> float:
    blob = f"{artist} {title}".lower()
    pos = sum(1 for kw in MUSIC_POSITIVE_KEYWORDS if kw in blob)
    neg = sum(1 for kw in MUSIC_NEGATIVE_KEYWORDS if kw in blob)
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


def _music_groups_score(groups: dict[str, Any]) -> tuple[float, int]:
    if not groups:
        return 0.5, 0

    music_scores: list[float] = []
    for g in groups.values():
        name = (g.get("name") or "").lower()
        desc = (g.get("description") or "").lower()
        blob = f"{name} {desc}"

        is_music = any(kw in blob for kw in MUSIC_GROUP_KEYWORDS)
        if not is_music:
            continue

        pos = sum(1 for kw in MUSIC_POSITIVE_KEYWORDS if kw in blob)
        neg = sum(1 for kw in MUSIC_NEGATIVE_KEYWORDS if kw in blob)
        if pos + neg == 0:
            music_scores.append(0.5)
        else:
            music_scores.append(pos / (pos + neg))

    if not music_scores:
        return 0.5, 0
    return sum(music_scores) / len(music_scores), len(music_scores)


def compute_music_mood(user: dict[str, Any]) -> float:
    scores: list[float] = []
    weights: list[float] = []

    audio_tracks = user.get("audio_attachments") or []
    if audio_tracks:
        track_scores = [
            _classify_track(t.get("artist", ""), t.get("title", ""))
            for t in audio_tracks
        ]
        tracks_mean = sum(track_scores) / len(track_scores)
        w = min(len(track_scores) / 5.0, 3.0)
        scores.append(tracks_mean)
        weights.append(w)

    groups = user.get("groups") or {}
    group_score, group_count = _music_groups_score(groups)
    if group_count > 0:
        w = min(group_count / 3.0, 2.0)
        scores.append(group_score)
        weights.append(w)

    audio_num = int(user.get("audio_num") or 0)
    if audio_num > 0:
        import math
        audio_signal = 0.5 + 0.1 * (1 - math.exp(-audio_num / 50.0))
        scores.append(audio_signal)
        weights.append(0.5)

    if not scores:
        return 0.5

    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
