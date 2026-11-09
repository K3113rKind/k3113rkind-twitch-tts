"""Tempo über Pausen, Spurbau der YouTube-Vertonung, Untertitel zu Sätzen."""
import json

import numpy as np

from app import klang, youtube_dub, youtube_subs

SR = 24000


def _speech_with_pauses(n=6):
    parts = []
    for _ in range(n):
        t = np.arange(int(0.6 * SR)) / SR
        parts.append((0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
        parts.append(np.zeros(int(0.12 * SR), np.float32) + 1e-4)
    return np.concatenate(parts)


def test_pause_tempo_hits_target():
    x = _speech_with_pauses()
    y = klang.apply_tempo(x, SR, 0.9, "pausen")
    assert abs(len(y) - len(x) / 0.9) < 0.02 * SR


def test_pause_tempo_keeps_min_pause_when_faster():
    x = _speech_with_pauses()
    y = klang.apply_tempo(x, SR, 1.5, "pausen")
    assert len(y) < len(x)
    assert len(y) >= len(x) - 5 * (0.12 - klang.KEEP_PAUSE_S) * SR - SR * 0.02


def test_build_track_places_and_delays():
    sents = [{"t": 0.0}, {"t": 1.0}, {"t": 10.0}]
    aud = [(np.full(int(0.5 * SR), 0.3, np.float32), SR)] * 3
    track, stats = youtube_dub.build_track(sents, aud, 12)
    assert abs(len(track) / SR - 12.5) < 0.1
    assert stats["max_delay"] == 0.0


def test_subtitles_to_sentences():
    ev = {"events": [
        {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Hallo zusammen, heute"}]},
        {"tStartMs": 2000, "dDurationMs": 2000, "segs": [{"utf8": "schauen wir z. B. die"}]},
        {"tStartMs": 4000, "dDurationMs": 1500, "segs": [{"utf8": "Karte an."}]},
        {"tStartMs": 9000, "dDurationMs": 2000, "segs": [{"utf8": "[Musik]"}]},
        {"tStartMs": 12000, "dDurationMs": 2000, "segs": [{"utf8": "Am 3. Mai Neues!"}]},
    ]}
    s = youtube_subs._sentences(youtube_subs._cues_json3(json.dumps(ev)))
    assert [x["text"] for x in s] == ["Hallo zusammen, heute schauen wir z. B. die Karte an.",
                                      "Am 3. Mai Neues!"]
    assert s[1]["t"] == 12.0
