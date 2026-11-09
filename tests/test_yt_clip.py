import json

from app.youtube_subs import _cues_json3, clip_cues


def test_json3_split_at_cut():
    raw = json.dumps({"events": [
        {"tStartMs": 85000, "dDurationMs": 8000, "segs": [
            {"utf8": "before", "tOffsetMs": 0}, {"utf8": " cut", "tOffsetMs": 3000},
            {"utf8": " inside", "tOffsetMs": 5500}, {"utf8": " more", "tOffsetMs": 7000}]},
        {"tStartMs": 93000, "dDurationMs": 2000, "segs": [{"utf8": "next"}]},
    ]})
    cues = clip_cues(_cues_json3(raw, cut=90.0), 90.0, None)
    assert [t for _, _, t in cues] == ["inside more", "next"]
    assert abs(cues[0][0] - 0.5) < 1e-6 and abs(cues[1][0] - 3.0) < 1e-6


def test_clip_keeps_mostly_inside_block():
    cues = [(88.0, 96.0, "mostly inside"), (80.0, 91.0, "mostly before"), (120.0, 125.0, "after end")]
    got = clip_cues(cues, 90.0, 110.0)
    assert [t for _, _, t in got] == ["mostly inside"]
    assert got[0][0] == 0.0 and got[0][1] == 6.0
