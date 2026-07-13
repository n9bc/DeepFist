import asyncio
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("rbn_harvest", ROOT / "tools" / "rbn_harvest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rbn_harvest"] = mod   # dataclass w/ `from __future__ import annotations`
    spec.loader.exec_module(mod)       # resolves string annotations via sys.modules
    return mod


RH = _load()


# --- Task 1: parser --------------------------------------------------------------
def test_parse_cw_spot():
    line = "DX de W3LPL-#:    14009.0  K1GHL          CW    18 dB  28 wpm  CQ      0012Z"
    s = RH.parse_rbn_line(line, now=100.0)
    assert s is not None
    assert s.spotter == "W3LPL"
    assert s.freq_khz == 14009.0
    assert s.call == "K1GHL"
    assert s.snr_db == 18
    assert s.wpm == 28
    assert s.t == 100.0


def test_parse_rejects_non_cw_and_garbage():
    assert RH.parse_rbn_line("DX de OH6BG-#:  14074.0  N5XYZ  FT8  10 dB  0 wpm  0012Z") is None
    assert RH.parse_rbn_line("random telnet banner text") is None
    assert RH.parse_rbn_line("") is None


# --- Task 2: tuning math + band filter -------------------------------------------
def test_dial_hz_sideband_offsets():
    assert RH.dial_hz_for(14009.0, "CWU", 650) == 14009000 - 650
    assert RH.dial_hz_for(7010.0, "CWL", 600) == 7010000 + 600


def test_range_parse_and_filter():
    r = [RH.parse_range("14000-14100"), RH.parse_range("7000-7040")]
    assert r[0] == (14000.0, 14100.0)
    assert RH.in_range(14009.0, r) is True
    assert RH.in_range(7035.0, r) is True
    assert RH.in_range(21020.0, r) is False
    assert RH.in_range(21020.0, []) is True


# --- Task 3: consensus gate ------------------------------------------------------
def _spot(spotter, call, freq, t, snr=10, wpm=25):
    return RH.Spot(spotter=spotter, freq_khz=freq, call=call, snr_db=snr, wpm=wpm, t=t)


def test_gate_fires_at_four_distinct_spotters():
    buf = RH.SpotBuffer(window_s=120, min_skimmers=4, freq_tol_khz=0.3)
    now = 1000.0
    for sp in ("AA1A", "BB2B", "CC3C"):
        buf.add(_spot(sp, "K1GHL", 14009.0, now))
    assert buf.ready_candidates(now) == []
    buf.add(_spot("AA1A", "K1GHL", 14009.05, now))     # duplicate spotter, not a 4th
    assert buf.ready_candidates(now) == []
    buf.add(_spot("DD4D", "K1GHL", 14009.1, now, snr=22))
    cands = buf.ready_candidates(now)
    assert len(cands) == 1
    assert cands[0].call == "K1GHL"
    assert cands[0].skimmers == 4
    assert cands[0].peak_snr == 22


def test_gate_respects_freq_cluster_time_window_and_cooldown():
    # window spans the cooldown so the re-chase path is exercised independently of eviction
    buf = RH.SpotBuffer(window_s=2000, min_skimmers=4, freq_tol_khz=0.3, cooldown_s=600)
    now = 1000.0
    buf.add(_spot("AA1A", "K1GHL", 14009.0, now))
    buf.add(_spot("BB2B", "K1GHL", 14009.1, now))
    buf.add(_spot("CC3C", "K1GHL", 14050.0, now))      # different signal, far away
    buf.add(_spot("DD4D", "K1GHL", 14050.1, now))
    assert buf.ready_candidates(now) == []             # only 2 clustered on-frequency
    buf.add(_spot("EE5E", "K1GHL", 14009.05, now))
    buf.add(_spot("FF6F", "K1GHL", 14008.95, now))
    assert len(buf.ready_candidates(now)) == 1         # 4 distinct on-frequency -> ready
    buf.mark_chased("K1GHL", now)
    assert buf.ready_candidates(now + 10) == []         # cooldown active
    assert len(buf.ready_candidates(now + 601)) == 1    # cooldown expired, spots still in window
    assert buf.ready_candidates(now + 2001) == []       # spots older than window -> evicted


# --- Task 4: corpus sink ---------------------------------------------------------
def test_write_sample_is_rbn_confirm_readable(tmp_path):
    cand = RH.Candidate(call="K1GHL", freq_khz=14009.0, skimmers=5,
                        peak_snr=22, wpm=28, spotters=["AA1A", "BB2B", "CC3C", "DD4D", "EE5E"])
    audio = (0.1 * np.sin(np.linspace(0, 500, 48000))).astype(np.float32)
    created = datetime(2026, 7, 12, 3, 15, 0, tzinfo=timezone.utc)
    d = RH.write_sample(tmp_path, cand, "CWU", 14008350, 48000, audio, "K1GHL", 7.4, created=created)
    sj = json.loads((d / "session.json").read_text())
    assert sj["freqHz"] == 14008350
    assert sj["mode"] == "CWU"
    assert datetime.strptime(sj["created"], "%Y-%m-%dT%H:%M:%S")
    assert (d / sj["audio"][0]["file"]).exists()
    assert abs(sj["durationSec"] - 1.0) < 0.05
    assert sj["rbn"]["callsign"] == "K1GHL" and sj["rbn"]["skimmers"] == 5
    assert sj["decode"]["agree"] is True and sj["decode"]["our_pick"] == "K1GHL"


def test_append_manifest_jsonl(tmp_path):
    RH.append_manifest(tmp_path, {"call": "K1GHL", "agree": True})
    RH.append_manifest(tmp_path, {"call": "W1AW", "agree": False})
    lines = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["call"] == "W1AW"


# --- Task 5: orchestrator --------------------------------------------------------
def _args(tmp_path, **over):
    a = types.SimpleNamespace(
        uri="ws://x", rx=0, sideband="CWU", pitch=650, dwell=18,
        ckpt="runs/exp15/model.pt", out=str(tmp_path),
    )
    a.__dict__.update(over)
    return a


def test_chase_once_writes_sample_on_agreement(tmp_path, monkeypatch):
    buf = RH.SpotBuffer(min_skimmers=4)
    now = 1000.0
    for sp in ("AA1A", "BB2B", "CC3C", "DD4D"):
        buf.add(RH.Spot(sp, 14009.0, "K1GHL", 20, 28, now))

    tuned = {}

    async def fake_tune(uri, rx, dial, sb):
        tuned["dial"] = dial

    async def fake_capture(uri, dwell, rx):
        return 48000, (0.1 * np.ones(48000, dtype="float32"))

    monkeypatch.setattr(RH, "tune_once", fake_tune)
    monkeypatch.setattr(RH, "_capture", fake_capture)
    monkeypatch.setattr(RH, "_is_active", lambda audio, sr: True)
    monkeypatch.setattr(RH, "_decode_picks", lambda sr, audio, ckpt: {"K1GHL": 7.4, "W1AW": 1.1})

    rec = asyncio.run(RH._chase_once(buf, _args(tmp_path), now, ranges=[]))
    assert rec["agree"] is True and rec["our_pick"] == "K1GHL"
    assert tuned["dial"] == 14009000 - 650
    dirs = [p for p in Path(tmp_path).iterdir() if p.is_dir()]
    assert len(dirs) == 1 and "K1GHL" in dirs[0].name
    assert (Path(tmp_path) / "manifest.jsonl").exists()
    assert buf.ready_candidates(now) == []   # on cooldown after chase


def test_chase_once_disagreement_writes_no_wav_dir(tmp_path, monkeypatch):
    buf = RH.SpotBuffer(min_skimmers=4)
    now = 1000.0
    for sp in ("AA1A", "BB2B", "CC3C", "DD4D"):
        buf.add(RH.Spot(sp, 14009.0, "K1GHL", 20, 28, now))

    async def fake_tune(uri, rx, dial, sb):
        pass

    async def fake_capture(uri, dwell, rx):
        return 48000, (0.1 * np.ones(48000, dtype="float32"))

    monkeypatch.setattr(RH, "tune_once", fake_tune)
    monkeypatch.setattr(RH, "_capture", fake_capture)
    monkeypatch.setattr(RH, "_is_active", lambda audio, sr: True)
    monkeypatch.setattr(RH, "_decode_picks", lambda sr, audio, ckpt: {"W1AW": 3.0})

    rec = asyncio.run(RH._chase_once(buf, _args(tmp_path), now, ranges=[]))
    assert rec["agree"] is False and rec["our_pick"] == "W1AW"
    assert [p for p in Path(tmp_path).iterdir() if p.is_dir()] == []
    assert (Path(tmp_path) / "manifest.jsonl").exists()


def test_chase_once_no_candidate_returns_none(tmp_path):
    buf = RH.SpotBuffer(min_skimmers=4)
    assert asyncio.run(RH._chase_once(buf, _args(tmp_path), 1000.0, ranges=[])) is None
