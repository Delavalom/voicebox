"""Tests for the voicebox.add_voice_samples / retrain_voice MCP tools.

Exercises the real ``_add_voice_samples`` helper (the body behind the
``@mcp.tool`` wrapper) against a temporary SQLite DB and on-disk storage.
Whisper and the TTS encoder are stubbed so the suite stays CPU-only and fast:

  - ``auto_transcribe`` is verified by stubbing ``_transcribe_file``.
  - ``retrain`` is verified by stubbing ``create_voice_prompt_for_profile``
    so ``warm_voice_prompt``'s own logic (cloned-check, sample count, cache
    clear) still runs, minus the heavy model load.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import (
    Base,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from backend.mcp_server import context, tools
from backend.services import profiles as profiles_service


def _write_tone(path, seconds: float, sr: int = 24000, freq: float = 220.0, amp: float = 0.3):
    """Write a clean sine tone WAV — passes the silence/RMS gate."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), audio, sr)
    return str(path)


def _make_profile(db, *, name="NBA Master", voice_type="cloned"):
    profile = DBVoiceProfile(
        name=name,
        voice_type=voice_type,
        language="en",
        preset_engine="kokoro" if voice_type == "preset" else None,
        preset_voice_id="am_adam" if voice_type == "preset" else None,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated data dir + test DB, with the tool's get_db()/loopback wired up."""
    original_data_dir = config.get_data_dir()
    config.set_data_dir(tmp_path / "data")

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # The tool opens its own session via get_db(); hand it a fresh session
    # from the test engine each call (it closes the session in its finally).
    monkeypatch.setattr(tools, "get_db", lambda: iter([session_local()]))

    # audio_paths is gated to loopback callers.
    addr_token = context.current_remote_addr.set("127.0.0.1")

    db = session_local()
    try:
        yield SimpleNamespace(db=db, tmp=tmp_path)
    finally:
        db.close()
        context.current_remote_addr.reset(addr_token)
        config.set_data_dir(original_data_dir)


async def test_add_single_clip_appears_as_sample(env):
    profile = _make_profile(env.db)
    clip = _write_tone(env.tmp / "good.wav", 6.0)

    out = await tools._add_voice_samples(
        profile="NBA Master",
        audio_paths=[clip],
        audio_base64=None,
        labels=["the shot goes in"],
        auto_transcribe=False,
        retrain=False,
        min_duration=5.0,
    )

    assert out["profile_id"] == profile.id
    assert out["skipped"] == []
    assert len(out["added"]) == 1
    assert out["added"][0]["label"] == "the shot goes in"
    assert out["added"][0]["duration_s"] == pytest.approx(6.0, abs=0.2)

    env.db.expire_all()
    rows = env.db.query(DBProfileSample).filter_by(profile_id=profile.id).all()
    assert len(rows) == 1
    assert rows[0].reference_text == "the shot goes in"


async def test_batch_partial_success_skips_short_clip(env):
    _make_profile(env.db)
    good = _write_tone(env.tmp / "good.wav", 6.0)
    short = _write_tone(env.tmp / "short.wav", 3.0)  # < 5s gate

    out = await tools._add_voice_samples(
        profile="NBA Master",
        audio_paths=[good, short],
        audio_base64=None,
        labels=["keeper", "tail"],
        auto_transcribe=False,
        retrain=False,
        min_duration=5.0,
    )

    assert len(out["added"]) == 1
    assert out["added"][0]["path"] == good
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["path"] == short
    assert "short" in out["skipped"][0]["reason"].lower()


async def test_preset_profile_is_rejected(env):
    _make_profile(env.db, name="Kokoro Adam", voice_type="preset")
    clip = _write_tone(env.tmp / "good.wav", 6.0)

    with pytest.raises(ValueError, match="cloned voices") as exc:
        await tools._add_voice_samples(
            profile="Kokoro Adam",
            audio_paths=[clip],
            audio_base64=None,
            labels=["x"],
            auto_transcribe=False,
            retrain=False,
            min_duration=5.0,
        )

    message = str(exc.value).lower()
    assert "preset" in message
    assert "cloned" in message


async def test_auto_transcribe_labels_clips(env, monkeypatch):
    profile = _make_profile(env.db)
    clip = _write_tone(env.tmp / "good.wav", 6.0)

    async def fake_transcribe(path, language, model):
        return {"text": "auto generated label", "duration": 6.0}

    monkeypatch.setattr(tools, "_transcribe_file", fake_transcribe)

    out = await tools._add_voice_samples(
        profile="NBA Master",
        audio_paths=[clip],
        audio_base64=None,
        labels=None,
        auto_transcribe=True,
        retrain=False,
        min_duration=5.0,
    )

    assert out["added"][0]["label"] == "auto generated label"
    env.db.expire_all()
    row = env.db.query(DBProfileSample).filter_by(profile_id=profile.id).one()
    assert row.reference_text == "auto generated label"


async def test_rerun_dedupes_identical_clip(env):
    profile = _make_profile(env.db)
    clip = _write_tone(env.tmp / "good.wav", 6.0)
    kwargs = dict(
        profile="NBA Master",
        audio_base64=None,
        labels=["x"],
        auto_transcribe=False,
        retrain=False,
        min_duration=5.0,
    )

    first = await tools._add_voice_samples(audio_paths=[clip], **kwargs)
    assert len(first["added"]) == 1

    second = await tools._add_voice_samples(audio_paths=[clip], **kwargs)
    assert second["added"] == []
    assert len(second["skipped"]) == 1
    assert "duplicate" in second["skipped"][0]["reason"].lower()

    env.db.expire_all()
    assert env.db.query(DBProfileSample).filter_by(profile_id=profile.id).count() == 1


async def test_retrain_warms_and_reports_ready(env, monkeypatch):
    _make_profile(env.db)
    clip = _write_tone(env.tmp / "good.wav", 6.0)

    async def fake_create_prompt(profile_id, db, use_cache=True, engine="qwen"):
        return {}

    monkeypatch.setattr(
        profiles_service, "create_voice_prompt_for_profile", fake_create_prompt
    )

    out = await tools._add_voice_samples(
        profile="NBA Master",
        audio_paths=[clip],
        audio_base64=None,
        labels=["x"],
        auto_transcribe=False,
        retrain=True,
        min_duration=5.0,
    )

    assert out["retrain"]["status"] == "ready"
    assert out["retrain"]["sample_count"] == 1


async def test_base64_input_path_does_not_require_loopback(env):
    """Base64 clips work for remote callers (no loopback gate)."""
    import base64 as b64

    profile = _make_profile(env.db)
    _write_tone(env.tmp / "good.wav", 6.0)
    encoded = b64.b64encode((env.tmp / "good.wav").read_bytes()).decode()

    # Drop loopback to prove base64 is allowed regardless.
    token = context.current_remote_addr.set("203.0.113.5")
    try:
        out = await tools._add_voice_samples(
            profile="NBA Master",
            audio_paths=None,
            audio_base64=[encoded],
            labels=["remote clip"],
            auto_transcribe=False,
            retrain=False,
            min_duration=5.0,
        )
    finally:
        context.current_remote_addr.reset(token)

    assert len(out["added"]) == 1
    assert out["added"][0]["path"] == "<base64[0]>"
    env.db.expire_all()
    assert env.db.query(DBProfileSample).filter_by(profile_id=profile.id).count() == 1
