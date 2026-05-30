"""Voicebox MCP tool implementations.

Thin wrappers over existing services/routes. Tools are registered with dotted
names (``voicebox.speak`` etc.) so they look natural in agent logs —
the Python function name stays snake_case.
"""

from __future__ import annotations

import asyncio
import base64 as b64
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .. import models
from ..database import get_db
from ..services import captures as captures_service, profiles as profiles_service
from . import events as mcp_events
from .context import current_client_id, request_is_loopback
from .resolve import resolve_profile

logger = logging.getLogger(__name__)

# Absolute-path transcribes are bounded to keep a bad client from
# asking us to ingest a 20 GB file.
MAX_TRANSCRIBE_BYTES = 200 * 1024 * 1024  # 200 MB


def register_tools(mcp: FastMCP) -> None:
    """Attach all Voicebox tools to the given FastMCP instance."""

    @mcp.tool(
        name="voicebox.speak",
        description=(
            "Speak text in a Voicebox voice profile. Returns a generation id "
            "the caller can poll at /generate/{id}/status. Audio plays on the "
            "user's speakers and is saved to the Captures / History tab."
        ),
    )
    async def voicebox_speak(
        text: str,
        profile: str | None = None,
        engine: str | None = None,
        personality: bool | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Speak ``text`` in a voice profile.

        ``profile`` accepts a voice profile name (e.g. "Morgan") or id. If
        omitted, the server looks up the per-client binding for the calling
        MCP client, then falls back to the global default voice.

        ``personality`` only matters for profiles that have a personality
        prompt — when true, the text is first rewritten in character by the
        LLM before TTS. When omitted, the per-client binding's
        ``default_personality`` flag decides; when that is unset, the
        default is plain TTS.
        """
        from ..database.models import MCPClientBinding

        db = next(get_db())
        try:
            client_id = current_client_id.get()
            vp = resolve_profile(profile, client_id, db)
            if vp is None:
                raise ValueError(
                    "No voice profile resolved. Pass `profile=` with a "
                    "voice profile name or id, or set a default voice in "
                    "Voicebox → Settings → MCP."
                )

            binding = None
            if client_id:
                binding = (
                    db.query(MCPClientBinding)
                    .filter(MCPClientBinding.client_id == client_id)
                    .first()
                )

            resolved_personality = personality
            if resolved_personality is None and binding is not None:
                resolved_personality = bool(binding.default_personality)

            resolved_engine = engine
            if resolved_engine is None and binding is not None:
                resolved_engine = binding.default_engine

            use_persona = bool(resolved_personality) and bool(vp.personality)
            return await _speak(
                profile_id=vp.id,
                profile_name=vp.name,
                text=text,
                engine=resolved_engine,
                language=language,
                personality=use_persona,
                db=db,
            )
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.transcribe",
        description=(
            "Transcribe an audio clip to text using Voicebox's local Whisper. "
            "Pass exactly one of `audio_base64` (bytes as base64) or "
            "`audio_path` (absolute local file path — loopback callers only)."
        ),
    )
    async def voicebox_transcribe(
        audio_base64: str | None = None,
        audio_path: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if bool(audio_base64) == bool(audio_path):
            raise ValueError(
                "Pass exactly one of `audio_base64` or `audio_path`."
            )

        # Absolute-path mode: validate and transcribe in place. Restricted
        # to loopback callers so a Voicebox bound on 0.0.0.0 doesn't double
        # as an unauthenticated arbitrary-local-file read primitive.
        if audio_path is not None:
            if not request_is_loopback():
                raise ValueError(
                    "`audio_path` is only available to loopback callers — "
                    "remote callers must use `audio_base64`."
                )
            path = Path(audio_path)
            if not path.is_absolute():
                raise ValueError("`audio_path` must be absolute.")
            if not path.is_file():
                raise ValueError(f"File not found: {audio_path}")
            if path.stat().st_size > MAX_TRANSCRIBE_BYTES:
                raise ValueError(
                    f"File exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit."
                )
            return await _transcribe_file(path, language, model)

        # Base64 mode: decode into a temp file, transcribe, clean up.
        try:
            raw = b64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid audio_base64: {exc}") from exc
        if len(raw) > MAX_TRANSCRIBE_BYTES:
            raise ValueError(
                f"Audio exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit."
            )
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            return await _transcribe_file(tmp_path, language, model)
        finally:
            tmp_path.unlink(missing_ok=True)

    @mcp.tool(
        name="voicebox.list_captures",
        description=(
            "List recent voice captures (dictations, recordings, uploads) "
            "with their transcripts. Most-recent first."
        ),
    )
    async def voicebox_list_captures(
        limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        if not (1 <= limit <= 200):
            raise ValueError("`limit` must be between 1 and 200.")
        if offset < 0:
            raise ValueError("`offset` must be >= 0.")
        db = next(get_db())
        try:
            items, total = captures_service.list_captures(
                db, limit=limit, offset=offset
            )
            return {
                "captures": [
                    item.model_dump(mode="json") for item in items
                ],
                "total": total,
            }
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.list_profiles",
        description=(
            "List available voice profiles (both cloned voices and presets). "
            "Use the returned `name` with voicebox.speak(profile=...)."
        ),
    )
    async def voicebox_list_profiles() -> dict[str, Any]:
        db = next(get_db())
        try:
            profiles = await profiles_service.list_profiles(db)
            return {
                "profiles": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "voice_type": p.voice_type,
                        "language": p.language,
                        "has_personality": bool(getattr(p, "personality", None)),
                    }
                    for p in profiles
                ]
            }
        finally:
            db.close()

    @mcp.tool(
        name="voicebox.add_voice_samples",
        description=(
            "Attach one or more audio clips to an existing CLONED voice "
            "profile as training samples. Pass exactly one of `audio_paths` "
            "(absolute local paths — loopback callers only) or `audio_base64` "
            "(base64-encoded clips). Accepts opus/wav/mp3/m4a — clips are "
            "transcoded to 24 kHz mono internally. Clips that fail the quality "
            "gate (too short/long, silent) or duplicate an existing sample are "
            "skipped with a per-clip reason rather than failing the batch. "
            "Each sample needs a transcript: pass `labels` (index-aligned) or "
            "leave `auto_transcribe` on to label via local Whisper. Set "
            "`retrain=true` to re-encode the voice immediately (otherwise the "
            "new samples apply on the next voicebox.speak)."
        ),
    )
    async def voicebox_add_voice_samples(
        profile: str,
        audio_paths: list[str] | None = None,
        audio_base64: list[str] | None = None,
        labels: list[str] | None = None,
        auto_transcribe: bool = True,
        retrain: bool = False,
        min_duration: float = 5.0,
    ) -> dict[str, Any]:
        """Add samples to a cloned profile. See the tool description."""
        return await _add_voice_samples(
            profile=profile,
            audio_paths=audio_paths,
            audio_base64=audio_base64,
            labels=labels,
            auto_transcribe=auto_transcribe,
            retrain=retrain,
            min_duration=min_duration,
        )

    @mcp.tool(
        name="voicebox.retrain_voice",
        description=(
            "Re-encode a cloned voice profile from its current sample set and "
            "return when ready. Voicebox uses zero-shot cloning, so this is a "
            "fast synchronous re-encode (no long-running training job). Use "
            "after adding samples across several calls to warm the voice once."
        ),
    )
    async def voicebox_retrain_voice(profile: str) -> dict[str, Any]:
        """Synchronously re-encode a cloned profile's voice prompt."""
        db = next(get_db())
        try:
            vp = _resolve_cloned_profile(profile, db)
            result = await profiles_service.warm_voice_prompt(vp.id, db)
            return {"profile_id": vp.id, **result}
        finally:
            db.close()


# ─── Speak helper ──────────────────────────────────────────────────────────


async def _speak(
    *,
    profile_id: str,
    profile_name: str,
    text: str,
    engine: str | None,
    language: str | None,
    personality: bool,
    db,
) -> dict[str, Any]:
    """Delegate to POST /generate — the route handles personality-rewrite
    internally when ``personality=true`` and the profile has a prompt."""
    from ..routes.generations import generate_speech

    req = models.GenerationRequest(
        profile_id=profile_id,
        text=text,
        language=language or "en",
        engine=engine,
        personality=personality,
    )
    generation = await generate_speech(req, db)
    return _speak_response(generation, profile_name, source="mcp")


def _speak_response(
    generation, profile_name: str, *, source: str
) -> dict[str, Any]:
    """Normalize a GenerationResponse into the MCP tool's return shape.

    Also fires a speak-start event so the DictateWindow pill surfaces
    the agent's speech. Speak-end is fired from run_generation's
    completion hook.
    """
    payload = generation.model_dump(mode="json") if hasattr(
        generation, "model_dump"
    ) else dict(generation)
    generation_id = payload.get("id")
    mcp_events.publish(
        "speak-start",
        {
            "generation_id": generation_id,
            "profile_name": profile_name,
            "source": source,
            "client_id": current_client_id.get(),
        },
    )
    return {
        "generation_id": generation_id,
        "status": payload.get("status"),
        "profile": profile_name,
        "source": source,
        "poll_url": f"/generate/{generation_id}/status"
        if generation_id
        else None,
    }


# ─── Transcribe helper ─────────────────────────────────────────────────────


async def _transcribe_file(
    path: Path, language: str | None, model: str | None
) -> dict[str, Any]:
    from ..backends import WHISPER_HF_REPOS
    from ..services import transcribe as transcribe_service
    from ..utils.audio import load_audio

    whisper = transcribe_service.get_whisper_model()
    model_size = model or whisper.model_size
    valid = list(WHISPER_HF_REPOS.keys())
    if model_size not in valid:
        raise ValueError(
            f"Invalid STT model '{model_size}'. Must be one of: {', '.join(valid)}"
        )

    # load_audio is sync; keep the event loop responsive.
    audio, sr = await asyncio.to_thread(load_audio, str(path))
    duration = len(audio) / sr

    if (
        not whisper.is_loaded() or whisper.model_size != model_size
    ) and not whisper._is_model_cached(model_size):
        raise ValueError(
            f"Whisper model '{model_size}' is not yet downloaded. Open "
            "Voicebox → Settings → Models to download it first."
        )

    text = await whisper.transcribe(str(path), language, model_size)
    return {
        "text": text,
        "duration": duration,
        "language": language,
        "model": model_size,
    }


# ─── Add-voice-samples helpers ───────────────────────────────────────────────


def _resolve_cloned_profile(profile: str, db) -> Any:
    """Resolve a profile arg to a cloned-voice ORM row, or raise ValueError.

    Presets and designed voices can't take samples, so reject them with an
    actionable message rather than silently no-op'ing.
    """
    vp = resolve_profile(profile, current_client_id.get(), db)
    if vp is None:
        raise ValueError(
            f"No voice profile matched '{profile}'. Pass a profile name or id "
            "from voicebox.list_profiles."
        )
    voice_type = getattr(vp, "voice_type", None) or "cloned"
    if voice_type != "cloned":
        raise ValueError(
            f"Profile '{vp.name}' is a {voice_type} voice; only cloned voices "
            "accept samples. Presets and designed voices can't be augmented."
        )
    return vp


def _audio_file_hash(path: Path | str) -> str:
    """SHA-256 of a file's raw bytes (streamed, for arbitrary clip sizes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


async def _add_voice_samples(
    *,
    profile: str,
    audio_paths: list[str] | None,
    audio_base64: list[str] | None,
    labels: list[str] | None,
    auto_transcribe: bool,
    retrain: bool,
    min_duration: float,
) -> dict[str, Any]:
    """Attach clips to a cloned profile with per-clip partial success.

    Reuses the same ``add_profile_sample`` path the desktop UI calls, so
    MCP-added samples land in exactly the same place. Dedupe hashes the
    stored WAV representation (encode each candidate through the identical
    ``save_audio`` used on insert, then hash the bytes) so re-running an
    import is idempotent regardless of source container/codec.
    """
    from .. import config
    from ..database import ProfileSample as DBProfileSample
    from ..utils.audio import save_audio, validate_and_load_reference_audio

    if bool(audio_paths) == bool(audio_base64):
        raise ValueError("Pass exactly one of `audio_paths` or `audio_base64`.")

    db = next(get_db())
    tmp_paths: list[Path] = []
    skipped: list[dict[str, Any]] = []
    try:
        vp = _resolve_cloned_profile(profile, db)
        require_absolute = audio_paths is not None

        # Materialize inputs into a (path, display-label) work list.
        work: list[tuple[Path, str]] = []
        if require_absolute:
            if not request_is_loopback():
                raise ValueError(
                    "`audio_paths` is only available to loopback callers — "
                    "remote callers must use `audio_base64`."
                )
            for raw in audio_paths:
                work.append((Path(raw), str(raw)))
        else:
            for i, b64str in enumerate(audio_base64):
                display = f"<base64[{i}]>"
                try:
                    data = b64.b64decode(b64str, validate=True)
                except Exception as exc:
                    skipped.append(
                        {"path": display, "reason": f"invalid base64: {exc}"}
                    )
                    continue
                if len(data) > MAX_TRANSCRIBE_BYTES:
                    skipped.append(
                        {
                            "path": display,
                            "reason": f"exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit",
                        }
                    )
                    continue
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(data)
                    tp = Path(tmp.name)
                tmp_paths.append(tp)
                work.append((tp, display))

        # Hashes of the profile's existing stored samples — for idempotent re-imports.
        existing_hashes: set[str] = set()
        for s in db.query(DBProfileSample).filter_by(profile_id=vp.id).all():
            stored = config.resolve_storage_path(s.audio_path)
            if stored is not None and Path(stored).is_file():
                try:
                    existing_hashes.add(_audio_file_hash(stored))
                except OSError:
                    continue

        added: list[dict[str, Any]] = []
        for idx, (clip_path, display) in enumerate(work):
            try:
                if require_absolute and not clip_path.is_absolute():
                    skipped.append(
                        {"path": display, "reason": "path must be absolute"}
                    )
                    continue
                if not clip_path.is_file():
                    skipped.append({"path": display, "reason": "file not found"})
                    continue
                if clip_path.stat().st_size > MAX_TRANSCRIBE_BYTES:
                    skipped.append(
                        {
                            "path": display,
                            "reason": f"exceeds {MAX_TRANSCRIBE_BYTES // (1024 * 1024)} MB limit",
                        }
                    )
                    continue

                # Quality gate + preprocess, off the event loop.
                ok, err, audio, sr = await asyncio.to_thread(
                    validate_and_load_reference_audio, str(clip_path), min_duration
                )
                if not ok:
                    skipped.append({"path": display, "reason": err})
                    continue

                # Content hash on the stored representation.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as htmp:
                    hpath = Path(htmp.name)
                tmp_paths.append(hpath)
                await asyncio.to_thread(save_audio, audio, str(hpath), sr)
                clip_hash = _audio_file_hash(hpath)
                if clip_hash in existing_hashes:
                    skipped.append(
                        {"path": display, "reason": "duplicate of an existing sample"}
                    )
                    continue

                # Resolve the transcript/label.
                label = None
                if labels is not None and idx < len(labels) and labels[idx]:
                    label = labels[idx]
                elif auto_transcribe:
                    transcription = await _transcribe_file(clip_path, None, None)
                    label = (transcription.get("text") or "").strip()
                if not label:
                    skipped.append(
                        {
                            "path": display,
                            "reason": "no label and auto_transcribe off (or empty transcript)",
                        }
                    )
                    continue

                sample = await profiles_service.add_profile_sample(
                    vp.id, str(clip_path), label, db, min_duration=min_duration
                )
                existing_hashes.add(clip_hash)
                added.append(
                    {
                        "path": display,
                        "sample_id": sample.id,
                        "duration_s": round(len(audio) / sr, 2),
                        "label": label,
                    }
                )
            except Exception as exc:
                logger.exception("add_voice_samples: clip %s failed", display)
                skipped.append({"path": display, "reason": str(exc)})

        out: dict[str, Any] = {
            "profile_id": vp.id,
            "added": added,
            "skipped": skipped,
        }
        if retrain and added:
            out["retrain"] = await profiles_service.warm_voice_prompt(vp.id, db)
        elif retrain:
            out["retrain"] = {
                "status": "skipped",
                "reason": "no new samples were added",
            }
        return out
    finally:
        for tp in tmp_paths:
            tp.unlink(missing_ok=True)
        db.close()
