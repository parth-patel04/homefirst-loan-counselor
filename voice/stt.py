"""
voice/stt.py — HomeFirst Vernacular Loan Counselor
Speech-to-Text using Sarvam AI.

Sarvam AI supports Indian languages natively:
  - Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam
  - Also handles English and Hinglish (code-switched speech)

API Docs : https://docs.sarvam.ai/api-reference-docs/endpoints/speech-to-text
Endpoint : POST https://api.sarvam.ai/speech-to-text
Model    : saarika:v2 (best multilingual Indian language model)
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import BinaryIO

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_STT_URL  = "https://api.sarvam.ai/speech-to-text"
SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
REQUEST_TIMEOUT = 30
MAX_AUDIO_SIZE  = 10 * 1024 * 1024   # 10 MB

# Sarvam language codes mapped from brain's locked language keys
SARVAM_LANG_CODES = {
    "hindi":   "hi-IN",
    "marathi": "mr-IN",
    "tamil":   "ta-IN",
    "english": "en-IN",
    "unknown": "en-IN",   # Use English for unknown — avoids mis-transcribing English as Hindi
}

SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

class STTResult:
    def __init__(
        self,
        transcript:    str,
        language_code: str,
        success:       bool,
        latency_ms:    float,
        error:         str | None = None,
        raw_response:  dict | None = None,
    ):
        self.transcript    = transcript
        self.language_code = language_code
        self.success       = success
        self.latency_ms    = latency_ms
        self.error         = error
        self.raw_response  = raw_response or {}

    def __repr__(self):
        return (
            f"STTResult(success={self.success}, "
            f"lang={self.language_code}, "
            f"transcript='{self.transcript[:60]}', "
            f"latency={self.latency_ms:.0f}ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core STT class
# ─────────────────────────────────────────────────────────────────────────────

class SarvamSTT:
    """
    Wrapper around Sarvam AI's speech-to-text API.

    Usage:
        stt = SarvamSTT()

        # From bytes (Streamlit audio recorder output)
        result = stt.transcribe_bytes(audio_bytes, locked_language="hindi")

        # From file path
        result = stt.transcribe_file("recording.wav", locked_language="english")
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SARVAM_API_KEY
        if not self.api_key:
            logger.warning(
                "SARVAM_API_KEY not set. STT calls will fail. "
                "Set it in your .env file."
            )

    # ── Public methods ────────────────────────────────────────────────────────

    def transcribe_bytes(
        self,
        audio_bytes:     bytes,
        locked_language: str = "unknown",
        filename:        str = "audio.wav",
    ) -> STTResult:
        """
        Transcribe raw audio bytes.

        Args:
            audio_bytes     : Raw audio data (wav/mp3/webm etc.)
            locked_language : Locked language from brain
            filename        : Hint for MIME type detection

        Returns:
            STTResult with transcript and metadata
        """
        if not audio_bytes:
            return STTResult(
                transcript="", language_code="",
                success=False, latency_ms=0,
                error="Empty audio bytes received."
            )

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            return STTResult(
                transcript="", language_code="",
                success=False, latency_ms=0,
                error=f"Audio too large ({len(audio_bytes)//1024}KB). Max is 10MB."
            )

        audio_file      = io.BytesIO(audio_bytes)
        audio_file.name = filename
        return self._call_sarvam(audio_file, locked_language, filename)

    def transcribe_file(
        self,
        file_path:       str | Path,
        locked_language: str = "unknown",
    ) -> STTResult:
        """
        Transcribe an audio file from disk.

        Args:
            file_path       : Path to audio file
            locked_language : Locked language from brain

        Returns:
            STTResult with transcript and metadata
        """
        path = Path(file_path)

        if not path.exists():
            return STTResult(
                transcript="", language_code="",
                success=False, latency_ms=0,
                error=f"File not found: {path}"
            )

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            return STTResult(
                transcript="", language_code="",
                success=False, latency_ms=0,
                error=f"Unsupported format '{suffix}'. Use: {SUPPORTED_FORMATS}"
            )

        with open(path, "rb") as f:
            return self._call_sarvam(f, locked_language, path.name)

    # ── Internal API call ─────────────────────────────────────────────────────

    def _call_sarvam(
        self,
        audio_file:      BinaryIO,
        locked_language: str,
        filename:        str,
    ) -> STTResult:
        """POST to Sarvam AI STT endpoint."""
        lang_code = SARVAM_LANG_CODES.get(locked_language, "hi-IN")
        mime_type = _get_mime_type(filename)

        headers = {"api-subscription-key": self.api_key}
        files   = {"file": (filename, audio_file, mime_type)}
        data    = {
            "language_code":   lang_code,
            "model":           "saarika:v2.5",
            "with_timestamps": "false",
        }

        logger.info(
            "Sarvam STT | lang=%s | file=%s | mime=%s",
            lang_code, filename, mime_type
        )

        t0 = time.time()
        try:
            response   = requests.post(
                SARVAM_STT_URL,
                headers = headers,
                files   = files,
                data    = data,
                timeout = REQUEST_TIMEOUT,
            )
            latency_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                error_msg = _parse_error(response)
                logger.error("Sarvam STT %d: %s", response.status_code, error_msg)
                return STTResult(
                    transcript="", language_code=lang_code,
                    success=False, latency_ms=latency_ms,
                    error=error_msg
                )

            result     = response.json()
            transcript = result.get("transcript", "").strip()

            if not transcript:
                return STTResult(
                    transcript="", language_code=lang_code,
                    success=False, latency_ms=latency_ms,
                    error="Empty transcript. Please speak clearly and try again.",
                    raw_response=result,
                )

            logger.info("STT OK | '%s...' | %.0fms", transcript[:50], latency_ms)
            return STTResult(
                transcript    = transcript,
                language_code = lang_code,
                success       = True,
                latency_ms    = latency_ms,
                raw_response  = result,
            )

        except requests.Timeout:
            latency_ms = (time.time() - t0) * 1000
            return STTResult(
                transcript="", language_code=lang_code,
                success=False, latency_ms=latency_ms,
                error="STT request timed out. Please try again."
            )
        except requests.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            return STTResult(
                transcript="", language_code=lang_code,
                success=False, latency_ms=latency_ms,
                error=f"Network error: {e}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_mime_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".wav":  "audio/wav",
        ".mp3":  "audio/mpeg",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".m4a":  "audio/mp4",
        ".webm": "audio/webm",
    }.get(ext, "audio/wav")


def _parse_error(response: requests.Response) -> str:
    try:
        body = response.json()
        return body.get("message") or body.get("error") or str(body)
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:200]}"


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + convenience function (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

_stt_instance: SarvamSTT | None = None

def get_stt() -> SarvamSTT:
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = SarvamSTT()
    return _stt_instance


def transcribe(
    audio_bytes:     bytes,
    locked_language: str = "unknown",
    filename:        str = "audio.wav",
) -> STTResult:
    """
    Top-level function called by app.py.
    Transcribes audio bytes using Sarvam AI.
    """
    return get_stt().transcribe_bytes(audio_bytes, locked_language, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run: python voice/stt.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Sarvam STT — Self Test")
    print("=" * 55)

    stt = SarvamSTT()

    r = stt.transcribe_bytes(b"", "hindi")
    assert not r.success and "Empty" in r.error
    print("✅ Empty audio guard")

    big = b"0" * (11 * 1024 * 1024)
    r = stt.transcribe_bytes(big, "english")
    assert not r.success and "large" in r.error.lower()
    print("✅ Oversized audio guard")

    r = stt.transcribe_file("test.xyz")
    assert not r.success
    print("✅ Unsupported format guard")

    r = stt.transcribe_file("nonexistent.wav")
    assert not r.success and "not found" in r.error.lower()
    print("✅ Missing file guard")

    for lang, expected in [("hindi","hi-IN"),("english","en-IN"),("marathi","mr-IN"),("tamil","ta-IN"),("unknown","hi-IN")]:
        assert SARVAM_LANG_CODES[lang] == expected
    print("✅ Language code mapping")

    assert _get_mime_type("audio.wav")  == "audio/wav"
    assert _get_mime_type("audio.webm") == "audio/webm"
    print("✅ MIME type mapping")

    if not stt.api_key:
        print("\n⚠️  SARVAM_API_KEY not set — skipping live API test.")
    else:
        test_wav = Path("test_audio.wav")
        if test_wav.exists():
            print(f"\n🎤 Testing with {test_wav}...")
            r = stt.transcribe_file(str(test_wav), "hindi")
            print(f"   {r}")
        else:
            print("\n💡 Place a test_audio.wav in project root to test live STT.")

    print("\nAll STT tests passed ✅")
