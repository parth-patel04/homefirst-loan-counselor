"""
voice/tts.py — HomeFirst Vernacular Loan Counselor
Text-to-Speech with automatic fallback:
  Primary  → ElevenLabs  (high quality, multilingual v2)
  Fallback → Sarvam AI   (Indian languages, lower latency)

app.py only ever calls TTSOrchestrator.speak() — never the
individual provider classes directly.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — ElevenLabs
# ─────────────────────────────────────────────────────────────────────────────

ELEVENLABS_BASE_URL      = "https://api.elevenlabs.io/v1"
ELEVENLABS_API_KEY       = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL         = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
MAX_TEXT_LENGTH          = 2500   # chars per request

# Voice IDs per language — configurable via .env
# Defaults are real public ElevenLabs voices that work on free tier
ELEVENLABS_VOICES = {
    "english": os.getenv("ELEVENLABS_VOICE_ID_ENGLISH", "21m00Tcm4TlvDq8ikWAM"),  # Rachel
    "hindi":   os.getenv("ELEVENLABS_VOICE_ID_HINDI",   "pFZP5JQG7iQjIQuC4Bku"),  # Lily (multilingual)
    "marathi": os.getenv("ELEVENLABS_VOICE_ID_MARATHI",  "pFZP5JQG7iQjIQuC4Bku"),
    "tamil":   os.getenv("ELEVENLABS_VOICE_ID_TAMIL",   "pFZP5JQG7iQjIQuC4Bku"),
    "unknown": os.getenv("ELEVENLABS_VOICE_ID_ENGLISH", "21m00Tcm4TlvDq8ikWAM"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Constants — Sarvam TTS
# ─────────────────────────────────────────────────────────────────────────────

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

SARVAM_TTS_LANG = {
    "hindi":   "hi-IN",
    "marathi": "mr-IN",
    "tamil":   "ta-IN",
    "english": "en-IN",
    "unknown": "hi-IN",
}

SARVAM_SPEAKERS = {
    "hindi":   "anushka",
    "marathi": "anushka",
    "tamil":   "anushka",
    "english": "anushka",
    "unknown": "anushka",
}

REQUEST_TIMEOUT = 30


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTSResult:
    audio_bytes: bytes          # raw MP3/WAV bytes — play directly in Streamlit
    success:     bool
    provider:    str            # "elevenlabs" | "sarvam" | "none"
    latency_ms:  float
    error:       str | None = None

    @property
    def has_audio(self) -> bool:
        return self.success and len(self.audio_bytes) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_error(response: requests.Response) -> str:
    """Extract a human-readable error message from an API error response."""
    try:
        body = response.json()
        # ElevenLabs wraps errors in {"detail": {"message": ...}}
        if isinstance(body.get("detail"), dict):
            return body["detail"].get("message") or str(body["detail"])
        return body.get("message") or body.get("error") or str(body)
    except Exception:
        return f"HTTP {response.status_code}: {response.text[:200]}"


# ─────────────────────────────────────────────────────────────────────────────
# ElevenLabs TTS
# ─────────────────────────────────────────────────────────────────────────────

class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech.
    Uses eleven_multilingual_v2 which handles Hindi/Marathi/Tamil natively.
    Returns MP3 bytes.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not set.")

    def synthesize(self, text: str, locked_language: str = "unknown") -> TTSResult:
        """
        Convert text to speech.

        Args:
            text            : Text to speak (in locked language)
            locked_language : "english" | "hindi" | "marathi" | "tamil" | "unknown"

        Returns:
            TTSResult with MP3 bytes
        """
        text = text.strip()
        if not text:
            return TTSResult(b"", False, "elevenlabs", 0, "Empty text.")

        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
            logger.warning("Text truncated to %d chars for ElevenLabs.", MAX_TEXT_LENGTH)

        voice_id = ELEVENLABS_VOICES.get(locked_language, ELEVENLABS_VOICES["unknown"])
        url      = f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key":   self.api_key,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        }

        payload = {
            "text":            text,
            "model_id":        ELEVENLABS_MODEL,
            "output_format":   ELEVENLABS_OUTPUT_FORMAT,
            "voice_settings":  {
                "stability":        0.55,
                "similarity_boost": 0.75,
                "style":            0.20,
                "use_speaker_boost": True,
            },
        }

        logger.info(
            "ElevenLabs TTS | lang=%s | voice=%s | chars=%d",
            locked_language, voice_id, len(text)
        )

        t0 = time.time()
        try:
            response   = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            latency_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                error = _parse_error(response)
                logger.error("ElevenLabs %d: %s", response.status_code, error)
                return TTSResult(b"", False, "elevenlabs", latency_ms, error)

            logger.info("ElevenLabs OK | %d bytes | %.0fms", len(response.content), latency_ms)
            return TTSResult(response.content, True, "elevenlabs", latency_ms)

        except requests.Timeout:
            latency_ms = (time.time() - t0) * 1000
            return TTSResult(b"", False, "elevenlabs", latency_ms, "TTS timed out.")
        except requests.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            return TTSResult(b"", False, "elevenlabs", latency_ms, f"Network error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sarvam TTS (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class SarvamTTS:
    """
    Sarvam AI Text-to-Speech.
    Used as fallback when ElevenLabs fails or has no key.
    Lower latency for Indian languages. Returns WAV bytes (base64 decoded).
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or SARVAM_API_KEY
        if not self.api_key:
            logger.warning("SARVAM_API_KEY not set (needed for TTS fallback).")

    def synthesize(self, text: str, locked_language: str = "unknown") -> TTSResult:
        """
        Convert text to speech using Sarvam AI.

        Args:
            text            : Text to speak
            locked_language : Locked language key

        Returns:
            TTSResult with WAV bytes
        """
        text = text.strip()
        if not text:
            return TTSResult(b"", False, "sarvam", 0, "Empty text.")

        lang_code = SARVAM_TTS_LANG.get(locked_language, "hi-IN")
        speaker   = SARVAM_SPEAKERS.get(locked_language, "meera")

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type":         "application/json",
        }

        payload = {
            "inputs":               [text],
            "target_language_code": lang_code,
            "speaker":              speaker,
            "pitch":                0,
            "pace":                 1.0,
            "loudness":             1.5,
            "speech_sample_rate":   22050,
            "enable_preprocessing": True,
            "model":                "bulbul:v2",
        }

        logger.info(
            "Sarvam TTS fallback | lang=%s | speaker=%s | chars=%d",
            lang_code, speaker, len(text)
        )

        t0 = time.time()
        try:
            response   = requests.post(
                SARVAM_TTS_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            latency_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                error = _parse_error(response)
                logger.error("Sarvam TTS %d: %s", response.status_code, error)
                return TTSResult(b"", False, "sarvam", latency_ms, error)

            result     = response.json()
            audio_list = result.get("audios", [])

            if not audio_list:
                return TTSResult(b"", False, "sarvam", latency_ms, "No audio in response.")

            audio_bytes = base64.b64decode(audio_list[0])
            logger.info("Sarvam TTS OK | %d bytes | %.0fms", len(audio_bytes), latency_ms)
            return TTSResult(audio_bytes, True, "sarvam", latency_ms)

        except requests.Timeout:
            latency_ms = (time.time() - t0) * 1000
            return TTSResult(b"", False, "sarvam", latency_ms, "Sarvam TTS timed out.")
        except requests.RequestException as e:
            latency_ms = (time.time() - t0) * 1000
            return TTSResult(b"", False, "sarvam", latency_ms, f"Network error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TTSOrchestrator — ElevenLabs → Sarvam fallback
# ─────────────────────────────────────────────────────────────────────────────

class TTSOrchestrator:
    """
    Tries ElevenLabs first. If it fails, automatically falls back to Sarvam AI.

    app.py only ever uses this class:
        tts    = TTSOrchestrator()
        result = tts.speak("Aapka loan approved hai!", locked_language="hindi")
        if result.has_audio:
            st.audio(result.audio_bytes, format="audio/mp3")
    """

    def __init__(self):
        self.elevenlabs = ElevenLabsTTS()
        self.sarvam     = SarvamTTS()

    def speak(self, text: str, locked_language: str = "unknown") -> TTSResult:
        """
        Synthesize speech with automatic fallback.

        Args:
            text            : LLM reply text (will be cleaned before synthesis)
            locked_language : Brain's locked language key

        Returns:
            TTSResult — always returns something (even silent on total failure)
        """
        # Clean LLM output before sending to TTS
        text = clean_text_for_tts(text)

        if not text:
            return TTSResult(b"", False, "none", 0, "No speakable text after cleaning.")

        # ── Try ElevenLabs first ─────────────────────────────────────────────
        if self.elevenlabs.api_key:
            result = self.elevenlabs.synthesize(text, locked_language)
            if result.success:
                return result
            logger.warning("ElevenLabs failed (%s) — falling back to Sarvam.", result.error)
        else:
            logger.info("No ElevenLabs key — going straight to Sarvam fallback.")

        # ── Fallback: Sarvam TTS ─────────────────────────────────────────────
        if self.sarvam.api_key:
            result = self.sarvam.synthesize(text, locked_language)
            if result.success:
                return result
            logger.error("Sarvam TTS also failed: %s", result.error)
            return result

        # ── Both failed ──────────────────────────────────────────────────────
        logger.error("Both TTS providers unavailable.")
        return TTSResult(
            b"", False, "none", 0,
            "TTS unavailable: no API keys configured for ElevenLabs or Sarvam."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaner
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_for_tts(text: str) -> str:
    """
    Clean LLM output before sending to TTS.
    Removes markdown, emoji, debug blocks, and handoff lines.
    """
    # Remove INTERNAL_STATE block (safety net — brain.py should already strip it)
    text = re.sub(r"<INTERNAL_STATE>.*?</INTERNAL_STATE>", "", text, flags=re.DOTALL)

    # Remove markdown bold / italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)

    # Remove markdown headers
    text = re.sub(r"#{1,6}\s*", "", text)

    # Remove inline and block code
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)

    # Remove horizontal rules
    text = re.sub(r"---+", "", text)

    # Remove emoji (broad Unicode ranges)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # Remove [HANDOFF TRIGGERED...] line
    text = re.sub(r"\[HANDOFF TRIGGERED[^\]]*\]", "", text)

    # Collapse whitespace
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + convenience function (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

_tts_instance: TTSOrchestrator | None = None

def get_tts() -> TTSOrchestrator:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSOrchestrator()
    return _tts_instance


def speak(text: str, locked_language: str = "unknown") -> TTSResult:
    """
    Top-level convenience function called by app.py.

    Args:
        text            : LLM reply text
        locked_language : Brain's locked language

    Returns:
        TTSResult with audio bytes ready for st.audio()
    """
    return get_tts().speak(text, locked_language)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run: python voice/tts.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TTS Orchestrator — Self Test")
    print("=" * 55)

    # Test 1: text cleaner
    dirty = (
        "**Namaste!** Aapka loan _approved_ hai! \U0001F389\n\n"
        "# Details:\n```json\n{}\n```\n"
        "[HANDOFF TRIGGERED: Routing to Human RM]\n"
        "<INTERNAL_STATE>{}</INTERNAL_STATE>"
    )
    cleaned = clean_text_for_tts(dirty)
    assert "**"            not in cleaned, "Bold not removed"
    assert "```"           not in cleaned, "Code block not removed"
    assert "HANDOFF"       not in cleaned, "Handoff line not removed"
    assert "INTERNAL_STATE" not in cleaned, "INTERNAL_STATE not removed"
    assert "#"             not in cleaned, "Header not removed"
    print(f"✅ Text cleaner works")
    print(f"   Cleaned: '{cleaned}'")

    # Test 2: empty text guard
    tts = TTSOrchestrator()
    r = tts.speak("", "hindi")
    assert not r.success
    print("✅ Empty text guard works")

    # Test 3: text with only markdown
    r = tts.speak("**##**", "english")
    assert not r.success
    print("✅ Markdown-only text guard works")

    # Test 4: live synthesis (ElevenLabs)
    if not tts.elevenlabs.api_key:
        print("\n⚠️  ELEVENLABS_API_KEY not set — skipping live ElevenLabs test.")
    else:
        print("\n🔊 Testing ElevenLabs live synthesis...")
        r = tts.speak(
            "Namaste! Main Ghar Mitra hoon. Aapka home loan ke baare mein kya sawaal hai?",
            locked_language="hindi"
        )
        if r.success:
            with open("test_hindi_tts.mp3", "wb") as f:
                f.write(r.audio_bytes)
            print(f"✅ ElevenLabs Hindi TTS OK | {len(r.audio_bytes):,} bytes | {r.latency_ms:.0f}ms")
            print(f"   Saved to: test_hindi_tts.mp3")
        else:
            print(f"❌ ElevenLabs failed: {r.error}")
            print(f"   Provider: {r.provider}")

    # Test 5: live synthesis (Sarvam fallback)
    if not tts.sarvam.api_key:
        print("\n⚠️  SARVAM_API_KEY not set — skipping live Sarvam TTS test.")
    else:
        print("\n🔊 Testing Sarvam TTS directly...")
        r = tts.sarvam.synthesize(
            "Namaste! Aapka loan approved hai.",
            locked_language="hindi"
        )
        if r.success:
            with open("test_sarvam_tts.wav", "wb") as f:
                f.write(r.audio_bytes)
            print(f"✅ Sarvam TTS OK | {len(r.audio_bytes):,} bytes | {r.latency_ms:.0f}ms")
            print(f"   Saved to: test_sarvam_tts.wav")
        else:
            print(f"❌ Sarvam TTS failed: {r.error}")

    print("\nAll TTS unit tests passed ✅")
