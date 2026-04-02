"""
verify_keys.py — HomeFirst Vernacular Loan Counselor
Run this script to verify all your API keys are working correctly.

Usage:
    python verify_keys.py

Tests:
    1. .env file loading
    2. OpenAI / GPT-4o connection
    3. Sarvam AI connection (STT)
    4. ElevenLabs connection (TTS)
    5. ElevenLabs voice IDs
    6. tools.py logic (no API needed)
    7. Full pipeline simulation
"""

import os
import sys
import json
import time

# ── Load .env first ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("\n✅ [1/7] .env file loaded successfully")
except ImportError:
    print("\n❌ python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def ok(msg: str):
    print(f"  ✅ {msg}")

def warn(msg: str):
    print(f"  ⚠️  {msg}")

def fail(msg: str):
    print(f"  ❌ {msg}")

def check_env_var(name: str) -> str | None:
    val = os.getenv(name)
    if not val or val.startswith("your_"):
        fail(f"{name} is missing or still a placeholder")
        return None
    ok(f"{name} loaded → {val[:12]}...")
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Check all .env variables are present
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 1 — Environment Variables")

openai_key    = check_env_var("OPENAI_API_KEY")
sarvam_key    = check_env_var("SARVAM_API_KEY")
eleven_key    = check_env_var("ELEVENLABS_API_KEY")
voice_english = check_env_var("ELEVENLABS_VOICE_ID_ENGLISH")
voice_hindi   = check_env_var("ELEVENLABS_VOICE_ID_HINDI")
voice_marathi = check_env_var("ELEVENLABS_VOICE_ID_MARATHI")
voice_tamil   = check_env_var("ELEVENLABS_VOICE_ID_TAMIL")

missing = [
    k for k, v in {
        "OPENAI_API_KEY":            openai_key,
        "SARVAM_API_KEY":            sarvam_key,
        "ELEVENLABS_API_KEY":        eleven_key,
        "ELEVENLABS_VOICE_ID_ENGLISH": voice_english,
        "ELEVENLABS_VOICE_ID_HINDI":   voice_hindi,
    }.items() if not v
]

if missing:
    warn(f"{len(missing)} key(s) missing — some tests will be skipped.")
else:
    ok("All required environment variables are set!")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: OpenAI / GPT-4o
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 2 — OpenAI / GPT-4o Connection")

if not openai_key:
    warn("Skipping — OPENAI_API_KEY not set.")
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        t0 = time.time()
        response = client.chat.completions.create(
            model      = "gpt-4o",
            max_tokens = 15,
            messages   = [{"role": "user", "content": "Reply with exactly: GPT-4o connected!"}],
        )
        latency = (time.time() - t0) * 1000

        reply = response.choices[0].message.content.strip()
        model = response.model
        ok(f"GPT-4o connected!")
        ok(f"Model    : {model}")
        ok(f"Response : {reply}")
        ok(f"Latency  : {latency:.0f}ms")

        # Check token usage
        usage = response.usage
        ok(f"Tokens   : {usage.prompt_tokens} in / {usage.completion_tokens} out")

    except ImportError:
        fail("openai package not installed. Run: pip install openai")
    except Exception as e:
        error_type = type(e).__name__
        if "AuthenticationError" in error_type:
            fail(f"Invalid API key. Re-copy it from platform.openai.com/api-keys")
        elif "RateLimitError" in error_type:
            fail(f"Rate limit or billing issue. Add credits at platform.openai.com/billing")
        elif "InsufficientQuota" in error_type:
            fail(f"Balance is $0. Top up at platform.openai.com/billing")
        else:
            fail(f"{error_type}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Sarvam AI (STT endpoint ping)
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 3 — Sarvam AI Connection")

if not sarvam_key:
    warn("Skipping — SARVAM_API_KEY not set.")
else:
    try:
        import requests

        # Ping the Sarvam STT endpoint with a tiny invalid payload
        # We expect a 400/422 (validation error) not a 401 (auth error)
        # This confirms the key is valid without uploading real audio
        t0 = time.time()
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers = {"api-subscription-key": sarvam_key},
            files   = {"file": ("test.wav", b"RIFF", "audio/wav")},
            data    = {"language_code": "hi-IN", "model": "saarika:v2"},
            timeout = 15,
        )
        latency = (time.time() - t0) * 1000

        if response.status_code == 401:
            fail(f"Invalid Sarvam API key (401 Unauthorized)")
        elif response.status_code in (400, 422):
            # Expected — bad audio but key was accepted
            ok(f"Sarvam AI key is valid! (got {response.status_code} — expected for test audio)")
            ok(f"Latency: {latency:.0f}ms")
        elif response.status_code == 200:
            ok(f"Sarvam AI connected and STT responded! Latency: {latency:.0f}ms")
        else:
            warn(f"Unexpected status {response.status_code}: {response.text[:100]}")

    except ImportError:
        fail("requests not installed. Run: pip install requests")
    except requests.Timeout:
        fail("Sarvam AI request timed out. Check your internet connection.")
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: ElevenLabs — Account & Quota
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 4 — ElevenLabs Account & Quota")

if not eleven_key:
    warn("Skipping — ELEVENLABS_API_KEY not set.")
else:
    try:
        import requests

        t0 = time.time()
        response = requests.get(
            "https://api.elevenlabs.io/v1/user",
            headers = {"xi-api-key": eleven_key},
            timeout = 15,
        )
        latency = (time.time() - t0) * 1000

        if response.status_code == 401:
            fail("Invalid ElevenLabs API key (401 Unauthorized)")
        elif response.status_code == 200:
            data         = response.json()
            subscription = data.get("subscription", {})
            tier         = subscription.get("tier", "unknown")
            char_limit   = subscription.get("character_limit", 0)
            char_used    = subscription.get("character_count", 0)
            chars_left   = char_limit - char_used

            ok(f"ElevenLabs connected!")
            ok(f"Plan       : {tier}")
            ok(f"Chars used : {char_used:,} / {char_limit:,}")
            ok(f"Chars left : {chars_left:,}")
            ok(f"Latency    : {latency:.0f}ms")

            if chars_left < 1000:
                warn("Less than 1,000 characters remaining. Consider upgrading.")
        else:
            warn(f"Unexpected status {response.status_code}: {response.text[:100]}")

    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: ElevenLabs — Voice IDs validation
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 5 — ElevenLabs Voice IDs")

if not eleven_key:
    warn("Skipping — ELEVENLABS_API_KEY not set.")
else:
    try:
        import requests

        # Fetch all available voices
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers = {"xi-api-key": eleven_key},
            timeout = 15,
        )

        if response.status_code == 200:
            voices      = response.json().get("voices", [])
            voice_ids   = {v["voice_id"] for v in voices}
            voice_names = {v["voice_id"]: v["name"] for v in voices}

            voice_vars = {
                "English": voice_english,
                "Hindi":   voice_hindi,
                "Marathi": voice_marathi,
                "Tamil":   voice_tamil,
            }

            for lang, vid in voice_vars.items():
                if not vid:
                    warn(f"{lang} voice ID not set")
                elif vid in voice_ids:
                    ok(f"{lang} voice ID valid → '{voice_names.get(vid, vid)}'")
                else:
                    # Voice might be a shared/library voice — do a TTS test instead
                    warn(
                        f"{lang} voice ID '{vid[:20]}...' not in your voice list "
                        f"(may still work if it's a library voice)"
                    )
        else:
            warn(f"Could not fetch voices: {response.status_code}")

    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: ElevenLabs — Live TTS test (synthesize 1 sentence)
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 6 — ElevenLabs Live TTS Synthesis")

if not eleven_key or not voice_english:
    warn("Skipping — ElevenLabs key or English voice ID not set.")
else:
    try:
        import requests

        test_text = "Hello, I am Ghar Mitra, your home loan counselor from HomeFirst."
        voice_id  = voice_english

        t0 = time.time()
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers = {
                "xi-api-key":   eleven_key,
                "Content-Type": "application/json",
                "Accept":       "audio/mpeg",
            },
            json = {
                "text":     test_text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability":        0.55,
                    "similarity_boost": 0.75,
                },
            },
            timeout = 30,
        )
        latency = (time.time() - t0) * 1000

        if response.status_code == 200:
            audio_size = len(response.content)
            ok(f"TTS synthesis successful!")
            ok(f"Audio size : {audio_size:,} bytes ({audio_size//1024}KB)")
            ok(f"Latency    : {latency:.0f}ms")

            # Save test audio file
            with open("test_tts_output.mp3", "wb") as f:
                f.write(response.content)
            ok(f"Saved to   : test_tts_output.mp3 — play it to verify voice quality!")
        else:
            fail(f"TTS failed ({response.status_code}): {response.text[:200]}")

    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: tools.py — Deterministic logic (no API needed)
# ─────────────────────────────────────────────────────────────────────────────

print_section("TEST 7 — tools.py Loan Logic")

try:
    from tools import calculate_emi, check_eligibility, dispatch_tool, TOOL_SCHEMAS

    # EMI test
    emi = calculate_emi(1_500_000)
    assert "monthly_emi" in emi
    assert abs(emi["monthly_emi"] - 13981.97) < 1
    ok(f"EMI for Rs15L = Rs{emi['monthly_emi']:,.2f}/month")

    # Eligibility — approved
    result = check_eligibility(60_000, 2_500_000, 1_800_000, "salaried")
    assert result["eligible"] == True
    ok(f"Eligible salaried → approved Rs{result['approved_loan_amount']:,.0f}")

    # Eligibility — rejected
    result = check_eligibility(10_000, 1_000_000, 800_000, "salaried")
    assert result["eligible"] == False
    ok(f"Rejected (low income) → {len(result['rejection_reasons'])} reason(s)")

    # Tool schema format — GPT-4o
    assert TOOL_SCHEMAS[0]["type"] == "function"
    assert "parameters" in TOOL_SCHEMAS[0]["function"]
    ok(f"Tool schemas are in correct GPT-4o format")

    # Dispatcher
    r = dispatch_tool("calculate_emi", {"loan_amount": 2_000_000})
    assert "monthly_emi" in r
    ok(f"dispatch_tool works correctly")

except ImportError:
    fail("tools.py not found. Make sure you're running from the project root.")
except AssertionError as e:
    fail(f"Logic assertion failed: {e}")
except Exception as e:
    fail(f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*55}")
print("  VERIFICATION COMPLETE")
print(f"{'═'*55}")
print("""
Next steps:
  ✅ All green  → Run: python agent/brain.py  (full GPT-4o test)
  ⚠️  Some warn  → Fix the warned keys, then re-run this script
  ❌ Any red    → Fix the error shown, then re-run this script

Once all green:
  → Move to Phase 3: voice layer (stt.py + tts.py)
  → Then Phase 4: Streamlit UI (app.py)
""")
