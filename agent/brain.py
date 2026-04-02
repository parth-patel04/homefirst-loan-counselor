"""
brain.py — HomeFirst Vernacular Loan Counselor
The LLM orchestration layer. Handles:
  - Multi-turn conversation state
  - Language detection & locking (turns 1-2)
  - Structured entity extraction from INTERNAL_STATE JSON
  - Tool calling (calculate_emi, check_eligibility) via Claude API
  - Out-of-domain detection
  - Lead scoring & human handoff trigger
  - RAG context injection (optional, if rag.py is available)
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.prompts import build_system_prompt, build_language_detection_prompt, LANGUAGE_LABELS
from tools import TOOL_SCHEMAS, dispatch_tool
import db

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL               = "gpt-4o"
MAX_TOKENS          = 1500
HANDOFF_SCORE_GATE  = 7          # intent score threshold for handoff
LANGUAGE_LOCK_TURN  = 1          # lock language after turn 1 (first message)
MAX_TOOL_LOOPS      = 5          # safety cap on agentic tool-call loops


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityState:
    """Tracks all financial entities extracted from the conversation."""
    monthly_income:           float | None = None
    property_value:           float | None = None
    loan_amount_requested:    float | None = None
    employment_status:        str   | None = None   # "salaried" | "self_employed"
    existing_emi_obligations: float | None = None
    tenure_years:             int   | None = None

    def missing_required(self) -> list[str]:
        required = {
            "monthly_income":        self.monthly_income,
            "property_value":        self.property_value,
            "loan_amount_requested": self.loan_amount_requested,
            "employment_status":     self.employment_status,
        }
        return [k for k, v in required.items() if v is None]

    def to_dict(self) -> dict:
        return {
            "monthly_income":           self.monthly_income,
            "property_value":           self.property_value,
            "loan_amount_requested":    self.loan_amount_requested,
            "employment_status":        self.employment_status,
            "existing_emi_obligations": self.existing_emi_obligations,
            "tenure_years":             self.tenure_years,
        }


@dataclass
class AgentState:
    """Full mutable state of the agent across turns."""
    locked_language:   str          = "unknown"   # english/hindi/marathi/tamil/unknown
    language_locked:   bool         = False
    user_turn_count:   int          = 0
    entities:          EntityState  = field(default_factory=EntityState)
    eligibility_status: str         = "unknown"   # unknown/eligible/not_eligible
    tool_called:       str          = "none"       # none/calculate_emi/check_eligibility/both
    lead_intent_score: int          = 0
    handoff_triggered: bool         = False
    current_step:      str          = "greet"
    last_tool_results: dict         = field(default_factory=dict)

    def debug_dict(self) -> dict:
        """Serialisable snapshot for the Streamlit debug panel."""
        return {
            "locked_language":    self.locked_language,
            "language_label":     LANGUAGE_LABELS.get(self.locked_language, "Unknown"),
            "language_locked":    self.language_locked,
            "user_turn_count":    self.user_turn_count,
            "entities":           self.entities.to_dict(),
            "missing_fields":     self.entities.missing_required(),
            "eligibility_status": self.eligibility_status,
            "tool_called":        self.tool_called,
            "lead_intent_score":  self.lead_intent_score,
            "handoff_triggered":  self.handoff_triggered,
            "current_step":       self.current_step,
            "last_tool_results":  self.last_tool_results,
        }


@dataclass
class TurnResult:
    """What brain.py returns to app.py after each turn."""
    reply_text:       str          # cleaned reply for TTS + display
    agent_state:      AgentState   # full updated state
    handoff:          bool         # True if handoff was triggered this turn
    tool_results:     dict         # raw tool outputs (for debug panel)
    latency_ms:       float        # total LLM + tool latency
    raw_internal_state: dict       # parsed INTERNAL_STATE block from LLM


# ─────────────────────────────────────────────────────────────────────────────
# Brain class
# ─────────────────────────────────────────────────────────────────────────────

class LoanCounselorBrain:
    """
    Stateful LLM brain for the HomeFirst Vernacular Loan Counselor.

    Usage:
        brain = LoanCounselorBrain()
        result = brain.chat("Mujhe home loan chahiye")
        print(result.reply_text)
    """

    def __init__(self, rag_retriever=None):
        """
        Args:
            rag_retriever: Optional callable(query: str) -> str
                           Returns relevant FAQ context for RAG.
                           If None, RAG is disabled.
        """
        self.client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.state        = AgentState()
        self.history: list[dict] = []   # OpenAI-style message list
        self.rag          = rag_retriever
        self.session_id   = str(uuid.uuid4())   # unique ID per conversation session
        logger.info("LoanCounselorBrain initialised. Model: %s | Session: %s", MODEL, self.session_id)

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> TurnResult:
        """
        Process one user turn and return a TurnResult.
        This is the only method app.py needs to call.
        """
        t0 = time.time()
        self.state.user_turn_count += 1

        # 1. Detect & lock language BEFORE building the system prompt
        #    so the very first LLM call already gets the correct language.
        if not self.state.language_locked:
            self._detect_and_lock_language(user_message)

        # 2. Optionally inject RAG context
        augmented_message = self._augment_with_rag(user_message)

        # 3. Append user turn to history
        self.history.append({"role": "user", "content": augmented_message})

        # 4. Run agentic loop (system prompt is now built with locked language)
        reply_text, tool_results, internal_state = self._agentic_loop()

        # 5. Parse internal state & update agent state
        self._sync_state_from_internal(internal_state)

        # 6. Check handoff
        handoff_this_turn = self._evaluate_handoff()

        latency_ms = (time.time() - t0) * 1000
        logger.info(
            "Turn %d | Lang: %s (locked=%s) | Step: %s | Score: %d | Latency: %.0fms",
            self.state.user_turn_count,
            self.state.locked_language,
            self.state.language_locked,
            self.state.current_step,
            self.state.lead_intent_score,
            latency_ms,
        )

        return TurnResult(
            reply_text        = reply_text,
            agent_state       = self.state,
            handoff           = handoff_this_turn,
            tool_results      = tool_results,
            latency_ms        = latency_ms,
            raw_internal_state = internal_state,
        )

    def reset(self):
        """Reset conversation state (new session)."""
        # Save final lead record before resetting
        db.save_lead(self.session_id, self.state)
        self.state      = AgentState()
        self.history    = []
        self.session_id = str(uuid.uuid4())
        logger.info("Brain reset — new session started. New session: %s", self.session_id)

    # ── Language detection ───────────────────────────────────────────────────

    def _detect_and_lock_language(self, user_message: str):
        """
        Detect and lock the session language from the user's message.

        Strategy (fastest first):
          1. Unicode script fast-path — checks for Devanagari / Tamil codepoints.
             If found, locks immediately without an API call.
          2. GPT-4o classification — single-token response for ambiguous text.

        Always locks on the first user turn (LANGUAGE_LOCK_TURN = 1).
        After locking, injects a system reminder into conversation history
        so all subsequent LLM calls see an explicit enforcement notice.
        """
        detected = self._fast_path_detect(user_message)

        if detected == "unknown":
            # Fall back to LLM classification
            try:
                from agent.prompts import build_language_detection_prompt
                detection_prompt = build_language_detection_prompt(user_message)
                response = self.client.chat.completions.create(
                    model      = MODEL,
                    max_tokens = 10,
                    messages   = [{"role": "user", "content": detection_prompt}],
                )
                detected = response.choices[0].message.content.strip().lower()
            except Exception as e:
                logger.error("Language detection API call failed: %s", e)
                detected = "english"   # safe default

        valid = {"english", "hindi", "marathi", "tamil"}
        if detected not in valid:
            logger.warning("Unrecognised language '%s', defaulting to english.", detected)
            detected = "english"

        self.state.locked_language = detected
        logger.info("Language detected: %s (turn %d)", detected, self.state.user_turn_count)

        # Lock after LANGUAGE_LOCK_TURN turns
        if self.state.user_turn_count >= LANGUAGE_LOCK_TURN:
            self.state.language_locked = True
            logger.info("Language LOCKED: %s", self.state.locked_language)

            # ── Inject hard enforcement reminder into history ─────────────────
            # This message appears as a "system" injection in the conversation
            # so every subsequent LLM response is reminded of the lock.
            from agent.prompts import LANGUAGE_PERSONA, LANGUAGE_LABELS
            lang_label   = LANGUAGE_LABELS.get(detected, detected.capitalize())
            lang_enforce = LANGUAGE_PERSONA.get(detected, LANGUAGE_PERSONA["english"])
            reminder = (
                f"[SYSTEM — LANGUAGE LOCK ACTIVE]\n"
                f"The session language has been permanently locked to: {lang_label}.\n"
                f"{lang_enforce}\n"
                f"You MUST generate ALL remaining responses in {lang_label} ONLY. "
                f"This overrides any in-message language switch by the user."
            )
            # Insert as an assistant turn (disguised context injection)
            # We use role=system via a prepended message — done in _agentic_loop.
            self._language_lock_reminder = reminder
            logger.info("Language lock reminder injected into history.")

    def _fast_path_detect(self, text: str) -> str:
        """
        Quickly detect language by scanning Unicode codepoints.
        Returns a language key or 'unknown' if inconclusive.

        Guard: if the text is predominantly Latin/ASCII (English),
        return 'unknown' immediately \u2014 never misclassify it as Hindi.
        """
        total = len(text.strip())
        if total == 0:
            return "unknown"

        # Count script characters
        latin_count      = sum(1 for c in text if c.isascii() and c.isalpha())
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        tamil_count      = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')

        # If text is majority Latin \u2014 it's English (or Romanised Hinglish)
        # Let GPT-4o classify it; don't touch it with Devanagari heuristics.
        if latin_count > devanagari_count and latin_count > tamil_count:
            return "unknown"   # will fall through to LLM classification

        if tamil_count >= 2:
            return "tamil"

        if devanagari_count >= 2:
            # Distinguish Marathi from Hindi using common Marathi-specific words
            marathi_markers = ["आहे", "नाही", "मला", "आणि", "आहेत", "करायचे", "घर", "मराठी"]
            if any(m in text for m in marathi_markers):
                return "marathi"
            return "hindi"

        return "unknown"  # Let LLM decide

    # ── RAG augmentation ─────────────────────────────────────────────────────

    def _augment_with_rag(self, user_message: str) -> str:
        """
        If a RAG retriever is available and the message looks like a policy FAQ,
        prepend relevant context to the user message.
        """
        if not self.rag:
            return user_message

        faq_keywords = [
            "document", "required", "eligibility", "criteria", "what do i need",
            "kya chahiye", "documents", "कागज", "दस्तावेज", "कागदपत्रे",
        ]
        lower = user_message.lower()
        is_faq = any(kw in lower for kw in faq_keywords)

        if not is_faq:
            return user_message

        try:
            context = self.rag(user_message)
            if context:
                augmented = (
                    f"[POLICY CONTEXT — use this to answer the user's question]\n"
                    f"{context}\n\n"
                    f"[USER MESSAGE]\n{user_message}"
                )
                logger.info("RAG context injected (%d chars)", len(context))
                return augmented
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)

        return user_message

    # ── Agentic tool-calling loop ────────────────────────────────────────────

    def _agentic_loop(self) -> tuple[str, dict, dict]:
        """
        Run the LLM in a tool-calling loop (GPT-4o).
        The system prompt is rebuilt each loop with the current locked language
        so it is always authoritative.
        Returns (reply_text, tool_results_dict, internal_state_dict).
        """
        tool_results:   dict = {}
        internal_state: dict = {}
        system_prompt = build_system_prompt(self.state.locked_language)

        # ── Build messages with optional language-lock reminder ───────────────
        reminder_msg = getattr(self, "_language_lock_reminder", None)

        for loop_iter in range(MAX_TOOL_LOOPS):
            # Build full message list: system + (optional lock reminder) + history
            messages = [{"role": "system", "content": system_prompt}]
            if reminder_msg and self.state.language_locked:
                # Inject lock reminder as a second system message
                messages.append({"role": "system", "content": reminder_msg})
            messages += self.history

            response = self.client.chat.completions.create(
                model      = MODEL,
                max_tokens = MAX_TOKENS,
                tools      = TOOL_SCHEMAS,
                messages   = messages,
            )

            message       = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # ── Tool calls block ─────────────────────────────────────────────
            if finish_reason == "tool_calls":
                # Append the assistant's tool-call turn to history
                self.history.append({
                    "role":       "assistant",
                    "content":    message.content,   # may be None
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })

                # Execute each tool and append its result to history
                for tool_call in (message.tool_calls or []):
                    tool_name  = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    logger.info("Tool called: %s | Input: %s", tool_name, tool_input)

                    result = dispatch_tool(tool_name, tool_input)
                    tool_results[tool_name] = result
                    self.state.last_tool_results = tool_results

                    # Track which tools were called
                    if self.state.tool_called == "none":
                        self.state.tool_called = tool_name
                    elif self.state.tool_called != "both" and self.state.tool_called != tool_name:
                        self.state.tool_called = "both"

                    # Append tool result message (role=tool)
                    self.history.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      json.dumps(result),
                    })

                    # Update eligibility status from tool result
                    if tool_name == "check_eligibility" and "eligible" in result:
                        self.state.eligibility_status = (
                            "eligible" if result["eligible"] else "not_eligible"
                        )

                continue  # loop again — let LLM respond to tool results

            # ── End turn — extract text ──────────────────────────────────────
            if finish_reason == "stop":
                full_text = message.content or ""

                # Parse and strip INTERNAL_STATE block
                reply_text, internal_state = self._extract_internal_state(full_text)

                # Append clean assistant reply to history
                self.history.append({
                    "role":    "assistant",
                    "content": reply_text,
                })
                return reply_text, tool_results, internal_state

            # Unexpected finish reason
            logger.warning("Unexpected finish_reason: %s", finish_reason)
            break

        # Fallback if loop exhausted
        fallback = "I'm sorry, I ran into an issue processing your request. Please try again."
        return fallback, tool_results, internal_state

    # ── INTERNAL_STATE parsing ───────────────────────────────────────────────

    def _extract_internal_state(self, full_text: str) -> tuple[str, dict]:
        """
        Split the LLM output into:
          - reply_text : what the user sees/hears (no JSON block)
          - internal_state : parsed dict from <INTERNAL_STATE>…</INTERNAL_STATE>

        Strips the INTERNAL_STATE block from the spoken reply.
        """
        internal_state: dict = {}

        pattern = r"<INTERNAL_STATE>\s*(.*?)\s*</INTERNAL_STATE>"
        match = re.search(pattern, full_text, re.DOTALL)

        if match:
            raw_json = match.group(1).strip()
            try:
                internal_state = json.loads(raw_json)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse INTERNAL_STATE JSON: %s\nRaw: %s", e, raw_json)

            # Remove the block from the reply the user hears
            reply_text = re.sub(pattern, "", full_text, flags=re.DOTALL).strip()
        else:
            reply_text = full_text.strip()
            logger.debug("No INTERNAL_STATE block found in response.")

        # Also strip any [HANDOFF TRIGGERED...] line from spoken reply
        reply_text = re.sub(r"\[HANDOFF TRIGGERED[^\]]*\]", "", reply_text).strip()

        return reply_text, internal_state

    # ── Sync agent state from parsed INTERNAL_STATE ──────────────────────────

    def _sync_state_from_internal(self, internal: dict):
        """
        Update AgentState from the INTERNAL_STATE dict emitted by the LLM.
        Always prefer tool-verified eligibility over LLM-reported eligibility.
        """
        if not internal:
            return

        entities = internal.get("entities", {})

        # Update entities only if LLM provides a non-null value
        def _maybe_update(field: str, val):
            if val is not None:
                setattr(self.state.entities, field, val)

        _maybe_update("monthly_income",           entities.get("monthly_income"))
        _maybe_update("property_value",           entities.get("property_value"))
        _maybe_update("loan_amount_requested",    entities.get("loan_amount_requested"))
        _maybe_update("employment_status",        entities.get("employment_status"))
        _maybe_update("existing_emi_obligations", entities.get("existing_emi_obligations"))
        _maybe_update("tenure_years",             entities.get("tenure_years"))

        # Language
        detected_lang = internal.get("language_detected", "unknown")
        if detected_lang != "unknown" and not self.state.language_locked:
            self.state.locked_language = detected_lang

        # Step
        step = internal.get("current_step")
        if step:
            self.state.current_step = step

        # Intent score (LLM-reported, use as-is)
        score = internal.get("lead_intent_score")
        if isinstance(score, int):
            self.state.lead_intent_score = score

        # Eligibility — only update from LLM if tool hasn't set it yet
        llm_eligibility = internal.get("eligibility_status", "unknown")
        if self.state.eligibility_status == "unknown" and llm_eligibility != "unknown":
            self.state.eligibility_status = llm_eligibility

    # ── Handoff evaluation ───────────────────────────────────────────────────

    def _evaluate_handoff(self) -> bool:
        """
        Trigger handoff if:
          - eligibility confirmed as eligible (by tool, not LLM)
          - lead intent score >= threshold
          - not already triggered
        """
        if (
            not self.state.handoff_triggered
            and self.state.eligibility_status == "eligible"
            and self.state.lead_intent_score >= HANDOFF_SCORE_GATE
        ):
            self.state.handoff_triggered = True
            print("\n[HANDOFF TRIGGERED: Routing to Human RM]\n")
            logger.info("HANDOFF TRIGGERED — score: %d", self.state.lead_intent_score)

            # ── Persist to Supabase ──────────────────────────────────────
            approved_amount = 0.0
            if "check_eligibility" in self.state.last_tool_results:
                approved_amount = self.state.last_tool_results["check_eligibility"].get(
                    "approved_loan_amount", 0.0
                )
            db.save_handoff(self.session_id, self.state, approved_amount)
            db.save_lead(self.session_id, self.state)
            # ─────────────────────────────────────────────────────────────

            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test (run: python agent/brain.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    def _print_turn(turn: int, user: str, result: TurnResult):
        print(f"\n{'═'*60}")
        print(f"  TURN {turn}")
        print(f"{'═'*60}")
        print(f"  USER  : {user}")
        print(f"  BOT   : {textwrap.fill(result.reply_text, 56, subsequent_indent='          ')}")
        print(f"  Lang  : {result.agent_state.locked_language} (locked={result.agent_state.language_locked})")
        print(f"  Step  : {result.agent_state.current_step}")
        print(f"  Score : {result.agent_state.lead_intent_score}")
        print(f"  Entities: {result.agent_state.entities.to_dict()}")
        if result.tool_results:
            print(f"  Tools : {list(result.tool_results.keys())}")
        if result.handoff:
            print("  *** [HANDOFF TRIGGERED] ***")
        print(f"  Latency: {result.latency_ms:.0f}ms")

    print("Starting CLI test — HomeFirst Loan Counselor Brain")
    brain = LoanCounselorBrain()

    conversations = [
        "Namaste, mujhe home loan chahiye",                                  # Hindi greeting
        "Meri salary 55 hazaar hai per month",                               # income in Hinglish
        "Maine ek flat dekha hai, uski value 35 lakh hai",                   # property value
        "Mujhe 25 lakh ka loan chahiye. Main salaried hoon.",                # loan + employment
        "Koi existing EMI nahi hai. Aur tenure 20 saal theek rahega.",       # remaining fields
        "Haan bilkul, main kharidna chahta hoon. Agla step kya hai?",        # high intent → handoff
    ]

    for i, msg in enumerate(conversations, 1):
        result = brain.chat(msg)
        _print_turn(i, msg, result)

    # Save final lead record at end of CLI test
    db.save_lead(brain.session_id, brain.state)
    print("\n✅ Session saved to Supabase (if credentials are set).")
