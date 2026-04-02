"""
db.py — HomeFirst Vernacular Loan Counselor
Supabase database layer.

Handles:
  - save_lead()    : Write user financial profile (end of session / on handoff)
  - save_handoff() : Log handoff trigger events
"""

from __future__ import annotations

import os
import uuid
import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Lazy Supabase client (only initialised if credentials are present) ─────────

_client = None


def _get_client():
    """Return a cached Supabase client, or None if credentials are missing."""
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key or url.startswith("your_"):
        logger.warning("Supabase credentials not set — DB writes disabled.")
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("Supabase client initialised: %s", url)
        return _client
    except ImportError:
        logger.error("supabase package not installed. Run: pip install supabase")
        return None
    except Exception as e:
        logger.error("Failed to initialise Supabase client: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_lead(session_id: str, agent_state) -> bool:
    """
    Write user financial profile to the `leads` table.

    Args:
        session_id  : Unique session identifier (UUID string)
        agent_state : AgentState dataclass from brain.py

    Returns:
        True if saved successfully, False otherwise.
    """
    client = _get_client()
    if not client:
        return False

    entities = agent_state.entities

    row = {
        "session_id":             session_id,
        "language":               agent_state.locked_language,
        "monthly_income":         entities.monthly_income,
        "property_value":         entities.property_value,
        "loan_amount_requested":  entities.loan_amount_requested,
        "employment_status":      entities.employment_status,
        "existing_emi":           entities.existing_emi_obligations,
        "tenure_years":           entities.tenure_years,
        "eligibility_status":     agent_state.eligibility_status,
        "lead_intent_score":      agent_state.lead_intent_score,
        "handoff_triggered":      agent_state.handoff_triggered,
    }

    try:
        result = client.table("leads").insert(row).execute()
        logger.info("Lead saved to Supabase | session: %s", session_id)
        return True
    except Exception as e:
        logger.error("Failed to save lead: %s", e)
        return False


def save_handoff(session_id: str, agent_state, approved_loan_amount: float = 0.0) -> bool:
    """
    Log a handoff trigger event to the `handoffs` table.

    Args:
        session_id          : Unique session identifier
        agent_state         : AgentState dataclass from brain.py
        approved_loan_amount: Approved amount from the last check_eligibility call

    Returns:
        True if saved successfully, False otherwise.
    """
    client = _get_client()
    if not client:
        return False

    row = {
        "session_id":           session_id,
        "lead_intent_score":    agent_state.lead_intent_score,
        "eligibility_status":   agent_state.eligibility_status,
        "approved_loan_amount": approved_loan_amount,
        "language":             agent_state.locked_language,
        "entities": {
            "monthly_income":           agent_state.entities.monthly_income,
            "property_value":           agent_state.entities.property_value,
            "loan_amount_requested":    agent_state.entities.loan_amount_requested,
            "employment_status":        agent_state.entities.employment_status,
            "existing_emi_obligations": agent_state.entities.existing_emi_obligations,
            "tenure_years":             agent_state.entities.tenure_years,
        },
    }

    try:
        result = client.table("handoffs").insert(row).execute()
        logger.info(
            "Handoff saved to Supabase | session: %s | score: %d",
            session_id, agent_state.lead_intent_score,
        )
        return True
    except Exception as e:
        logger.error("Failed to save handoff: %s", e)
        return False
