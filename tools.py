"""
tools.py — HomeFirst Vernacular Loan Counselor
Deterministic loan eligibility and EMI calculation tools.
These are called by the LLM via function/tool calling — never calculated by the LLM itself.

Indian Home Loan Rules Encoded:
  - FOIR (Fixed Obligation to Income Ratio): max 50%
  - LTV  (Loan to Value Ratio)             : max 90% of property value
  - Minimum monthly income                 : ₹15,000
  - Interest rate                          : 9.5% p.a. (HomeFirst standard)
  - Tenure range                           : 5–20 years (default: 20)
  - Employment types accepted              : salaried, self_employed
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────
# Constants (HomeFirst-aligned parameters)
# ─────────────────────────────────────────────

ANNUAL_INTEREST_RATE: float = 9.5          # % per annum
DEFAULT_TENURE_YEARS: int   = 20           # years
MIN_TENURE_YEARS: int       = 5
MAX_TENURE_YEARS: int       = 20

MAX_FOIR: float             = 0.50         # 50% of net monthly income
MAX_LTV:  float             = 0.90         # 90% of property value
MIN_MONTHLY_INCOME: float   = 15_000.0     # ₹15,000/month minimum

VALID_EMPLOYMENT: set[str]  = {"salaried", "self_employed"}


# ─────────────────────────────────────────────
# Return dataclasses (serialisable to JSON for LLM)
# ─────────────────────────────────────────────

@dataclass
class EMIResult:
    monthly_emi:        float
    loan_amount:        float
    tenure_years:       int
    annual_rate_pct:    float
    total_payment:      float
    total_interest:     float
    breakdown: dict     = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "monthly_emi":      round(self.monthly_emi, 2),
            "loan_amount":      self.loan_amount,
            "tenure_years":     self.tenure_years,
            "annual_rate_pct":  self.annual_rate_pct,
            "total_payment":    round(self.total_payment, 2),
            "total_interest":   round(self.total_interest, 2),
            "breakdown":        self.breakdown,
        }


@dataclass
class EligibilityResult:
    eligible:               bool
    approved_loan_amount:   float
    rejection_reasons:      list[str]
    warnings:               list[str]
    foir_used:              float          # ratio, e.g. 0.42
    ltv_used:               float          # ratio, e.g. 0.85
    foir_capacity_amount:   float          # max loan FOIR allows
    ltv_capacity_amount:    float          # max loan LTV allows
    recommended_tenure:     int
    details:                str

    def to_dict(self) -> dict:
        return {
            "eligible":               self.eligible,
            "approved_loan_amount":   self.approved_loan_amount,
            "rejection_reasons":      self.rejection_reasons,
            "warnings":               self.warnings,
            "foir_used_pct":          round(self.foir_used * 100, 1),
            "ltv_used_pct":           round(self.ltv_used * 100, 1),
            "foir_capacity_amount":   round(self.foir_capacity_amount, 0),
            "ltv_capacity_amount":    round(self.ltv_capacity_amount, 0),
            "recommended_tenure_yrs": self.recommended_tenure,
            "details":                self.details,
        }


# ─────────────────────────────────────────────
# Core helper: EMI formula
# ─────────────────────────────────────────────

def _compute_emi(principal: float, annual_rate_pct: float, tenure_years: int) -> float:
    """
    Standard reducing-balance EMI formula:
        EMI = P * r * (1+r)^n  /  ((1+r)^n - 1)
    where r = monthly interest rate, n = number of months.
    """
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / (12 * 100)          # monthly rate
    n = tenure_years * 12                     # total months
    emi = principal * r * math.pow(1 + r, n) / (math.pow(1 + r, n) - 1)
    return emi


def _max_loan_from_foir(
    monthly_income: float,
    existing_obligations: float,
    annual_rate_pct: float,
    tenure_years: int,
) -> float:
    """
    Back-calculate max principal where EMI ≤ (MAX_FOIR * income - existing_obligations).
    Formula: P = EMI_max * ((1+r)^n - 1) / (r * (1+r)^n)
    """
    max_emi = (MAX_FOIR * monthly_income) - existing_obligations
    if max_emi <= 0:
        return 0.0
    r = annual_rate_pct / (12 * 100)
    n = tenure_years * 12
    factor = math.pow(1 + r, n)
    principal = max_emi * (factor - 1) / (r * factor)
    return principal


# ─────────────────────────────────────────────
# Tool 1: calculate_emi
# ─────────────────────────────────────────────

def calculate_emi(
    loan_amount: float,
    tenure_years: int   = DEFAULT_TENURE_YEARS,
    annual_rate_pct: float = ANNUAL_INTEREST_RATE,
) -> dict:
    """
    Calculate monthly EMI for a given loan amount.

    Args:
        loan_amount     : Principal in ₹ (e.g. 1500000 for 15L)
        tenure_years    : Repayment period in years (5–20, default 20)
        annual_rate_pct : Annual interest rate in % (default 9.5)

    Returns:
        dict with monthly_emi, total_payment, total_interest, etc.
    """
    # ── Validation ──────────────────────────────────────────────────────────
    errors = []
    if loan_amount <= 0:
        errors.append("loan_amount must be positive.")
    if not (MIN_TENURE_YEARS <= tenure_years <= MAX_TENURE_YEARS):
        errors.append(f"tenure_years must be between {MIN_TENURE_YEARS} and {MAX_TENURE_YEARS}.")
    if annual_rate_pct <= 0:
        errors.append("annual_rate_pct must be positive.")
    if errors:
        return {"error": " | ".join(errors)}

    # ── Calculation ─────────────────────────────────────────────────────────
    emi           = _compute_emi(loan_amount, annual_rate_pct, tenure_years)
    n             = tenure_years * 12
    total_payment = emi * n
    total_interest = total_payment - loan_amount

    result = EMIResult(
        monthly_emi     = emi,
        loan_amount     = loan_amount,
        tenure_years    = tenure_years,
        annual_rate_pct = annual_rate_pct,
        total_payment   = total_payment,
        total_interest  = total_interest,
        breakdown = {
            "monthly_emi_inr":        round(emi, 2),
            "total_months":           n,
            "total_amount_paid_inr":  round(total_payment, 2),
            "total_interest_paid_inr": round(total_interest, 2),
            "interest_as_pct_of_loan": round((total_interest / loan_amount) * 100, 1),
        }
    )
    return result.to_dict()


# ─────────────────────────────────────────────
# Tool 2: check_eligibility
# ─────────────────────────────────────────────

def check_eligibility(
    monthly_income: float,
    property_value: float,
    loan_amount_requested: float,
    employment_status: Literal["salaried", "self_employed"],
    existing_emi_obligations: float = 0.0,
    tenure_years: int = DEFAULT_TENURE_YEARS,
    annual_rate_pct: float = ANNUAL_INTEREST_RATE,
) -> dict:
    """
    Check home loan eligibility using HomeFirst rules.

    Args:
        monthly_income          : Net monthly income in ₹
        property_value          : Property value in ₹
        loan_amount_requested   : Loan amount the user wants in ₹
        employment_status       : "salaried" or "self_employed"
        existing_emi_obligations: Sum of any existing EMIs (₹/month), default 0
        tenure_years            : Requested loan tenure (default 20 years)
        annual_rate_pct         : Interest rate % (default 9.5)

    Returns:
        dict with eligible (bool), approved_loan_amount, reasons, warnings, etc.
    """
    rejection_reasons = []
    warnings          = []

    # ── Normalise employment status ──────────────────────────────────────────
    employment_status = employment_status.strip().lower().replace(" ", "_").replace("-", "_")
    if employment_status not in VALID_EMPLOYMENT:
        return {
            "error": (
                f"Invalid employment_status '{employment_status}'. "
                f"Must be one of: {VALID_EMPLOYMENT}"
            )
        }

    # ── Rule 1: Minimum income ───────────────────────────────────────────────
    if monthly_income < MIN_MONTHLY_INCOME:
        rejection_reasons.append(
            f"Monthly income ₹{monthly_income:,.0f} is below the minimum "
            f"required ₹{MIN_MONTHLY_INCOME:,.0f}."
        )

    # ── Rule 2: LTV cap ─────────────────────────────────────────────────────
    ltv_capacity = MAX_LTV * property_value          # max loan by LTV rule
    ltv_used     = loan_amount_requested / property_value if property_value > 0 else 0

    if loan_amount_requested > ltv_capacity:
        rejection_reasons.append(
            f"Requested loan ₹{loan_amount_requested:,.0f} exceeds LTV cap "
            f"({int(MAX_LTV*100)}% of ₹{property_value:,.0f} = ₹{ltv_capacity:,.0f})."
        )

    # ── Rule 3: FOIR check ───────────────────────────────────────────────────
    foir_capacity = _max_loan_from_foir(
        monthly_income, existing_emi_obligations, annual_rate_pct, tenure_years
    )
    proposed_emi  = _compute_emi(loan_amount_requested, annual_rate_pct, tenure_years)
    foir_used     = (proposed_emi + existing_emi_obligations) / monthly_income if monthly_income > 0 else 1.0

    if foir_used > MAX_FOIR:
        rejection_reasons.append(
            f"FOIR {foir_used*100:.1f}% exceeds maximum allowed {int(MAX_FOIR*100)}%. "
            f"Based on income, max eligible EMI is "
            f"₹{(MAX_FOIR * monthly_income - existing_emi_obligations):,.0f}/month."
        )

    # ── Rule 4: Tenure bounds ────────────────────────────────────────────────
    if not (MIN_TENURE_YEARS <= tenure_years <= MAX_TENURE_YEARS):
        rejection_reasons.append(
            f"Tenure {tenure_years} years is outside the allowed range "
            f"({MIN_TENURE_YEARS}–{MAX_TENURE_YEARS} years)."
        )

    # ── Rule 5: Self-employed extra caution ──────────────────────────────────
    if employment_status == "self_employed":
        warnings.append(
            "Self-employed applicants require 2 years of ITR and business proof. "
            "Income verification may take additional time."
        )
        # Slightly conservative: apply 90% of stated income for self-employed
        effective_income   = monthly_income * 0.90
        foir_capacity_adj  = _max_loan_from_foir(
            effective_income, existing_emi_obligations, annual_rate_pct, tenure_years
        )
        if foir_capacity_adj < foir_capacity:
            foir_capacity = foir_capacity_adj
            warnings.append(
                "Income haircut of 10% applied for self-employed profile — "
                f"adjusted FOIR capacity ₹{foir_capacity:,.0f}."
            )

    # ── Determine approved amount ────────────────────────────────────────────
    # Approved = min(requested, LTV cap, FOIR capacity) — but only if eligible
    approved_loan_amount = min(loan_amount_requested, ltv_capacity, foir_capacity)
    approved_loan_amount = max(0.0, approved_loan_amount)   # never negative

    # Warn if we had to reduce the amount (but not reject)
    if not rejection_reasons and approved_loan_amount < loan_amount_requested:
        warnings.append(
            f"Requested ₹{loan_amount_requested:,.0f} reduced to "
            f"₹{approved_loan_amount:,.0f} due to FOIR/LTV constraints."
        )

    eligible = len(rejection_reasons) == 0

    # ── Human-readable summary ───────────────────────────────────────────────
    if eligible:
        details = (
            f"Applicant is ELIGIBLE for a home loan of ₹{approved_loan_amount:,.0f} "
            f"at {annual_rate_pct}% p.a. over {tenure_years} years. "
            f"Estimated EMI: ₹{_compute_emi(approved_loan_amount, annual_rate_pct, tenure_years):,.0f}/month. "
            f"FOIR: {foir_used*100:.1f}% | LTV: {ltv_used*100:.1f}%."
        )
    else:
        details = (
            f"Applicant is NOT ELIGIBLE. Reasons: {'; '.join(rejection_reasons)}"
        )

    result = EligibilityResult(
        eligible              = eligible,
        approved_loan_amount  = round(approved_loan_amount, 0),
        rejection_reasons     = rejection_reasons,
        warnings              = warnings,
        foir_used             = round(foir_used, 4),
        ltv_used              = round(ltv_used, 4),
        foir_capacity_amount  = round(foir_capacity, 0),
        ltv_capacity_amount   = round(ltv_capacity, 0),
        recommended_tenure    = tenure_years,
        details               = details,
    )
    return result.to_dict()


# ─────────────────────────────────────────────
# Tool schemas — passed to LLM for tool calling
# ─────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_emi",
            "description": (
                "Calculate the monthly EMI for a home loan using the reducing-balance method. "
                "Call this when the user wants to know their monthly repayment amount. "
                "Do NOT do this math yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "number",
                        "description": "Principal loan amount in INR (e.g. 1500000 for ₹15L)."
                    },
                    "tenure_years": {
                        "type": "integer",
                        "description": f"Loan tenure in years. Must be between {MIN_TENURE_YEARS} and {MAX_TENURE_YEARS}. Default is {DEFAULT_TENURE_YEARS}.",
                    },
                    "annual_rate_pct": {
                        "type": "number",
                        "description": "Annual interest rate in %. Default is 9.5 (HomeFirst standard).",
                    },
                },
                "required": ["loan_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": (
                "Check whether the user is eligible for a HomeFirst home loan and compute the "
                "maximum approved loan amount. Call this ONLY when you have all required fields: "
                "monthly_income, property_value, loan_amount_requested, and employment_status. "
                "Do NOT calculate eligibility or FOIR yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_income": {
                        "type": "number",
                        "description": "Net monthly income in INR."
                    },
                    "property_value": {
                        "type": "number",
                        "description": "Market value of the property in INR."
                    },
                    "loan_amount_requested": {
                        "type": "number",
                        "description": "Loan amount the user wants in INR."
                    },
                    "employment_status": {
                        "type": "string",
                        "enum": ["salaried", "self_employed"],
                        "description": "Employment type of the applicant."
                    },
                    "existing_emi_obligations": {
                        "type": "number",
                        "description": "Sum of existing EMI payments per month in INR. Default 0.",
                    },
                    "tenure_years": {
                        "type": "integer",
                        "description": f"Requested loan tenure in years ({MIN_TENURE_YEARS}–{MAX_TENURE_YEARS}). Default {DEFAULT_TENURE_YEARS}.",
                    },
                },
                "required": [
                    "monthly_income",
                    "property_value",
                    "loan_amount_requested",
                    "employment_status",
                ],
            },
        },
    },
]


# ─────────────────────────────────────────────
# Tool dispatcher — called by brain.py
# ─────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Route a tool call from the LLM to the correct Python function.
    Returns a dict that is sent back to the LLM as a tool_result.
    """
    if tool_name == "calculate_emi":
        return calculate_emi(**tool_input)
    elif tool_name == "check_eligibility":
        return check_eligibility(**tool_input)
    else:
        return {"error": f"Unknown tool '{tool_name}'."}


# ─────────────────────────────────────────────
# Quick self-test (run: python tools.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("TEST 1 — EMI for ₹15L, 20 years @ 9.5%")
    print("=" * 60)
    emi_result = calculate_emi(loan_amount=1_500_000, tenure_years=20)
    print(json.dumps(emi_result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 2 — Eligible salaried applicant")
    print("=" * 60)
    result = check_eligibility(
        monthly_income        = 60_000,
        property_value        = 2_500_000,
        loan_amount_requested = 1_800_000,
        employment_status     = "salaried",
    )
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 3 — Rejected: income too low")
    print("=" * 60)
    result = check_eligibility(
        monthly_income        = 10_000,
        property_value        = 1_000_000,
        loan_amount_requested = 800_000,
        employment_status     = "salaried",
    )
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 4 — Rejected: FOIR breach")
    print("=" * 60)
    result = check_eligibility(
        monthly_income          = 40_000,
        property_value          = 3_000_000,
        loan_amount_requested   = 2_700_000,
        employment_status       = "salaried",
        existing_emi_obligations = 15_000,
    )
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print("TEST 5 — Self-employed with income haircut")
    print("=" * 60)
    result = check_eligibility(
        monthly_income        = 80_000,
        property_value        = 4_000_000,
        loan_amount_requested = 2_500_000,
        employment_status     = "self_employed",
    )
    print(json.dumps(result, indent=2))
