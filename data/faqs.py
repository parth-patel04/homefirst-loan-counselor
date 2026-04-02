"""
data/faqs.py — HomeFirst Vernacular Loan Counselor
10 mock HomeFirst policy FAQ documents for RAG.

These are loaded into ChromaDB by rag.py.
Based on publicly available HomeFirst Finance information.
"""

FAQ_DOCUMENTS = [
    {
        "id":       "faq_001",
        "category": "eligibility",
        "title":    "Basic Home Loan Eligibility Criteria",
        "content":  """
HomeFirst Finance Home Loan Eligibility Criteria:

1. Age: Minimum 21 years at loan application. Maximum 65 years at loan maturity.
2. Income: Minimum net monthly income of Rs 15,000 for salaried applicants.
   Self-employed applicants must show minimum annual income of Rs 2,00,000.
3. Employment: Both salaried and self-employed individuals are eligible.
   Salaried applicants need minimum 1 year of current employment.
   Self-employed applicants need minimum 2 years of business continuity.
4. Credit Score: Minimum CIBIL score of 650 is preferred.
   Applicants with no credit history may be evaluated on alternative parameters.
5. Property: The property must be located in areas where HomeFirst operates.
   Both under-construction and ready-to-move properties are eligible.
6. Co-applicant: A co-applicant (spouse or family member) can strengthen the application.
        """.strip(),
    },
    {
        "id":       "faq_002",
        "category": "documents_salaried",
        "title":    "Documents Required for Salaried Applicants",
        "content":  """
Documents Required for Salaried Home Loan Applicants at HomeFirst Finance:

Identity and Address Proof (any one of each):
- Aadhaar Card
- PAN Card
- Voter ID
- Passport
- Driving Licence

Income Documents:
- Latest 3 months salary slips
- Form 16 (last 2 years)
- Bank statements for last 6 months (salary account)
- Appointment letter or employment certificate (if less than 1 year in current job)

Property Documents:
- Allotment letter / Sale agreement
- Title deed / Index II
- Approved building plan
- NOC from builder/society
- Property tax receipts (last 2 years)

Loan Application:
- Duly filled HomeFirst loan application form
- 2 recent passport-sized photographs
        """.strip(),
    },
    {
        "id":       "faq_003",
        "category": "documents_self_employed",
        "title":    "Documents Required for Self-Employed Applicants",
        "content":  """
Documents Required for Self-Employed Home Loan Applicants at HomeFirst Finance:

Identity and Address Proof:
- Aadhaar Card (mandatory)
- PAN Card (mandatory)
- Any other government-issued ID

Business Proof:
- Business registration certificate / Shop Act licence
- GST registration certificate (if applicable)
- Partnership deed (for partnership firms)
- Certificate of Incorporation (for companies)

Income Documents:
- ITR (Income Tax Returns) for last 2 years with computation of income
- Profit & Loss statement for last 2 years (CA certified)
- Balance sheet for last 2 years (CA certified)
- Bank statements for last 12 months (business and personal accounts)

Property Documents:
- Same as salaried applicants (see FAQ 002)

Note: Self-employed applicants may face a 10 percent income haircut during
eligibility calculation due to income variability.
        """.strip(),
    },
    {
        "id":       "faq_004",
        "category": "loan_parameters",
        "title":    "HomeFirst Loan Amount, Tenure and Interest Rate",
        "content":  """
HomeFirst Finance Home Loan Parameters:

Loan Amount:
- Minimum loan: Rs 2,00,000 (2 Lakhs)
- Maximum loan: Rs 75,00,000 (75 Lakhs)
- Loan amount is subject to eligibility based on income and property value.

Loan Tenure:
- Minimum tenure: 5 years
- Maximum tenure: 20 years
- Longer tenure means lower EMI but higher total interest paid.

Interest Rate:
- Starting from 9.5% per annum (reducing balance method)
- Rate may vary based on applicant profile, loan amount, and market conditions.
- Both fixed and floating rate options may be available.

Processing Fee:
- Up to 2% of the loan amount + GST
- Non-refundable once processing begins.

Prepayment:
- Partial or full prepayment allowed without penalty for floating rate loans.
- Fixed rate loans may have prepayment charges.
        """.strip(),
    },
    {
        "id":       "faq_005",
        "category": "ltv_foir",
        "title":    "LTV Ratio and FOIR Explained",
        "content":  """
Understanding LTV and FOIR at HomeFirst Finance:

LTV (Loan to Value Ratio):
- HomeFirst finances up to 90% of the property value.
- Example: For a property worth Rs 30 Lakhs, maximum loan = Rs 27 Lakhs.
- The remaining 10% (Rs 3 Lakhs) must be arranged by the applicant as down payment.
- LTV may be lower for higher value properties or under-construction properties.

FOIR (Fixed Obligation to Income Ratio):
- Maximum FOIR allowed: 50% of net monthly income.
- FOIR includes the proposed new EMI plus all existing EMI obligations.
- Example: If net monthly income is Rs 50,000, total EMI obligations cannot exceed Rs 25,000.
- Existing obligations include: car loans, personal loans, credit card minimums, other home loans.

How they interact:
- The actual loan approved will be the LOWER of:
  (a) 90% of property value (LTV cap)
  (b) Maximum loan sustainable within 50% FOIR limit
  (c) The loan amount requested by the applicant.
        """.strip(),
    },
    {
        "id":       "faq_006",
        "category": "process",
        "title":    "HomeFirst Home Loan Application Process",
        "content":  """
Step-by-Step HomeFirst Home Loan Application Process:

Step 1 - Initial Inquiry:
Contact HomeFirst via website, app, or branch. Speak with a loan counselor
(like Ghar Mitra) to understand eligibility and loan parameters.

Step 2 - Document Submission:
Submit all required documents (identity, income, property) to the HomeFirst
representative or upload via the app.

Step 3 - Credit Appraisal:
HomeFirst evaluates your CIBIL score, income, FOIR, and employment stability.
This typically takes 3-5 business days.

Step 4 - Property Valuation:
A HomeFirst empanelled valuer inspects and values the property.
Legal verification of property documents is also conducted.

Step 5 - Loan Sanction:
If approved, HomeFirst issues a sanction letter mentioning loan amount,
tenure, interest rate, and terms and conditions.

Step 6 - Loan Agreement:
Sign the loan agreement and mortgage documents at the HomeFirst branch.
Property is mortgaged as security.

Step 7 - Disbursement:
Loan amount is disbursed directly to the seller/builder.
For under-construction properties, disbursement is in tranches based on
construction progress.

Total timeline: 7-15 business days from complete document submission.
        """.strip(),
    },
    {
        "id":       "faq_007",
        "category": "emi_repayment",
        "title":    "EMI Calculation and Repayment Details",
        "content":  """
EMI and Repayment Information at HomeFirst Finance:

EMI Calculation:
HomeFirst uses the reducing balance method for EMI calculation.
Formula: EMI = P x r x (1+r)^n / ((1+r)^n - 1)
Where P = principal, r = monthly interest rate, n = number of months.

Example EMIs (at 9.5% p.a., 20-year tenure):
- Loan Rs 10 Lakhs  → EMI approx Rs 9,321/month
- Loan Rs 15 Lakhs  → EMI approx Rs 13,982/month
- Loan Rs 20 Lakhs  → EMI approx Rs 18,643/month
- Loan Rs 25 Lakhs  → EMI approx Rs 23,303/month

Repayment Mode:
- EMI via NACH (National Automated Clearing House) auto-debit from salary/bank account.
- Due date: typically 1st or 5th of every month.

Prepayment:
- Partial prepayment: Allowed any time, reduces principal and future EMIs.
- Full prepayment (foreclosure): Allowed after 6 months of first disbursement.
- No prepayment penalty for floating rate loans.

EMI Bounce:
- Bounce charges applicable if EMI auto-debit fails.
- Late payment may affect CIBIL score negatively.
        """.strip(),
    },
    {
        "id":       "faq_008",
        "category": "property_types",
        "title":    "Eligible Property Types at HomeFirst",
        "content":  """
Property Types Eligible for HomeFirst Home Loans:

Residential Properties:
1. Ready-to-move flats and apartments (in approved societies/buildings)
2. Under-construction flats from RERA-registered builders
3. Independent houses / row houses
4. Builder floors
5. Plots with simultaneous construction (plot + construction loan)

Property Requirements:
- Property must be in HomeFirst serviceable locations (primarily Tier 2 and Tier 3 cities)
- Clear and marketable title deed
- No legal disputes or encumbrances
- Approved building plan from local municipal authority
- RERA registration for under-construction projects (mandatory post 2017)

Not Eligible:
- Commercial properties
- Agricultural land
- Properties with disputed titles
- Properties in flood-prone or legally restricted zones
- Properties under demolition order

Valuation:
HomeFirst conducts independent technical and legal valuation.
Final loan amount is based on HomeFirst's assessed value, not seller's price.
        """.strip(),
    },
    {
        "id":       "faq_009",
        "category": "top_up_balance_transfer",
        "title":    "Top-Up Loans and Balance Transfer",
        "content":  """
Top-Up Loans and Balance Transfer at HomeFirst Finance:

Top-Up Loan:
- Existing HomeFirst customers with good repayment track record can apply.
- Available after 12 months of regular EMI payments.
- Amount: Up to Rs 10 Lakhs or based on eligibility.
- Purpose: Home renovation, children's education, medical expenses, etc.
- Interest rate: Slightly higher than home loan rate.
- Tenure: Co-terminus with existing home loan.

Balance Transfer (from another bank):
- Transfer your existing home loan from another bank to HomeFirst.
- Benefit if HomeFirst offers lower interest rate.
- Process: Submit existing loan statement, property documents, income proof.
- HomeFirst evaluates and offers competitive rate.
- Processing fee applicable on transfer amount.
- Ideal when existing rate is more than 0.5% higher than HomeFirst's rate.

Benefits of Balance Transfer to HomeFirst:
- Potential interest savings over remaining tenure.
- Better customer service and flexible repayment options.
- Can club a top-up with balance transfer for additional funds.
        """.strip(),
    },
    {
        "id":       "faq_010",
        "category": "insurance_tax",
        "title":    "Home Loan Insurance and Tax Benefits",
        "content":  """
Insurance and Tax Benefits for HomeFirst Home Loan Customers:

Home Loan Insurance:
- HomeFirst recommends (not mandatory) home loan protection insurance.
- Covers outstanding loan amount in case of borrower's death or disability.
- Premium can be added to loan amount (single premium) or paid annually.
- Protects family from liability in case of unforeseen events.
- Property insurance against fire, flood, earthquake also recommended.

Tax Benefits under Indian Income Tax Act:

Section 80C:
- Principal repayment deduction: Up to Rs 1,50,000 per year.
- Applicable from year of possession of property.
- Also includes stamp duty and registration charges (one-time, year of payment).

Section 24(b):
- Interest paid on home loan: Up to Rs 2,00,000 per year for self-occupied property.
- For under-construction property: Pre-possession interest is deductible in 5 equal
  instalments starting from year of possession.
- No upper limit for let-out property (actual interest deductible).

Section 80EEA (First-time home buyers):
- Additional deduction of Rs 1,50,000 on interest (over and above Section 24).
- Applicable for loans sanctioned between April 2019 and March 2022.
- Property stamp duty value must not exceed Rs 45 Lakhs.

Note: Tax benefits are subject to current Income Tax rules. Consult a tax advisor.
        """.strip(),
    },
]

# Quick index for lookups
FAQ_BY_ID       = {doc["id"]:       doc for doc in FAQ_DOCUMENTS}
FAQ_BY_CATEGORY = {}
for doc in FAQ_DOCUMENTS:
    cat = doc["category"]
    FAQ_BY_CATEGORY.setdefault(cat, []).append(doc)


if __name__ == "__main__":
    print(f"Total FAQ documents: {len(FAQ_DOCUMENTS)}")
    for doc in FAQ_DOCUMENTS:
        print(f"  [{doc['id']}] {doc['title']} ({doc['category']})")
