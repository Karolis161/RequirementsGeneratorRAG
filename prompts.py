def zero_shot(query, context):
    return f"""
You are a financial security architect generating highly specific, implementation-ready requirements for payment authorization workflows in banking systems.

**Query:** {query}

**Retrieved Text:**
{context}

**Structured Output:**
- **Functional Requirements**
- **Non-Functional Requirements**
"""


def few_shot(query, context):
    example = """
**Example Query:** What are the security requirements for a digital banking platform?

**Example Retrieved Text:**
- Multi-factor authentication should be implemented.
- All traffic should be encrypted with TLS 1.3.
- System must be available 99.99% of the time.
- Should follow ISO 27001 and PCI-DSS standards.
- Passwords should follow NIST guidelines.

**Example Output:**
- **Functional Requirements**
  1. Implement multi-factor authentication (MFA).  
     → *"Multi-factor authentication should be implemented."*
  2. Encrypt all user sessions with TLS 1.3.  
     → *"All traffic should be encrypted with TLS 1.3."*
  3. Ensure password policies follow NIST SP 800-63B.  
     → *"Passwords should follow NIST guidelines."*

- **Non-Functional Requirements**
  1. System uptime must be ≥ 99.99%.  
     → *"System must be available 99.99% of the time."*
  2. Ensure ISO 27001 and PCI-DSS compliance.  
     → *"Should follow ISO 27001 and PCI-DSS standards."*
"""

    return f"""
You are a financial security architect generating highly specific, implementation-ready requirements for payment authorization workflows in banking systems.

Use the format and reasoning from the example. Do not infer or generalize beyond the retrieved context.

{example}

**Query:** {query}

**Retrieved Text:**
{context}

**Structured Output:**
- **Functional Requirements**
- **Non-Functional Requirements**
"""


def chain_of_thought(query, context):
    return f"""
You are a financial systems architect generating specific and actionable requirements for secure payment authorization workflows in banking systems.

Think step-by-step to avoid hallucination. Base everything strictly on the retrieved documents.

**Query:** {query}

**Reasoning Steps:**
1. Identify risks, security standards, or regulatory mandates from the context.
2. Detect implementation patterns (e.g., biometric MFA, transaction limits, PSD2 clauses).
3. Filter out vague ideas—only retain technical, verifiable measures.
4. Formulate clear functional and non-functional requirements using the available evidence.

**Retrieved Text:**
{context}

**Structured Output:**
- **Functional Requirements**
- **Non-Functional Requirements**
"""


def cot_few_shot(query, context):
    example_reasoning = """
**Example Query:** List at least 5 highly specific functional and non-functional requirements for secure payment authorization workflows in banking systems.

**Reasoning Steps:**
1. Use only clearly supported information from the retrieved text. You may rephrase and combine multiple fragments, but do not invent or infer beyond what is written.
2. Include a justification quote (paraphrased or exact) to show which chunk supports the requirement.
3. Prefer requirements that involve:
   - Security protocols (e.g. OAuth2, TLS 1.3, SAML)
   - Authentication methods (e.g. MFA, biometric, device-bound token)
   - Standards (e.g. PCI DSS, PSD2, NIST SP 800-63)
   - Timings, thresholds, or control logic (e.g. “within 5 seconds”, “after 3 failed attempts”)
   - Logging, encryption, data retention
4. Avoid vague or general statements like “ensure privacy” unless backed by specific controls.
5. Do not repeat ideas across functional and non-functional sections.
6. **Ensure requirements are unambiguous and contain full implementation context**:
   - When does the rule apply?
   - What are the exact thresholds, actors, or steps?
   - What happens when the condition fails?
7. Requirements should be concise, technical, and ready for implementation review.
8. Use the format: **"[Subject] must [do something] using [tech/method] to achieve [goal]."**
9. If a quote is too long or unclear, paraphrase it.
10. **Maintain internal consistency. Don't contradict earlier constraints or reuse logic in conflicting ways.**

---

**Example Retrieved Text:**
- "High-value transactions require step-up authorization using biometric re-authentication and device binding."
- "Authorization requests must be responded to within 5 seconds to meet real-time payment SLAs."
- "CSPs must implement audit logs for all access attempts and ensure logs are retained for a minimum of 12 months."
- "Authentication credentials must never be transmitted without encryption."

**Example Output:**

- **Functional Requirements**
  1. The system must enforce biometric re-authentication for high-value transactions to mitigate fraud.  
     → *"High-value transactions require step-up authorization using biometric re-authentication..."*
  2. Authorization must be completed within 5 seconds to meet SLA guarantees.  
     → *"Authorization requests must be responded to within 5 seconds..."*

- **Non-Functional Requirements**
  1. All access attempts must be logged and retained for at least 12 months using centralized logging tools.  
     → *"CSPs must implement audit logs... retained for a minimum of 12 months."*
  2. Authentication credentials must only be transmitted over encrypted channels to maintain data confidentiality.  
     → *"Authentication credentials must never be transmitted without encryption."*
"""
    return f"""
You are a financial systems architect tasked with generating implementation-ready security requirements for payment authorization workflows.

Focus on producing **precise**, **measurable**, and **technically grounded** requirements based only on the text below. Do not guess or generalize.

**Avoid vague descriptions, unclear triggers, or missing follow-up logic. Requirements should be clear even to someone unfamiliar with the system's context.**

{example_reasoning}

**Query:** {query}

**Retrieved Text:**
{context}

**Output:**
- **Functional Requirements**
- **Non-Functional Requirements**
"""


PROMPT_TEMPLATES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "cot_few_shot": cot_few_shot,
}
