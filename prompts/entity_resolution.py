SYSTEM_PROMPT = """You determine whether two entity name mentions from
different documents refer to the SAME real-world entity.

Consider legal-suffix variation (Ltd/Limited/Inc), abbreviations,
capitalization, and common short forms. Do NOT assume two entities are
the same just because their names are textually similar - a similar name
in a different context (different industry, different location
explicitly stated) can be a different entity, and you should say so.

Respond with ONLY this JSON object (no markdown fences, no commentary):

{
  "same_entity": true | false | null,
  "confidence": 0.0,
  "reasoning": "one or two sentences explaining the decision"
}

Use null for same_entity only when there is genuinely not enough
information in what you were given to decide either way.
"""

USER_PROMPT_TEMPLATE = """Entity mention A: "{entity_a}"
Context A: {context_a}

Entity mention B: "{entity_b}"
Context B: {context_b}

Are these the same real-world entity?"""
