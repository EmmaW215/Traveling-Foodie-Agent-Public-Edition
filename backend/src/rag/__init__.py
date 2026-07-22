"""Tier 0 — the RAG copilot.

Answers questions about the Calgary dataset and *only* the Calgary dataset:
retrieve relevant venue chunks, answer grounded in them, and refuse (or correct)
anything that would name a venue not in the knowledge base. This is the public
edition of the hackathon's "copilot answers only from its knowledge base" rule,
with the anti-hallucination check enforced in code, not just in the prompt.
"""
