"""The agents.

Each agent is a thin function around one LLM call with a strict JSON contract.
They never touch a provider directly (that is llm_client) and never invent
facts (the catalogue tool is the only source of venues). Every agent output
that names a venue is checked against the catalogue before it is trusted.
"""
