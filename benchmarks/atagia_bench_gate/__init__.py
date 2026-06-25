"""Atagia-bench-gate: adaptive retrieval gate diagnostic suite.

A small paired-question suite that probes the adaptive retrieval gate: each
topic appears as a world-pure variant (answerable from parametric knowledge,
no stored memory needed) and a personally-anchored variant (depends on the
user's stored memory). Conversation-window questions probe the
``conversation`` classification. The suite scores gate classification accuracy
against shadow-mode traces and end-to-end behavior with the flag ON.
"""
