"""Shared service-level exceptions."""

from __future__ import annotations


class AtagiaServiceError(RuntimeError):
    """Base error for service and library mode operations."""


class ConversationNotFoundError(AtagiaServiceError):
    """Raised when a conversation does not exist for the requested user."""


class WorkspaceNotFoundError(AtagiaServiceError):
    """Raised when a workspace does not exist for the requested user."""


class UnknownAssistantModeError(AtagiaServiceError):
    """Raised when a requested assistant mode is not configured."""


class AssistantModeMismatchError(AtagiaServiceError):
    """Raised when a requested mode conflicts with an existing conversation mode."""


class LLMUnavailableError(AtagiaServiceError):
    """Raised when the configured LLM backend is unavailable."""


class RuntimeNotInitializedError(AtagiaServiceError):
    """Raised when the engine runtime has not been set up yet."""
