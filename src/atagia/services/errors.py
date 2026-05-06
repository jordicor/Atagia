"""Shared service-level exceptions."""

from __future__ import annotations


class AtagiaServiceError(RuntimeError):
    """Base error for service and library mode operations."""


class ConversationNotFoundError(AtagiaServiceError):
    """Raised when a conversation does not exist for the requested user."""


class ConversationNotActiveError(AtagiaServiceError):
    """Raised when a write path targets a non-active conversation."""


class MessageIdConflictError(AtagiaServiceError):
    """Raised when a caller reuses a message id for incompatible content."""


class SourceSequenceConflictError(AtagiaServiceError):
    """Raised when a caller reuses a source sequence for a different message."""


class ConversationAlreadyClosedError(AtagiaServiceError):
    """Raised when closing an already closed conversation."""


class InvalidConversationTransitionError(AtagiaServiceError):
    """Raised when a requested conversation lifecycle transition is invalid."""


class WorkspaceNotFoundError(AtagiaServiceError):
    """Raised when a workspace does not exist for the requested user."""


class WorkspaceMismatchError(AtagiaServiceError):
    """Raised when a requested workspace conflicts with an existing conversation."""


class UserDeletedError(AtagiaServiceError):
    """Raised when a user-facing path targets an erased user marker."""


class MemoryNotFoundError(AtagiaServiceError):
    """Raised when a memory object does not exist for the requested user."""


class MemoryNotEditableError(AtagiaServiceError):
    """Raised when a memory object cannot be edited in its current state."""


class DeletionConfirmationError(AtagiaServiceError):
    """Raised when a destructive operation is missing its confirmation token."""


class UnknownAssistantModeError(AtagiaServiceError):
    """Raised when a requested assistant mode is not configured."""


class AssistantModeMismatchError(AtagiaServiceError):
    """Raised when a requested mode conflicts with an existing conversation mode."""


class LLMUnavailableError(AtagiaServiceError):
    """Raised when the configured LLM backend is unavailable."""


class RuntimeNotInitializedError(AtagiaServiceError):
    """Raised when the engine runtime has not been set up yet."""
