"""User interaction module for fact-checking sessions.

This module provides interactive features during verification:
- Pre-verification clarification questions
- Async mid-verification questions with timeout
- User message queue for injecting guidance
"""

from .config import InteractionConfig
from .handler import UserInteraction
from .listener import InputListener
from .models import (
    ClarificationQuestion,
    ClarifiedGoal,
    PendingQuestion,
    UserMessage,
)

__all__ = [
    # Models
    "ClarificationQuestion",
    "ClarifiedGoal",
    "PendingQuestion",
    "UserMessage",
    # Config
    "InteractionConfig",
    # Handler
    "UserInteraction",
    # Listener
    "InputListener",
]
