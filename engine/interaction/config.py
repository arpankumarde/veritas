"""Configuration for user interaction features."""

from dataclasses import dataclass


@dataclass
class InteractionConfig:
    """Configuration for user interaction during fact-checking."""

    # Pre-verification clarification
    enable_clarification: bool = True
    max_clarification_questions: int = 4
    clarification_timeout: int = 120  # seconds to wait for each answer

    # Mid-verification async questions
    enable_async_questions: bool = True
    question_timeout: int = 60  # seconds to wait for response
    max_questions_per_session: int = 5

    # User message queue
    enable_message_queue: bool = True

    # Fully autonomous mode (no interaction at all)
    autonomous_mode: bool = False

    @classmethod
    def from_cli_args(
        cls,
        no_clarify: bool = False,
        autonomous: bool = False,
        timeout: int = 60,
    ) -> "InteractionConfig":
        """Create config from CLI arguments.

        Args:
            no_clarify: Skip pre-verification clarification questions
            autonomous: Run fully autonomous (no interaction)
            timeout: Timeout for mid-verification questions in seconds

        Returns:
            InteractionConfig with appropriate settings
        """
        if autonomous:
            return cls(
                enable_clarification=False,
                enable_async_questions=False,
                enable_message_queue=False,
                autonomous_mode=True,
            )

        return cls(
            enable_clarification=not no_clarify,
            question_timeout=timeout,
        )

    @classmethod
    def autonomous(cls) -> "InteractionConfig":
        """Create a fully autonomous config (no user interaction)."""
        return cls.from_cli_args(autonomous=True)

    @classmethod
    def interactive(cls) -> "InteractionConfig":
        """Create a fully interactive config (all features enabled)."""
        return cls()
