"""Background input listener for user messages during fact-checking."""

import asyncio
import sys
import threading
from collections.abc import Callable

from rich.console import Console
from rich.prompt import Prompt

from ..logging_config import get_logger
from .handler import UserInteraction

logger = get_logger(__name__)


class InputListener:
    """Background listener for user input during fact-checking.

    Type a message and press Enter to inject guidance. The spinner will
    pause, show your input, and let you confirm or add more detail.
    """

    def __init__(
        self,
        interaction: UserInteraction,
        console: Console | None = None,
        on_interact_start: Callable[[], None] | None = None,
        on_interact_end: Callable[[], None] | None = None,
    ):
        """Initialize the input listener.

        Args:
            interaction: UserInteraction handler to route input to
            console: Rich console for output
            on_interact_start: Callback when entering interact mode (pause spinner)
            on_interact_end: Callback when exiting interact mode (resume spinner)
        """
        self.interaction = interaction
        self.console = console or Console()
        self.on_interact_start = on_interact_start
        self.on_interact_end = on_interact_end

        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._in_interact_mode = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Start the background input listener."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._loop = asyncio.get_event_loop()

        # Start the listener in a daemon thread
        self._thread = threading.Thread(target=self._listen_thread, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background input listener."""
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _listen_thread(self) -> None:
        """Background thread that reads stdin lines."""
        while self._running and not self._stop_event.is_set():
            try:
                # Use select to check if input is available (non-blocking check)
                if sys.platform != 'win32':
                    import select
                    readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if not readable:
                        continue
                else:
                    import msvcrt
                    import time
                    # On Windows, poll for input
                    if not msvcrt.kbhit():
                        time.sleep(0.1)
                        continue

                # Read the line (this blocks until Enter)
                if self._stop_event.is_set():
                    break

                line = sys.stdin.readline()
                if not line:
                    continue

                line = line.strip()
                if not line:
                    continue

                # Schedule the interact mode on the main event loop
                if self._loop and self._running:
                    asyncio.run_coroutine_threadsafe(
                        self._handle_input(line),
                        self._loop
                    )

            except Exception:
                logger.debug("Input listener read error", exc_info=True)
                if not self._stop_event.is_set():
                    continue
                break

    async def _handle_input(self, initial_text: str) -> None:
        """Handle user input - pause spinner and process."""
        if self._in_interact_mode:
            return

        self._in_interact_mode = True

        # Pause the spinner
        if self.on_interact_start:
            self.on_interact_start()

        # Show what was typed
        self.console.print()
        self.console.print("[bold cyan]--- User Guidance ---[/bold cyan]")
        self.console.print(f"[bold]You typed:[/bold] {initial_text}")

        try:
            # Ask for confirmation or more input
            loop = asyncio.get_event_loop()
            more = await loop.run_in_executor(
                None,
                lambda: Prompt.ask(
                    "[dim]Press Enter to send, or type more to replace[/dim]",
                    default=""
                )
            )

            # Use the new text if provided, otherwise use original
            final_text = more.strip() if more.strip() else initial_text

            # Route the input
            if self.interaction.has_pending_question():
                self.console.print("[green]Answering pending question...[/green]")
                self.interaction.respond(final_text)
            else:
                self.interaction.inject_message(final_text)

        except Exception as e:
            logger.warning("Input handling failed: %s", e, exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")

        finally:
            self.console.print("[bold cyan]---------------------[/bold cyan]")
            self.console.print()

            # Resume the spinner
            if self.on_interact_end:
                self.on_interact_end()

            self._in_interact_mode = False
