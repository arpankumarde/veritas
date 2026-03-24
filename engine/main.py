"""Main CLI entry point for Veritas fact-checking engine."""

import asyncio
import signal
import subprocess
import sys
import webbrowser

import typer
from rich.console import Console
from rich.panel import Panel

from .agents.director import VeritasHarness
from .interaction import InputListener, InteractionConfig
from .logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

console = Console()

app = typer.Typer(
    name="veritas",
    help="AI-powered fact-checking system with hierarchical multi-agent verification.",
    add_completion=False,
)

_harness: VeritasHarness | None = None


def _handle_interrupt(signum, frame):
    console.print("\n[yellow]Interrupt received - stopping fact check...[/yellow]")
    harness = _harness
    if harness is not None:
        harness.stop()


@app.command()
def main(
    claim: str = typer.Argument(..., help="The claim or statement to fact-check"),
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of verification iterations (1 is usually enough)", min=1, max=10),
    db_path: str = typer.Option("veritas.db", "--db", "-d", help="Path to SQLite database"),
    no_clarify: bool = typer.Option(False, "--no-clarify", help="Skip pre-check clarification questions"),
    autonomous: bool = typer.Option(False, "--autonomous", "-a", help="Run fully autonomous"),
    timeout: int = typer.Option(60, "--timeout", help="Timeout for mid-check questions", min=10, max=300),
    depth: int = typer.Option(5, "--depth", help="Maximum verification depth", min=1, max=10),
):
    """Fact-check a claim using multi-agent AI verification.

    Veritas decomposes claims into sub-claims, gathers evidence for and against,
    and produces a verdict: True / Mostly True / Mixed / Mostly False / False / Unverifiable.

    Output is saved to: output/{claim-slug}_{session-id}/
      - verdict.md        Narrative verdict report
      - evidence.json     Structured evidence data

    Examples:
        veritas "The Great Wall of China is visible from space"
        veritas "COVID vaccines cause autism" --iterations 10
        veritas "Earth is 4.5 billion years old" -n 3 --no-clarify
    """
    global _harness

    console.print()
    console.print(Panel(
        "[bold]Veritas Fact Checker[/bold]\n"
        "Hierarchical multi-agent claim verification system",
        border_style="blue",
    ))

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    interaction_config = InteractionConfig.from_cli_args(
        no_clarify=no_clarify, autonomous=autonomous, timeout=timeout,
    )

    if autonomous:
        console.print("[dim]Running in autonomous mode[/dim]")
    elif no_clarify:
        console.print("[dim]Skipping clarification. Type + Enter during verification to inject guidance.[/dim]")
    else:
        console.print("[dim]Interactive mode. Type + Enter during verification to inject guidance.[/dim]")

    async def run():
        global _harness
        async with VeritasHarness(db_path, interaction_config=interaction_config, max_depth=depth) as harness:
            _harness = harness

            listener: InputListener | None = None
            if not interaction_config.autonomous_mode:
                listener = InputListener(
                    harness.director.interaction,
                    console=console,
                    on_interact_start=harness.director.pause_progress,
                    on_interact_end=harness.director.resume_progress,
                )
                harness.director.set_input_listener(listener)

            try:
                report = await harness.check(claim, iterations)
                return report
            except asyncio.CancelledError:
                console.print("[yellow]Fact check cancelled[/yellow]")
                return None
            finally:
                if listener:
                    listener.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)


@app.command()
def ui(
    session_id: str | None = typer.Argument(None, help="Optional session ID to open"),
    port: int = typer.Option(9090, "--port", "-p", help="API server port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
    restart: bool = typer.Option(True, "--restart/--no-restart", help="Restart servers if ports in use"),
):
    """Launch the Veritas web UI for managing fact checks."""
    import os
    import socket
    import time
    from pathlib import Path

    console.print()
    console.print(Panel(
        "[bold]Veritas UI[/bold]\nLaunching web interface...",
        border_style="blue",
    ))

    def check_port(port_num):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port_num)) == 0
        sock.close()
        return result

    # Start API server
    if not check_port(port):
        console.print(f"[cyan]Starting API server on port {port}...[/cyan]")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "api.server"],
                start_new_session=True,
                env=os.environ.copy(),
            )
            for i in range(10):
                if check_port(port):
                    console.print("[green]API server started[/green]")
                    break
                time.sleep(0.5)
            else:
                console.print("[red]Failed to start API server[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    else:
        console.print(f"[yellow]API already running on port {port}[/yellow]")

    # Start Next.js frontend
    ui_port = 3004
    if not check_port(ui_port):
        console.print(f"[cyan]Starting frontend on port {ui_port}...[/cyan]")
        try:
            subprocess.Popen(
                ["pnpm", "dev"],
                start_new_session=True,
                env=os.environ.copy(),
            )
            import urllib.request
            for i in range(20):
                try:
                    urllib.request.urlopen(f"http://localhost:{ui_port}", timeout=1)
                    console.print("[green]Frontend started[/green]")
                    break
                except Exception:
                    time.sleep(1)
            else:
                console.print("[red]Failed to start frontend[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    url = f"http://localhost:{ui_port}"
    if session_id:
        url += f"/check/{session_id}"

    console.print(f"\n[bold green]All servers ready[/bold green]")
    console.print(f"[dim]Frontend: http://localhost:{ui_port}[/dim]")
    console.print(f"[dim]API: http://localhost:{port}[/dim]")

    if not no_browser:
        console.print(f"\n[cyan]Opening {url}...[/cyan]")
        webbrowser.open(url)

    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        sys.exit(0)


def cli():
    app()


if __name__ == "__main__":
    cli()
