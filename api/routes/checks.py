"""
Fact-check execution routes.

Handles starting and managing actual fact-check runs from the UI.
"""
import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Add engine to path
engine_path = str(Path(__file__).parent.parent.parent / "engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

router = APIRouter(prefix="/api/checks", tags=["checks"])

# Track running fact-check sessions and their harnesses
running_checks: dict[str, asyncio.Task] = {}
running_harnesses: dict[str, object] = {}  # session_id -> VeritasHarness


class StartCheckRequest(BaseModel):
    """Request to start a fact-check."""
    claim: str
    max_iterations: int = 1
    max_depth: int = 2
    autonomous: bool = True
    enable_mid_questions: bool = False


class ClarifyRequest(BaseModel):
    """Request to generate clarification questions."""
    claim: str
    max_questions: int = 4


class EnrichRequest(BaseModel):
    """Request to enrich a claim using clarifications."""
    claim: str
    questions: list[dict]
    answers: dict[str, str]


async def _haiku_callback(prompt: str, **kwargs) -> str:
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    options = ClaudeAgentOptions(
        model="haiku",
        max_turns=1,
        allowed_tools=[],
    )

    response_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    return response_text


def _build_interaction(max_questions: int):
    from engine.interaction import InteractionConfig, UserInteraction

    config = InteractionConfig.interactive()
    config.max_clarification_questions = max(1, min(6, max_questions))
    return UserInteraction(config=config, llm_callback=_haiku_callback)


class CheckStatusResponse(BaseModel):
    """Response with fact-check status."""
    session_id: str
    status: str  # 'starting', 'running', 'completed', 'error'
    message: str | None = None


async def run_check_background(session_id: str, claim: str, max_iterations: int, max_depth: int = 5, autonomous: bool = True, enable_mid_questions: bool = False):
    """
    Run fact-check in the background.

    This imports and runs the actual VeritasHarness.
    """
    print(f"\n{'='*60}")
    print("STARTING BACKGROUND FACT-CHECK")
    print(f"   Session ID: {session_id}")
    print(f"   Claim: {claim}")
    print(f"   Iterations: {max_iterations}")
    print(f"   Autonomous Mode: {autonomous}")
    print(f"   Mid-Check Questions: {enable_mid_questions}")
    print(f"{'='*60}\n")

    try:
        print("Importing VeritasHarness...")
        from engine.agents.director import VeritasHarness
        from engine.interaction import InteractionConfig
        print("Import successful")

        # Update session status to running
        from api.db import get_db
        db = await get_db()
        await db.update_session_status(session_id, "running")

        # Create interaction config
        # Pre-check clarification is handled by the UI via /clarify + /enrich endpoints,
        # so it's always disabled here. Mid-check questions use a 30s timeout.
        print("Creating interaction config...")
        interaction_config = InteractionConfig(
            enable_clarification=False,
            enable_async_questions=enable_mid_questions,
            autonomous_mode=autonomous and not enable_mid_questions,
            question_timeout=30,
            max_questions_per_session=5,
        )
        print(f"Config created (mid_questions: {enable_mid_questions}, autonomous: {autonomous})")

        print("Starting fact-check harness...")
        async with VeritasHarness(
            db_path="veritas.db",
            interaction_config=interaction_config,
            max_depth=max_depth,
        ) as harness:
            print("Harness initialized")

            # Save harness reference so pause endpoint can signal it
            running_harnesses[session_id] = harness

            # Replace CLI interaction with UI interaction if mid-questions enabled
            if enable_mid_questions:
                print("Setting up UI interaction handler...")
                # UI interaction would be configured here
                print("UI interaction configured")

            print(f"Starting fact-check for: {claim}")
            print(f"   Using existing session ID: {session_id}")

            # Pass existing session_id so we don't create a duplicate session
            result = await harness.verify(
                claim=claim,
                max_iterations=max_iterations,
                existing_session_id=session_id
            )

            # Check if fact-check was paused (don't overwrite status)
            session = await db.get_session(session_id)
            if session and session.status == "paused":
                print("Fact-check paused. State saved.")
                return result

            print("Fact-check completed successfully!")
            evidence_count = len(result.key_evidence) if hasattr(result, 'key_evidence') else 'N/A'
            print(f"   Evidence: {evidence_count}")

            # Update session status to completed
            from datetime import datetime
            await db.update_session_status(session_id, "completed", ended_at=datetime.now())

            return result

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in background fact-check for {session_id}")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()

        # Update session status to error
        try:
            from api.db import get_db
            db = await get_db()
            await db.update_session_status(session_id, "error")
        except Exception:
            pass

        raise
    finally:
        # Clean up
        print(f"Cleaning up session {session_id}")
        if session_id in running_checks:
            del running_checks[session_id]
            print("   Removed from running_checks")
        if session_id in running_harnesses:
            del running_harnesses[session_id]
            print("   Removed from running_harnesses")


@router.post("/start", response_model=CheckStatusResponse)
async def start_check(
    request: StartCheckRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new fact-check session.

    This creates the session AND starts the actual fact-check process.
    Events will be emitted via WebSocket as the fact-check progresses.
    """
    from api.db import get_db

    # Create session in database
    db = await get_db()
    session = await db.create_session(request.claim, request.max_iterations)
    session_id = session.id

    # Start fact-check in background
    task = asyncio.create_task(
        run_check_background(
            session_id,
            request.claim,
            request.max_iterations,
            request.max_depth,
            request.autonomous,
            request.enable_mid_questions,
        )
    )
    running_checks[session_id] = task

    return CheckStatusResponse(
        session_id=session_id,
        status="running",
        message=f"Fact-check started. Connect to WebSocket /ws/{session_id} for live updates."
    )


@router.post("/clarify")
async def clarify_claim(request: ClarifyRequest):
    """Generate clarification questions for a claim."""
    interaction = _build_interaction(request.max_questions)

    questions = await interaction._generate_clarification_questions(request.claim)
    questions = questions[: interaction.config.max_clarification_questions]

    return {"questions": [q.model_dump() for q in questions]}


@router.post("/enrich")
async def enrich_claim(request: EnrichRequest):
    """Enrich a claim with user clarifications."""
    from engine.interaction import UserInteraction
    from engine.interaction.models import ClarificationQuestion

    interaction = UserInteraction(config=None, llm_callback=_haiku_callback)

    questions = [ClarificationQuestion(**q) for q in request.questions]
    answers: dict[int, str] = {}
    for key, value in request.answers.items():
        try:
            answers[int(key)] = value
        except Exception:
            continue

    enriched = await interaction._enrich_claim(request.claim, questions, answers)
    return {"enriched_claim": enriched}


@router.get("/{session_id}/status", response_model=CheckStatusResponse)
async def get_check_status(session_id: str):
    """Get the status of a fact-check session."""
    from api.db import get_db

    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if fact-check is running
    is_running = session_id in running_checks

    return CheckStatusResponse(
        session_id=session_id,
        status="running" if is_running else session.status,
        message=f"Fact-check is {'currently running' if is_running else session.status}"
    )


@router.post("/{session_id}/stop")
async def stop_check(session_id: str):
    """Stop a running fact-check session."""
    if session_id not in running_checks:
        raise HTTPException(status_code=404, detail="No running fact-check found for this session")

    task = running_checks[session_id]
    task.cancel()
    del running_checks[session_id]

    return {"status": "stopped", "session_id": session_id}


@router.post("/{session_id}/pause")
async def pause_check(session_id: str):
    """Pause a running fact-check session. State is saved for later resume."""
    harness = running_harnesses.get(session_id)
    if not harness or not hasattr(harness, "director"):
        raise HTTPException(status_code=404, detail="No running fact-check found for this session")

    harness.director.pause_research()

    # Immediately emit a WebSocket event so the UI gets feedback
    try:
        from engine.events import emit_agent_event
        await emit_agent_event(
            session_id=session_id,
            event_type="system",
            agent="director",
            data={"message": "Pause requested - finishing current operation..."},
        )
    except Exception:
        pass

    return {"status": "pausing", "session_id": session_id}


async def run_check_resume_background(session_id: str):
    """Resume a paused or crashed fact-check session in the background."""
    print(f"\n{'='*60}")
    print("RESUMING BACKGROUND FACT-CHECK")
    print(f"   Session ID: {session_id}")
    print(f"{'='*60}\n")

    try:
        from api.db import get_db
        from engine.agents.director import VeritasHarness
        from engine.interaction import InteractionConfig

        db = await get_db()
        session = await db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Don't update status here - the Director's resume flow handles it
        # after loading and validating the session status

        interaction_config = InteractionConfig(
            enable_clarification=False,
            autonomous_mode=True,
        )

        async with VeritasHarness(
            db_path="veritas.db",
            interaction_config=interaction_config,
        ) as harness:
            # Save harness reference
            running_harnesses[session_id] = harness

            print(f"Resuming fact-check: {session.claim}")

            result = await harness.verify(
                claim=session.claim,
                max_iterations=session.max_iterations if hasattr(session, "max_iterations") else 5,
                existing_session_id=session_id,
                resume=True,
            )

            # Check if paused again
            refreshed = await db.get_session(session_id)
            if refreshed and refreshed.status == "paused":
                print("Fact-check paused again. State saved.")
                return result

            print("Resumed fact-check completed successfully!")

            from datetime import datetime
            await db.update_session_status(session_id, "completed", ended_at=datetime.now())
            return result

    except Exception as e:
        print(f"ERROR resuming fact-check for {session_id}: {e}")
        import traceback
        traceback.print_exc()

        try:
            from api.db import get_db
            db = await get_db()
            await db.update_session_status(session_id, "error")
        except Exception:
            pass
        raise

    finally:
        print(f"Cleaning up resumed session {session_id}")
        if session_id in running_checks:
            del running_checks[session_id]
        if session_id in running_harnesses:
            del running_harnesses[session_id]


@router.post("/{session_id}/resume", response_model=CheckStatusResponse)
async def resume_check(session_id: str):
    """Resume a paused or crashed fact-check session."""
    from api.db import get_db

    db = await get_db()
    session = await db.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("paused", "crashed"):
        raise HTTPException(
            status_code=400,
            detail=f"Session is '{session.status}', not 'paused' or 'crashed'"
        )

    # Start fact-check resume in background
    task = asyncio.create_task(run_check_resume_background(session_id))
    running_checks[session_id] = task

    return CheckStatusResponse(
        session_id=session_id,
        status="running",
        message=f"Fact-check resuming. Connect to WebSocket /ws/{session_id} for live updates.",
    )


class AnswerQuestionRequest(BaseModel):
    """Request to answer a mid-check question."""
    question_id: str
    response: str


@router.post("/{session_id}/answer")
async def answer_question(session_id: str, request: AnswerQuestionRequest):
    """Answer a mid-check question."""
    from api.question_manager import get_question_manager

    question_manager = get_question_manager()
    success = question_manager.answer_question(request.question_id, request.response)

    if not success:
        raise HTTPException(status_code=404, detail="Question not found or already answered")

    return {"status": "answered", "question_id": request.question_id}
