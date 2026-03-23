"""Decision logging for explainable agent reasoning."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..storage.database import VeritasDatabase


class DecisionType(Enum):
    SYNTHESIS_TRIGGER = "synthesis_trigger"
    TOPIC_SELECTION = "topic_selection"
    DIRECTIVE_CREATE = "directive_create"
    STOP_SEARCHING = "stop_searching"
    QUERY_EXPAND = "query_expand"
    DEDUP_SKIP = "dedup_skip"
    MULTI_QUERY_GEN = "multi_query_gen"
    CONTEXTUAL_EXPAND = "contextual_expand"
    SUFFICIENCY_CHECK = "sufficiency_check"
    QUERY_MERGE = "query_merge"
    ERROR_RECOVERY = "error_recovery"
    PRIORITY_CHANGE = "priority_change"
    VERDICT_DETERMINATION = "verdict_determination"


@dataclass
class AgentDecisionRecord:
    session_id: str
    agent_role: str
    decision_type: DecisionType
    decision_outcome: str
    reasoning: str | None = None
    inputs: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    iteration: int | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_role": self.agent_role,
            "decision_type": self.decision_type.value,
            "decision_outcome": self.decision_outcome,
            "reasoning": self.reasoning[:500] if self.reasoning else None,
            "inputs_json": json.dumps(self.inputs) if self.inputs else None,
            "metrics_json": json.dumps(self.metrics) if self.metrics else None,
            "iteration": self.iteration,
            "created_at": self.created_at.isoformat(),
        }


class DecisionLogger:
    def __init__(self, db: "VeritasDatabase", batch_size: int = 10, flush_interval_seconds: float = 1.0):
        self.db = db
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: list[AgentDecisionRecord] = []
        self._queue_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush()

    async def log(self, record: AgentDecisionRecord) -> None:
        async with self._queue_lock:
            self._queue.append(record)
            if len(self._queue) >= self.batch_size:
                asyncio.create_task(self._flush())

    async def _flush_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Decision logger flush failed", exc_info=True)

    async def _flush(self) -> None:
        async with self._queue_lock:
            if not self._queue:
                return
            records = self._queue.copy()
            self._queue.clear()
        if not records:
            return
        try:
            await self.db.save_agent_decisions_batch([r.to_dict() for r in records])
        except Exception:
            logger.debug("Decision flush error", exc_info=True)
            async with self._queue_lock:
                self._queue.extend(records)

    async def log_decision(
        self, session_id: str, agent_role: str, decision_type: DecisionType,
        decision_outcome: str, reasoning: str | None = None,
        inputs: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None,
        iteration: int | None = None,
    ) -> None:
        record = AgentDecisionRecord(
            session_id=session_id, agent_role=agent_role,
            decision_type=decision_type, decision_outcome=decision_outcome,
            reasoning=reasoning, inputs=inputs, metrics=metrics, iteration=iteration,
        )
        await self.log(record)

    def get_queue_size(self) -> int:
        return len(self._queue)


_decision_logger: DecisionLogger | None = None


def get_decision_logger(db: Optional["VeritasDatabase"] = None) -> DecisionLogger | None:
    global _decision_logger
    if _decision_logger is None and db is not None:
        _decision_logger = DecisionLogger(db)
    return _decision_logger


async def init_decision_logger(db: "VeritasDatabase") -> DecisionLogger:
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = DecisionLogger(db)
    await _decision_logger.start()
    return _decision_logger
