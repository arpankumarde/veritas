"""Cost tracking for API usage during fact-check sessions."""

from dataclasses import dataclass, field
from datetime import datetime


class ModelPricing:
    """Anthropic API pricing per million tokens."""

    OPUS_INPUT = 5.00
    OPUS_OUTPUT = 25.00
    SONNET_INPUT = 3.00
    SONNET_OUTPUT = 15.00
    HAIKU_INPUT = 1.00
    HAIKU_OUTPUT = 5.00
    WEB_SEARCH_COST = 0.01
    WEB_FETCH_COST = 0.00

    @classmethod
    def get_input_price(cls, model: str) -> float:
        model_lower = model.lower()
        if "opus" in model_lower:
            return cls.OPUS_INPUT
        elif "haiku" in model_lower:
            return cls.HAIKU_INPUT
        return cls.SONNET_INPUT

    @classmethod
    def get_output_price(cls, model: str) -> float:
        model_lower = model.lower()
        if "opus" in model_lower:
            return cls.OPUS_OUTPUT
        elif "haiku" in model_lower:
            return cls.HAIKU_OUTPUT
        return cls.SONNET_OUTPUT


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    calls: int = 0

    @property
    def total_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.total_output_tokens


@dataclass
class CostSummary:
    sonnet_usage: ModelUsage = field(default_factory=ModelUsage)
    opus_usage: ModelUsage = field(default_factory=ModelUsage)
    haiku_usage: ModelUsage = field(default_factory=ModelUsage)
    web_searches: int = 0
    web_fetches: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.sonnet_usage.input_tokens + self.opus_usage.input_tokens + self.haiku_usage.input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.sonnet_usage.total_output_tokens + self.opus_usage.total_output_tokens + self.haiku_usage.total_output_tokens

    @property
    def total_thinking_tokens(self) -> int:
        return self.sonnet_usage.thinking_tokens + self.opus_usage.thinking_tokens + self.haiku_usage.thinking_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_calls(self) -> int:
        return self.sonnet_usage.calls + self.opus_usage.calls + self.haiku_usage.calls

    @property
    def sonnet_cost(self) -> float:
        return (self.sonnet_usage.input_tokens / 1_000_000) * ModelPricing.SONNET_INPUT + \
               (self.sonnet_usage.total_output_tokens / 1_000_000) * ModelPricing.SONNET_OUTPUT

    @property
    def opus_cost(self) -> float:
        return (self.opus_usage.input_tokens / 1_000_000) * ModelPricing.OPUS_INPUT + \
               (self.opus_usage.total_output_tokens / 1_000_000) * ModelPricing.OPUS_OUTPUT

    @property
    def haiku_cost(self) -> float:
        return (self.haiku_usage.input_tokens / 1_000_000) * ModelPricing.HAIKU_INPUT + \
               (self.haiku_usage.total_output_tokens / 1_000_000) * ModelPricing.HAIKU_OUTPUT

    @property
    def search_cost(self) -> float:
        return self.web_searches * ModelPricing.WEB_SEARCH_COST

    @property
    def total_cost(self) -> float:
        return self.sonnet_cost + self.opus_cost + self.haiku_cost + self.search_cost

    def to_dict(self) -> dict:
        return {
            "models": {
                "sonnet": {"input_tokens": self.sonnet_usage.input_tokens, "output_tokens": self.sonnet_usage.output_tokens, "thinking_tokens": self.sonnet_usage.thinking_tokens, "calls": self.sonnet_usage.calls, "cost_usd": round(self.sonnet_cost, 4)},
                "opus": {"input_tokens": self.opus_usage.input_tokens, "output_tokens": self.opus_usage.output_tokens, "thinking_tokens": self.opus_usage.thinking_tokens, "calls": self.opus_usage.calls, "cost_usd": round(self.opus_cost, 4)},
                "haiku": {"input_tokens": self.haiku_usage.input_tokens, "output_tokens": self.haiku_usage.output_tokens, "thinking_tokens": self.haiku_usage.thinking_tokens, "calls": self.haiku_usage.calls, "cost_usd": round(self.haiku_cost, 4)},
            },
            "web": {"searches": self.web_searches, "fetches": self.web_fetches, "cost_usd": round(self.search_cost, 4)},
            "totals": {"input_tokens": self.total_input_tokens, "output_tokens": self.total_output_tokens, "thinking_tokens": self.total_thinking_tokens, "total_tokens": self.total_tokens, "api_calls": self.total_calls, "total_cost_usd": round(self.total_cost, 4)},
        }


class CostTracker:
    CHARS_PER_TOKEN = 4

    def __init__(self):
        self._summary = CostSummary()
        self._summary.started_at = datetime.now()

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // cls.CHARS_PER_TOKEN)

    def track_call(self, model: str, input_text: str, output_text: str, system_prompt: str = "", thinking_text: str = "") -> None:
        input_tokens = self.estimate_tokens(input_text) + self.estimate_tokens(system_prompt)
        output_tokens = self.estimate_tokens(output_text)
        thinking_tokens = self.estimate_tokens(thinking_text)

        model_lower = model.lower()
        if "opus" in model_lower:
            usage = self._summary.opus_usage
        elif "haiku" in model_lower:
            usage = self._summary.haiku_usage
        else:
            usage = self._summary.sonnet_usage

        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.thinking_tokens += thinking_tokens
        usage.calls += 1

    def track_web_search(self, count: int = 1) -> None:
        self._summary.web_searches += count

    def track_web_fetch(self, count: int = 1) -> None:
        self._summary.web_fetches += count

    def get_summary(self) -> CostSummary:
        self._summary.ended_at = datetime.now()
        return self._summary

    def reset(self) -> None:
        self._summary = CostSummary()
        self._summary.started_at = datetime.now()


_global_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker() -> None:
    global _global_tracker
    _global_tracker = CostTracker()
