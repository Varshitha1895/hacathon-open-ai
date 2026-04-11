"""Type-safe contracts and small reward helpers for the moderation environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class Task:
    task_id: str
    category: str
    text: str
    label: int
    context: Optional[str] = None


@dataclass(slots=True)
class AgentAction:
    task_id: str
    decision: str
    confidence: float = 0.5
    reason: str = ""

    def normalized_decision(self) -> str:
        decision = str(self.decision).strip().upper()
        return decision if decision in {"BLOCK", "ALLOW"} else "ALLOW"

    @classmethod
    def from_dict(cls, payload: dict[str, Any], default_task_id: str = "") -> "AgentAction":
        return cls(
            task_id=str(payload.get("task_id", default_task_id)),
            decision=str(payload.get("decision", "ALLOW")),
            confidence=_clamp(payload.get("confidence", 0.5)),
            reason=str(payload.get("reason", "")),
        )


@dataclass(slots=True)
class Observation:
    task_id: str
    category: str
    text: str
    context: Optional[str] = None
    step: int = 0
    done: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "text": self.text,
            "context": self.context,
            "step": self.step,
            "done": self.done,
            "message": self.message,
        }


@dataclass(slots=True)
class RewardResult:
    task_id: str
    score: float
    breakdown: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": _clamp(self.score),
            "breakdown": self.breakdown,
            "feedback": self.feedback,
        }


@dataclass(slots=True)
class StepResult:
    observation: Observation
    reward: RewardResult
    done: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward.to_dict(),
            "done": self.done,
            "info": self.info,
        }


TASK_BANK: list[Task] = [
    Task("abuse_001", "abuse", "Nee amma ni dengutanu", 1, "Vulgar comment"),
    Task("harassment_001", "harassment", "Aa aadhi vaalu chusi anipistundi", 1, "Harassment"),
    Task("spam_001", "spam", "Hot girls near you! Click: http://bit.ly/x9kQ", 1, "Spam link"),
]

ABUSE_KEYWORDS = ["dengu", "lanja", "randi", "bokka", "modda", "puku"]


def rule_based_flag(task: Task) -> tuple[bool, str]:
    text = task.text.lower()
    if task.category == "spam":
        if re.search(r"https?://[\w./-]*(bit\.ly|free|prize|win|click|hot|girls)", text):
            return True, "Spam pattern matched"
        return False, "No rule matched"

    if any(keyword in text for keyword in ABUSE_KEYWORDS):
        return True, "Abuse keyword matched"

    if task.category == "harassment" and re.search(r"(aadhi|vaalu|chusi|paduko|kitchen)", text):
        return True, "Harassment pattern matched"

    return False, "No rule matched"


def compute_reward(task: Task, action: AgentAction) -> RewardResult:
    blocked = action.normalized_decision() == "BLOCK"
    harmful = bool(task.label)
    score = 1.0 if blocked == harmful else 0.0
    feedback = "Correct" if score == 1.0 else "Incorrect"
    return RewardResult(
        task_id=task.task_id,
        score=score,
        breakdown={"blocked": blocked, "harmful": harmful},
        feedback=feedback,
    )
