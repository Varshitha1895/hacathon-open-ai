import re
import os
import json
import requests
from dataclasses import dataclass, field
from typing import Optional

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "hf_your_token_here")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/models")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

@dataclass
class Task:
    task_id: str
    category: str
    text: str
    label: int
    context: Optional[str] = None

@dataclass
class AgentAction:
    task_id: str
    decision: str
    confidence: float
    reason: str

@dataclass
class RewardResult:
    task_id: str
    score: float
    breakdown: dict = field(default_factory=dict)
    feedback: str = ""

TASK_BANK = [
    Task("abuse_001", "abuse", "Nee amma ni dengutanu", 1, "Vulgar comment"),
    Task("harassment_001", "harassment", "Aa aadhi vaalu chusi anipistundi", 1, "Harassment"),
    Task("spam_001", "spam", "Hot girls near you! Click: http://bit.ly/x9kQ", 1, "Spam link")
]

def rule_based_flag(task: Task):
    text = task.text.lower()
    abuse_keywords = ["dengu", "lanja", "randi", "bokka"]
    if any(kw in text for kw in abuse_keywords):
        return True, "Abuse keyword matched"
    return False, "No rule matched"

def compute_reward(task: Task, action: AgentAction):
    blocked = action.decision == "BLOCK"
    harmful = bool(task.label)
    score = 1.0 if blocked == harmful else 0.0
    return RewardResult(task.task_id, score, feedback="Correct" if score == 1.0 else "Incorrect")
