"""
models.py — Task & Reward Definitions
Safe Social Media for Women (Telugu-English Mix)
Meta OpenEnv Hackathon

Place this file at:
  openenv/
  └── tasks/
      └── safe_social_women/
          └── models.py   ← HERE

Usage: imported by inference.py
"""

import re
import os
import json
import requests
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# 🔧 CONFIGURATION — fill these in or set as env vars
# ─────────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN",    "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")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/models")
MODEL_NAME  = os.getenv("MODEL_NAME",  "mistralai/Mistral-7B-Instruct-v0.2")
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# ══════════════════════════════════════════════

@dataclass
class Task:
    """Represents one moderation task."""
    task_id:     str
    category:    str          # "abuse" | "harassment" | "spam"
    text:        str          # raw social-media post (Telugu/English/Manglish)
    label:       int          # ground-truth: 1 = harmful, 0 = safe
    context:     Optional[str] = None   # optional extra context


@dataclass
class AgentAction:
    """Action produced by the moderation agent."""
    task_id:    str
    decision:   str           # "BLOCK" | "ALLOW"
    confidence: float         # 0.0–1.0
    reason:     str           # free-text explanation


@dataclass
class RewardResult:
    """Reward returned by a grader."""
    task_id:    str
    score:      float         # always in [0.0, 1.0]
    breakdown:  dict = field(default_factory=dict)
    feedback:   str = ""


# ══════════════════════════════════════════════
# SECTION 2 — TASK BANK
# Three distinct task types, each with sample instances
# ══════════════════════════════════════════════

TASK_BANK: list[Task] = [

    # ── TASK TYPE 1: Direct Abuse (Boothulu) ──────────────────────────────
    Task(
        task_id  = "abuse_001",
        category = "abuse",
        text     = "Nee amma ni dengutanu, pichi dengu",          # vulgar Telugu
        label    = 1,
        context  = "Comment on a women's cricket team post",
    ),
    Task(
        task_id  = "abuse_002",
        category = "abuse",
        text     = "What a fantastic goal by the team! 🎉",
        label    = 0,
        context  = "Sports appreciation comment",
    ),
    Task(
        task_id  = "abuse_003",
        category = "abuse",
        text     = "Randi daaniki bokka choopinchu sala",          # severe Telugu slur
        label    = 1,
        context  = "Reply to a female journalist's tweet",
    ),

    # ── TASK TYPE 2: Women-Related Harassment (Misogyny in Manglish) ──────
    Task(
        task_id  = "harassment_001",
        category = "harassment",
        text     = "Aa aadhi vaalu chusi anipistundi, ilaanti vadini inka elli padukovaali",
        label    = 1,
        context  = "Comment on a female politician's photo",
    ),
    Task(
        task_id  = "harassment_002",
        category = "harassment",
        text     = "Great article! Very informative, keep it up 👏",
        label    = 0,
        context  = "Neutral encouragement on a blog post",
    ),
    Task(
        task_id  = "harassment_003",
        category = "harassment",
        text     = "Akka ni chusi gas anipistundi, ika nee pani em chestundi",
        label    = 1,
        context  = "Reply to a woman's LinkedIn post",
    ),

    # ── TASK TYPE 3: Spam / Suspicious Links ──────────────────────────────
    Task(
        task_id  = "spam_001",
        category = "spam",
        text     = "Hot girls near you! Click: http://bit.ly/x9kQ FREE",
        label    = 1,
        context  = "Unsolicited DM",
    ),
    Task(
        task_id  = "spam_002",
        category = "spam",
        text     = "Mee profile lo oka suspicious link pettandi: http://free-prize-win.ru/claim",
        label    = 1,
        context  = "Comment with suspicious URL",
    ),
    Task(
        task_id  = "spam_003",
        category = "spam",
        text     = "Check out my new blog post on Telugu literature: https://medium.com/telugu-lit",
        label    = 0,
        context  = "Legitimate link share",
    ),
]


# ══════════════════════════════════════════════
# SECTION 3 — RULE-BASED PRE-FILTERS
# ══════════════════════════════════════════════

# Telugu/Manglish vulgar keywords (romanised & partial stems — extend as needed)
ABUSE_KEYWORDS = [
    "dengu", "dengutanu", "bokka", "modda", "puku",
    "lanja", "randi", "sala", "daaniki", "pooku",
    "naakodaka", "lavada", "gaandu", "fuck", "bitch",
    "whore", "slut", "bastard",
]

# Misogynistic / harassment patterns
HARASSMENT_PATTERNS = [
    r"paduko(vaali|va)",          # "should sleep with"
    r"(vaalu|figure|body)\s+chusi",
    r"gas\s+anipistundi",
    r"aadhi\s+vaalu",
    r"ika\s+nee\s+pani\s+em",
    r"(kitchen|pots|cooking)\s+(lo|ki|ki\s+po)",
]

# Spam / URL patterns
SPAM_PATTERNS = [
    r"https?://bit\.ly/",
    r"https?://tinyurl\.com/",
    r"https?://[^\s]*(free|prize|win|click|hot|girls)[^\s]*",
    r"https?://[^\s]*\.(ru|xyz|tk|ml|ga|cf)[/\s]",
    r"(free\s+gift|click\s+here|you\s+won|earn\s+money|hot\s+girls)",
]


def rule_based_flag(task: Task) -> tuple[bool, str]:
    """
    Fast heuristic pre-check.
    Returns (flagged: bool, reason: str).
    """
    text_lower = task.text.lower()

    if task.category == "abuse":
        for kw in ABUSE_KEYWORDS:
            if kw in text_lower:
                return True, f"Matched abuse keyword: '{kw}'"

    elif task.category == "harassment":
        for pat in HARASSMENT_PATTERNS:
            if re.search(pat, text_lower):
                return True, f"Matched harassment pattern: '{pat}'"

    elif task.category == "spam":
        for pat in SPAM_PATTERNS:
            if re.search(pat, text_lower, re.IGNORECASE):
                return True, f"Matched spam pattern: '{pat}'"

    return False, "No rule matched"


# ══════════════════════════════════════════════
# SECTION 4 — LLM-AS-JUDGE
# Understands Telugu vulgarity & Manglish nuance
# ══════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """
You are a bilingual content-safety judge who understands:
- Telugu language (vulgar slang, romanised/Manglish script, cultural context)
- English (standard abusive/misogynistic vocabulary)
- Code-switching patterns between Telugu and English common on Indian social media

Your job: Given a social-media post and its category, decide if it is HARMFUL (1) or SAFE (0).

Categories:
  abuse       — Direct vulgar or abusive language (Telugu "boothulu" or English slurs)
  harassment  — Misogynistic / sexually threatening content targeting women
  spam        — Suspicious links, unsolicited promotions, phishing bait

Respond ONLY with valid JSON — no markdown, no preamble:
{
  "verdict": 1,
  "confidence": 0.95,
  "reason": "Brief explanation"
}
"""


def llm_judge(task: Task, rule_flag: bool) -> dict:
    """
    Call the configured LLM to verify/override the rule-based flag.
    Returns dict: {verdict: int, confidence: float, reason: str}
    """
    user_prompt = (
        f"Category: {task.category}\n"
        f"Context: {task.context or 'None'}\n"
        f"Post: {task.text}\n"
        f"Rule-based pre-flag: {'YES' if rule_flag else 'NO'}\n\n"
        "Analyse carefully, considering Telugu/Manglish nuance. Return JSON."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }

    url = f"{API_BASE_URL}/v1/chat/completions"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as exc:
        # Graceful degradation — fall back to rule result
        return {
            "verdict":    int(rule_flag),
            "confidence": 0.5,
            "reason":     f"LLM unavailable ({exc}); falling back to rule-based flag.",
        }


# ══════════════════════════════════════════════
# SECTION 5 — GRADERS (reward functions)
# All scores are strictly in [0.0, 1.0]
# ══════════════════════════════════════════════

def _clamp(value: float) -> float:
    """Clamp any float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def grade_abuse(task: Task, action: AgentAction, judge: dict) -> RewardResult:
    """
    Reward logic for Task Type 1 — Direct Abuse.
    1.0  → agent correctly BLOCKed harmful content  (true positive)
    0.8  → agent correctly ALLOWed safe content      (true negative)
    0.2  → agent ALLOWed abusive content             (false negative — dangerous)
    0.0  → agent BLOCKed safe content                (false positive)
    """
    blocked  = action.decision == "BLOCK"
    harmful  = bool(judge.get("verdict", 0))
    conf     = _clamp(judge.get("confidence", 0.5))

    if harmful and blocked:
        score = _clamp(0.8 + 0.2 * conf)          # 0.8–1.0
        fb = "✅ Correctly blocked abusive content."
    elif not harmful and not blocked:
        score = _clamp(0.6 + 0.2 * conf)          # 0.6–0.8
        fb = "✅ Correctly allowed safe content."
    elif harmful and not blocked:
        score = _clamp(0.2 * (1 - conf))          # 0.0–0.2  (miss on abuse)
        fb = "❌ Missed abusive content — false negative."
    else:
        score = _clamp(0.1 * (1 - conf))          # 0.0–0.1  (over-blocked)
        fb = "⚠️  Over-blocked safe content — false positive."

    return RewardResult(
        task_id=task.task_id, score=score,
        breakdown={"harmful": harmful, "blocked": blocked, "judge_conf": conf},
        feedback=fb,
    )


def grade_harassment(task: Task, action: AgentAction, judge: dict) -> RewardResult:
    """
    Reward logic for Task Type 2 — Women-Related Harassment.
    Extra penalty for missing misogynistic content (score → near 0).
    """
    blocked = action.decision == "BLOCK"
    harmful = bool(judge.get("verdict", 0))
    conf    = _clamp(judge.get("confidence", 0.5))

    if harmful and blocked:
        score = _clamp(0.85 + 0.15 * conf)
        fb = "✅ Correctly blocked harassing content."
    elif not harmful and not blocked:
        score = _clamp(0.65 + 0.15 * conf)
        fb = "✅ Correctly allowed safe content."
    elif harmful and not blocked:
        score = _clamp(0.1 * (1 - conf))          # near-zero for missing harassment
        fb = "❌ Missed misogynistic/harassing content."
    else:
        score = _clamp(0.15 * (1 - conf))
        fb = "⚠️  Over-blocked safe content."

    return RewardResult(
        task_id=task.task_id, score=score,
        breakdown={"harmful": harmful, "blocked": blocked, "judge_conf": conf},
        feedback=fb,
    )


def grade_spam(task: Task, action: AgentAction, judge: dict) -> RewardResult:
    """
    Reward logic for Task Type 3 — Spam / Suspicious Links.
    URL pattern confidence is factored into the score.
    """
    blocked = action.decision == "BLOCK"
    harmful = bool(judge.get("verdict", 0))
    conf    = _clamp(judge.get("confidence", 0.5))

    if harmful and blocked:
        score = _clamp(0.75 + 0.25 * conf)
        fb = "✅ Correctly blocked spam/suspicious link."
    elif not harmful and not blocked:
        score = _clamp(0.60 + 0.20 * conf)
        fb = "✅ Correctly allowed legitimate content."
    elif harmful and not blocked:
        score = _clamp(0.15 * (1 - conf))
        fb = "❌ Missed spam/phishing content."
    else:
        score = _clamp(0.20 * (1 - conf))
        fb = "⚠️  Over-blocked legitimate link."

    return RewardResult(
        task_id=task.task_id, score=score,
        breakdown={"harmful": harmful, "blocked": blocked, "judge_conf": conf},
        feedback=fb,
    )


# Router
GRADER_MAP = {
    "abuse":       grade_abuse,
    "harassment":  grade_harassment,
    "spam":        grade_spam,
}


def compute_reward(task: Task, action: AgentAction) -> RewardResult:
    """
    Main entry point called by inference.py.
    1. Rule-based pre-filter
    2. LLM-as-judge (with Telugu/Manglish understanding)
    3. Category-specific grader → RewardResult with score in [0.0, 1.0]
    """
    rule_flag, _rule_reason = rule_based_flag(task)
    judge_result = llm_judge(task, rule_flag)
    grader = GRADER_MAP.get(task.category, grade_abuse)
    return grader(task, action, judge_result)
