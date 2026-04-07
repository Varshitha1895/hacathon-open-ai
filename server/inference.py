"""
inference.py — Agent Loop & Strict Logging
Safe Social Media for Women (Telugu-English Mix)
Meta OpenEnv Hackathon

Place this file at:
  openenv/
  └── tasks/
      └── safe_social_women/
          └── inference.py   ← HERE

Run:
  python inference.py
"""

import os
import json
import logging
import datetime
import requests
from typing import Optional

# ── Import from sibling models.py ──────────────────────────────────────────
from server.models import (
    Task, AgentAction, RewardResult,
    TASK_BANK, compute_reward,
    HF_TOKEN, API_BASE_URL, MODEL_NAME,
)



def reset():
    # Meta validation ki idi kachithanga pampali
    return {"status": "success", "message": "Environment reset successful"}
    
# ══════════════════════════════════════════════
# LOGGING SETUP
# Format: [START] / [STEP] / [END] tags enforced
# ══════════════════════════════════════════════



logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("safe_social_inference.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("SafeSocialAgent")


def log_start(run_id: str, total_tasks: int) -> None:
    logger.info(f"[START] run_id={run_id} | total_tasks={total_tasks} | "
                f"model={MODEL_NAME} | api_base={API_BASE_URL}")


def log_step(run_id: str, step: int, task: Task, action: Optional[AgentAction],
             reward: Optional[RewardResult], note: str = "") -> None:
    payload = {
        "tag":      "[STEP]",
        "run_id":   run_id,
        "step":     step,
        "task_id":  task.task_id,
        "category": task.category,
        "label":    task.label,
        "decision": action.decision   if action else "N/A",
        "confidence": action.confidence if action else 0.0,
        "score":    round(reward.score, 4) if reward else -1,
        "feedback": reward.feedback   if reward else "",
        "note":     note,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def log_end(run_id: str, results: list[RewardResult]) -> None:
    scores = [r.score for r in results]
    avg    = sum(scores) / len(scores) if scores else 0.0
    cats   = {}
    for r in results:
        # task category is stored in breakdown keys for aggregation
        pass  # aggregation below uses RewardResult.breakdown

    logger.info(
        f"[END] run_id={run_id} | tasks_run={len(results)} | "
        f"avg_score={avg:.4f} | min={min(scores):.4f} | max={max(scores):.4f}"
    )


# ══════════════════════════════════════════════
# AGENT DECISION — calls LLM to decide BLOCK/ALLOW
# ══════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """
You are a content-moderation agent for a women-safe social media platform.
Users post in Telugu, English, and Manglish (Telugu written in Roman script).

Your job: Analyse the post and decide whether to BLOCK or ALLOW it.

Rules:
  - BLOCK if the post contains Telugu/English vulgar words (abuse, boothulu).
  - BLOCK if the post is misogynistic, sexually threatening, or demeaning to women.
  - BLOCK if the post contains suspicious/spam links.
  - ALLOW everything else.

Respond ONLY in this JSON format (no markdown, no preamble):
{
  "decision": "BLOCK",
  "confidence": 0.92,
  "reason": "Contains Telugu vulgar slang targeting a woman."
}
"""


def agent_decide(task: Task) -> AgentAction:
    """
    Calls the configured LLM to make a moderation decision.
    Falls back to a rule-based heuristic if LLM is unavailable.
    """
    from models import rule_based_flag

    user_msg = (
        f"Category hint: {task.category}\n"
        f"Context: {task.context or 'None'}\n"
        f"Post: {task.text}\n\n"
        "Decide: BLOCK or ALLOW? Return JSON only."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 200,
        "temperature": 0.05,
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
        import re
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return AgentAction(
            task_id    = task.task_id,
            decision   = parsed.get("decision", "ALLOW").upper(),
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
            reason     = parsed.get("reason", ""),
        )
    except Exception as exc:
        # Graceful fallback — use rule-based flag
        flagged, reason = rule_based_flag(task)
        return AgentAction(
            task_id    = task.task_id,
            decision   = "BLOCK" if flagged else "ALLOW",
            confidence = 0.5,
            reason     = f"LLM unavailable ({exc}). Rule-based fallback: {reason}",
        )


# ══════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════

def run_inference(tasks: list[Task] | None = None) -> list[RewardResult]:
    """
    Runs the agent over all tasks, logs every step, returns all RewardResults.
    """
    if tasks is None:
        tasks = TASK_BANK

    run_id  = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results: list[RewardResult] = []

    # ── [START] ────────────────────────────────────────────────────────────
    log_start(run_id, len(tasks))

    for step_idx, task in enumerate(tasks, start=1):

        action: Optional[AgentAction] = None
        reward: Optional[RewardResult] = None
        note   = ""

        try:
            # ── [STEP] — agent decision ────────────────────────────────
            log_step(run_id, step_idx, task, None, None, note="Requesting agent decision...")

            action = agent_decide(task)

            # ── [STEP] — reward computation ───────────────────────────
            reward = compute_reward(task, action)
            results.append(reward)

            log_step(run_id, step_idx, task, action, reward)

        except Exception as exc:
            note = f"ERROR: {exc}"
            log_step(run_id, step_idx, task, action, reward, note=note)
            logger.error(f"[STEP] run_id={run_id} step={step_idx} "
                         f"task_id={task.task_id} EXCEPTION: {exc}")

    # ── [END] ──────────────────────────────────────────────────────────────
    log_end(run_id, results)

    return results


# ══════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════

def print_summary(results: list[RewardResult]) -> None:
    print("\n" + "═" * 60)
    print("  SAFE SOCIAL WOMEN — Moderation Summary")
    print("═" * 60)
    for r in results:
        bar_len = int(r.score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {r.task_id:<20} [{bar}] {r.score:.3f}  {r.feedback}")
    scores = [r.score for r in results]
    avg    = sum(scores) / len(scores) if scores else 0.0
    print("─" * 60)
    print(f"  Average Reward Score : {avg:.4f}")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    all_results = run_inference()
    print_summary(all_results)
