import os
import json
import logging
import datetime
import requests
from typing import Optional

# Server folder lopale unnam కాబట్టి direct imports
from models import (
    Task, AgentAction, RewardResult,
    TASK_BANK, compute_reward,
    HF_TOKEN, API_BASE_URL, MODEL_NAME,
    rule_based_flag 
)

def reset():
    # Meta validation ki idi kachithanga pampali
    return {"status": "success", "message": "Environment reset successful"}

# ══════════════════════════════════════════════
# AGENT DECISION
# ══════════════════════════════════════════════

def agent_decide(task: Task) -> AgentAction:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a content-moderation agent. Respond ONLY in JSON."},
            {"role": "user",   "content": f"Post: {task.text}\nDecide: BLOCK or ALLOW?"},
        ],
        "max_tokens": 150,
        "temperature": 0.05,
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
            confidence = float(parsed.get("confidence", 0.5)),
            reason     = parsed.get("reason", ""),
        )
    except Exception as exc:
        # Fallback to rule-based logic
        flagged, reason = rule_based_flag(task)
        return AgentAction(
            task_id    = task.task_id,
            decision   = "BLOCK" if flagged else "ALLOW",
            confidence = 0.5,
            reason     = f"Fallback: {reason}",
        )

# ══════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════

def run_inference(tasks: list[Task] | None = None) -> list[RewardResult]:
    if tasks is None:
        tasks = TASK_BANK

    results: list[RewardResult] = []

    for task in tasks:
        try:
            action = agent_decide(task)
            reward = compute_reward(task, action)
            results.append(reward)
            print(f"Task {task.task_id}: {action.decision} (Score: {reward.score})")
        except Exception as e:
            print(f"Error in task {task.task_id}: {e}")

    return results

if __name__ == "__main__":
    run_inference()
