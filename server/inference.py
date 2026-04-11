import os
import json
import logging
import datetime
import requests
import sys
import re
from typing import Optional

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models import Task, AgentAction, RewardResult, TASK_BANK, compute_reward, HF_TOKEN, API_BASE_URL, MODEL_NAME, rule_based_flag 

def reset():
    return {"status": "success", "message": "Environment reset successful"}

def agent_decide(task: Task) -> AgentAction:
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a content-moderation agent. Respond ONLY in JSON."},
            {"role": "user", "content": f"Post: {task.text}\nDecide: BLOCK or ALLOW?"},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }
    try:
        url = f"{API_BASE_URL}/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return AgentAction(task.task_id, parsed.get("decision", "ALLOW").upper(), float(parsed.get("confidence", 0.8)), parsed.get("reason", ""))
    except:
        flagged, reason = rule_based_flag(task)
        return AgentAction(task.task_id, "BLOCK" if flagged else "ALLOW", 0.5, f"Fallback: {reason}")

def run_inference(tasks=None):
    if tasks is None: tasks = TASK_BANK
    results = [compute_reward(t, agent_decide(t)) for t in tasks]
    return results
