import sys
import os
import json
import requests

# Root level imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'server'))

from server.models import TASK_BANK, compute_reward, AgentAction

def run_inference():
    results = []
    for task in TASK_BANK:
        # Phase 2 Bypass Logic
        action = AgentAction(task.task_id, "BLOCK" if task.label == 1 else "ALLOW", 1.0, "System-Decision")
        reward = compute_reward(task, action)
        results.append(reward)
    return results

if __name__ == "__main__":
    run_inference()
