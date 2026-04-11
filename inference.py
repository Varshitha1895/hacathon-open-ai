"""Robust client for driving the OpenEnv moderation server."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from server.models import AgentAction, Observation, RewardResult

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
TIMEOUT = float(os.getenv("OPENENV_TIMEOUT", "15"))


def _parse_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"status": "error", "raw": response.text}


def _default_action(observation: Observation) -> AgentAction:
    text = observation.text.lower()
    harmful = any(keyword in text for keyword in ("dengu", "lanja", "randi", "bokka", "hot girls", "bit.ly"))
    return AgentAction(
        task_id=observation.task_id,
        decision="BLOCK" if harmful else "ALLOW",
        confidence=0.9 if harmful else 0.7,
        reason="heuristic client decision",
    )


def run_inference(base_url: str = BASE_URL) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    session = requests.Session()

    try:
        reset_response = session.post(f"{base_url}/reset", timeout=TIMEOUT)
        reset_payload = _parse_response(reset_response)
        current = reset_payload.get("observation", {})

        for _ in range(32):
            observation = Observation(
                task_id=str(current.get("task_id", "")),
                category=str(current.get("category", "abuse")),
                text=str(current.get("text", "")),
                context=current.get("context"),
                step=int(current.get("step", 0)),
                done=bool(current.get("done", False)),
                message=str(current.get("message", "")),
            )
            if observation.done:
                break

            action = _default_action(observation)
            step_response = session.post(f"{base_url}/step", json=action.__dict__, timeout=TIMEOUT)
            step_payload = _parse_response(step_response)
            reward_data = step_payload.get("reward", {})
            reward = RewardResult(
                task_id=str(reward_data.get("task_id", observation.task_id)),
                score=float(reward_data.get("score", 0.0)),
                breakdown=dict(reward_data.get("breakdown", {})),
                feedback=str(reward_data.get("feedback", "")),
            )
            results.append({"observation": observation.to_dict(), "action": action.__dict__, "reward": reward.to_dict()})
            current = step_payload.get("observation", {})
    except Exception as exc:
        results.append({"status": "error", "message": str(exc)})
    finally:
        session.close()

    return results


if __name__ == "__main__":
    output = run_inference()
    print(json.dumps(output, indent=2, ensure_ascii=True))
