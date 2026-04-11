"""Robust client for driving the OpenEnv moderation server."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    import requests
except ImportError as exc:
    print(json.dumps({"status": "error", "message": f"Missing dependency: {exc}"}, indent=2))
    sys.exit(0)

try:
    from server.models import AgentAction, Observation, RewardResult
except Exception as exc:
    print(json.dumps({"status": "error", "message": f"Import failed: {exc}"}, indent=2))
    sys.exit(0)

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
TIMEOUT = float(os.getenv("OPENENV_TIMEOUT", "15"))


def _parse_response(response: requests.Response) -> dict[str, Any]:
    try:
        return response.json()
    except Exception:
        return {"status": "error", "raw": response.text or ""}


def _default_action(observation: Observation) -> AgentAction:
    text = observation.text.lower()
    harmful = any(keyword in text for keyword in ("dengu", "lanja", "randi", "bokka", "hot girls", "bit.ly"))
    return AgentAction(
        task_id=observation.task_id,
        decision="BLOCK" if harmful else "ALLOW",
        confidence=0.9 if harmful else 0.7,
        reason="heuristic client decision",
    )


def _build_observation(payload: Any) -> Observation:
    if not isinstance(payload, dict):
        payload = {}
    return Observation(
        task_id=str(payload.get("task_id", "")),
        category=str(payload.get("category", "abuse")),
        text=str(payload.get("text", "")),
        context=payload.get("context"),
        step=int(payload.get("step", 0)) if payload.get("step") is not None else 0,
        done=bool(payload.get("done", False)),
        message=str(payload.get("message", "")),
    )


def run_inference(base_url: str = BASE_URL) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    session = requests.Session()
    session_id = ""

    try:
        reset_response = session.post(f"{base_url}/reset", timeout=TIMEOUT)
        reset_payload = _parse_response(reset_response)
        if reset_response.status_code != 200 or reset_payload.get("status") == "error":
            return [{"status": "error", "message": f"Reset failed: {reset_payload.get('message', reset_payload)}"}]

        session_id = str(reset_payload.get("session_id", ""))
        current = reset_payload.get("observation", {})

        for _ in range(32):
            observation = _build_observation(current)
            if observation.done:
                results.append({"observation": observation.to_dict(), "action": {}, "reward": {"score": 0.0, "feedback": "done"}})
                break

            action = _default_action(observation)
            step_payload_body = {**action.__dict__, "session_id": session_id}
            step_response = session.post(f"{base_url}/step", json=step_payload_body, timeout=TIMEOUT)
            step_payload = _parse_response(step_response)
            if step_response.status_code != 200 or step_payload.get("status") == "error":
                results.append(
                    {
                        "status": "error",
                        "message": f"Step failed: {step_payload.get('message', step_payload)}",
                        "action": step_payload_body,
                    }
                )
                break

            reward_data = step_payload.get("reward", {}) if isinstance(step_payload, dict) else {}

            reward = RewardResult(
                task_id=str(reward_data.get("task_id", observation.task_id)),
                score=float(reward_data.get("score", 0.0)) if reward_data.get("score") is not None else 0.0,
                breakdown=dict(reward_data.get("breakdown", {}) if isinstance(reward_data.get("breakdown", {}), dict) else {}),
                feedback=str(reward_data.get("feedback", "")),
            )
            results.append({"observation": observation.to_dict(), "action": step_payload_body, "reward": reward.to_dict()})

            next_observation_payload = step_payload.get("observation", {}) if isinstance(step_payload, dict) else {}
            current = next_observation_payload

    except Exception as exc:
        results.append({"status": "error", "message": str(exc), "type": type(exc).__name__})
    finally:
        session.close()

    return results


def main() -> None:
    output = run_inference()
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
