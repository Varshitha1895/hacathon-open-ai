"""FastAPI environment server for the OpenEnv moderation task."""

from __future__ import annotations

import threading
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from .models import AgentAction, Observation, RewardResult, StepResult, TASK_BANK, Task, compute_reward, rule_based_flag
except ImportError:
    from server.models import AgentAction, Observation, RewardResult, StepResult, TASK_BANK, Task, compute_reward, rule_based_flag

app = FastAPI(title="OpenEnv Moderation Server", version="1.0.0")

_lock = threading.Lock()
_state: dict[str, Any] = {
    "session_id": None,
    "index": 0,
    "step": 0,
    "done": True,
}


def _safe_observation(task: Task, step: int, done: bool, message: str = "") -> Observation:
    return Observation(
        task_id=task.task_id,
        category=task.category,
        text=task.text,
        context=task.context,
        step=step,
        done=done,
        message=message,
    )


def _current_task() -> Task:
    index = int(_state.get("index", 0))
    if index < 0 or index >= len(TASK_BANK):
        index = 0
    return TASK_BANK[index]


def _session_payload(status: str, observation: Observation | None = None, reward: RewardResult | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "session_id": _state.get("session_id"),
        "done": bool(_state.get("done", True)),
        "step": int(_state.get("step", 0)),
    }
    if observation is not None:
        payload["observation"] = observation.to_dict()
    if reward is not None:
        payload["reward"] = reward.to_dict()
    if info is not None:
        payload["info"] = info
    return payload


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


@app.get("/state")
async def state() -> JSONResponse:
    with _lock:
        task = _current_task()
        observation = _safe_observation(task, int(_state.get("step", 0)), bool(_state.get("done", True)), "current")
        return JSONResponse(_session_payload("ok", observation=observation, info={"task_count": len(TASK_BANK)}))


@app.post("/reset")
async def reset() -> JSONResponse:
    try:
        with _lock:
            import uuid

            _state["session_id"] = uuid.uuid4().hex
            _state["index"] = 0
            _state["step"] = 0
            _state["done"] = False
            task = _current_task()
            observation = _safe_observation(task, 0, False, "episode reset")
            return JSONResponse(
                _session_payload(
                    "success",
                    observation=observation,
                    info={"message": "Environment reset successful"},
                )
            )
    except Exception as exc:
        fallback = Observation(task_id="fallback", category="abuse", text="", context=None, step=0, done=True, message=str(exc))
        return JSONResponse(_session_payload("error", observation=fallback, info={"message": "reset failed"}), status_code=200)


@app.post("/step")
async def step(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        with _lock:
            if _state.get("done", True):
                task = _current_task()
                observation = _safe_observation(task, int(_state.get("step", 0)), True, "episode already complete")
                reward = RewardResult(task.task_id, 0.0, {"reason": "episode finished"}, "Episode already complete")
                return JSONResponse(_session_payload("success", observation=observation, reward=reward, info={"done": True}))

            task = _current_task()
            action = AgentAction.from_dict(body if isinstance(body, dict) else {}, default_task_id=task.task_id)
            reward = compute_reward(task, action)

            next_index = int(_state.get("index", 0)) + 1
            _state["step"] = int(_state.get("step", 0)) + 1
            _state["index"] = next_index if next_index < len(TASK_BANK) else len(TASK_BANK) - 1
            _state["done"] = next_index >= len(TASK_BANK)

            next_task = _current_task()
            observation = _safe_observation(
                next_task,
                int(_state.get("step", 0)),
                bool(_state.get("done", True)),
                "next task" if not _state.get("done", True) else "episode complete",
            )
            result = StepResult(
                observation=observation,
                reward=reward,
                done=bool(_state.get("done", True)),
                info={
                    "action": action.normalized_decision(),
                    "rule_flag": rule_based_flag(task)[0],
                },
            )
            return JSONResponse(_session_payload("success", observation=observation, reward=reward, info=result.info))
    except Exception as exc:
        with suppress(Exception):
            _state["done"] = True
        fallback_task = _current_task()
        observation = _safe_observation(fallback_task, int(_state.get("step", 0)), True, f"step failed: {exc}")
        reward = RewardResult(fallback_task.task_id, 0.0, {"error": str(exc)}, "Step failed safely")
        return JSONResponse(_session_payload("error", observation=observation, reward=reward, info={"message": str(exc)}), status_code=200)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
