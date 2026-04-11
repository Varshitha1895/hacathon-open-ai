"""FastAPI environment server for the OpenEnv moderation task."""

from __future__ import annotations

import threading
import uuid
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

try:
    from .models import AgentAction, Observation, RewardResult, StepResult, TASK_BANK, Task, compute_reward, rule_based_flag
except ImportError:
    from server.models import AgentAction, Observation, RewardResult, StepResult, TASK_BANK, Task, compute_reward, rule_based_flag

app = FastAPI(title="OpenEnv Moderation Server", version="1.0.0")

_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}
_latest_session_id: str | None = None


def _new_session_state(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "index": 0,
        "step": 0,
        "done": False,
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


def _current_task(state: dict[str, Any]) -> Task:
    index = int(state.get("index", 0))
    if index < 0 or index >= len(TASK_BANK):
        index = 0
    return TASK_BANK[index]


def _terminal_observation(task: Task, step: int, message: str) -> Observation:
    return Observation(
        task_id=task.task_id,
        category=task.category,
        text="",
        context=task.context,
        step=step,
        done=True,
        message=message,
    )


def _session_payload(
    state: dict[str, Any],
    status: str,
    observation: Observation | None = None,
    reward: RewardResult | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "session_id": state.get("session_id"),
        "done": bool(state.get("done", True)),
        "step": int(state.get("step", 0)),
    }
    if observation is not None:
        payload["observation"] = observation.to_dict()
    if reward is not None:
        payload["reward"] = reward.to_dict()
    if info is not None:
        payload["info"] = info
    return payload


def _resolve_session_id(request: Request, body: dict[str, Any] | None = None) -> str | None:
    if isinstance(body, dict):
        body_session_id = body.get("session_id")
        if body_session_id:
            return str(body_session_id)

    header_session_id = request.headers.get("x-session-id")
    if header_session_id:
        return header_session_id

    query_session_id = request.query_params.get("session_id")
    if query_session_id:
        return query_session_id

    return _latest_session_id


def _require_session(session_id: str | None) -> dict[str, Any]:
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id. Call /reset first.")

    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")

    return state


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


@app.get("/state")
async def state(request: Request) -> JSONResponse:
    with _lock:
        state = _require_session(_resolve_session_id(request))
        task = _current_task(state)
        observation = _safe_observation(task, int(state.get("step", 0)), bool(state.get("done", True)), "current")
        return JSONResponse(_session_payload(state, "ok", observation=observation, info={"task_count": len(TASK_BANK)}))


@app.post("/reset")
async def reset() -> JSONResponse:
    try:
        with _lock:
            global _latest_session_id

            session_id = uuid.uuid4().hex
            state = _new_session_state(session_id)
            _sessions[session_id] = state
            _latest_session_id = session_id

            task = _current_task(state)
            observation = _safe_observation(task, 0, False, "episode reset")
            return JSONResponse(
                _session_payload(
                    state,
                    "success",
                    observation=observation,
                    info={"message": "Environment reset successful"},
                )
            )
    except Exception as exc:
        fallback = Observation(task_id="fallback", category="abuse", text="", context=None, step=0, done=True, message=str(exc))
        fallback_state = {"session_id": None, "done": True, "step": 0}
        return JSONResponse(_session_payload(fallback_state, "error", observation=fallback, info={"message": "reset failed"}), status_code=500)


@app.post("/step")
async def step(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    try:
        with _lock:
            state = _require_session(_resolve_session_id(request, body if isinstance(body, dict) else None))
            if state.get("done", True):
                task = _current_task(state)
                observation = _terminal_observation(task, int(state.get("step", 0)), "episode already complete")
                reward = RewardResult(task.task_id, 0.0, {"reason": "episode finished"}, "Episode already complete")
                return JSONResponse(_session_payload(state, "success", observation=observation, reward=reward, info={"done": True}))

            task = _current_task(state)
            action = AgentAction.from_dict(body if isinstance(body, dict) else {}, default_task_id=task.task_id)
            reward = compute_reward(task, action)

            next_index = int(state.get("index", 0)) + 1
            state["step"] = int(state.get("step", 0)) + 1
            state["done"] = next_index >= len(TASK_BANK)
            state["index"] = min(next_index, len(TASK_BANK) - 1)

            if state["done"]:
                observation = _terminal_observation(task, int(state.get("step", 0)), "episode complete")
            else:
                next_task = _current_task(state)
                observation = _safe_observation(next_task, int(state.get("step", 0)), False, "next task")

            result = StepResult(
                observation=observation,
                reward=reward,
                done=bool(state.get("done", True)),
                info={
                    "action": action.normalized_decision(),
                    "rule_flag": rule_based_flag(task)[0],
                },
            )
            return JSONResponse(_session_payload(state, "success", observation=observation, reward=reward, info=result.info))
    except HTTPException as exc:
        return JSONResponse({"status": "error", "message": exc.detail}, status_code=exc.status_code)
    except Exception as exc:
        with suppress(Exception):
            if isinstance(body, dict):
                failed_state = _sessions.get(str(body.get("session_id", "")))
                if failed_state is not None:
                    failed_state["done"] = True
        fallback_state = {"session_id": None, "index": 0, "step": 0, "done": True}
        fallback_task = _current_task(fallback_state)
        observation = _terminal_observation(fallback_task, int(fallback_state.get("step", 0)), f"step failed: {exc}")
        reward = RewardResult(fallback_task.task_id, 0.0, {"error": str(exc)}, "Step failed safely")
        return JSONResponse(_session_payload(fallback_state, "error", observation=observation, reward=reward, info={"message": str(exc)}), status_code=500)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
