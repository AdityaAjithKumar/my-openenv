# server/app.py
import os
import uvicorn
from typing import Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app
from models import InjectionDetectionAction, InjectionDetectionObservation
from .environment import PromptInjectionEnvironment

# Read task level from env var so one Docker image serves all 3 tasks
TASK_LEVEL = os.getenv("TASK_LEVEL", "easy")
SEED = int(os.getenv("ENV_SEED", "42"))


def create_environment():
    """Factory: one isolated env instance per WebSocket session."""
    return PromptInjectionEnvironment(task_level=TASK_LEVEL, seed=SEED)


app = create_app(
    create_environment,
    InjectionDetectionAction,
    InjectionDetectionObservation,
    env_name="prompt-injection-detector",
)


@app.get("/")
def root():
    """Root endpoint for HuggingFace Spaces health check and discoverability."""
    return {
        "environment": "prompt-injection-detector",
        "version": "1.0.0",
        "description": "RL environment for training AI agents to detect prompt injection attacks",
        "task_level": TASK_LEVEL,
        "endpoints": {
            "health": "GET /health",
            "schema": "GET /schema",
            "state":  "GET /state",
            "reset":  "POST /reset",
            "step":   "POST /step",
        },
        "tasks": ["easy", "medium", "hard"],
        "status": "running",
    }


@app.get("/health")
def health():
    """Liveness probe — evaluator and Docker HEALTHCHECK hit this endpoint."""
    return {"status": "ok", "environment": "prompt-injection-detector", "task_level": TASK_LEVEL}


@app.get("/state")
def state():
    """
    OpenEnv spec: expose current environment state.
    Returns a lightweight snapshot; full state is managed per-session by create_app.
    """
    return {
        "environment": "prompt-injection-detector",
        "task_level": TASK_LEVEL,
        "episode_length": PromptInjectionEnvironment.EPISODE_LENGTH,
        "tasks": [
            {
                "id": "easy",
                "name": "Direct Prompt Injection Detection",
                "difficulty": "easy",
                "score_range": [0.0, 1.0],
            },
            {
                "id": "medium",
                "name": "Indirect / Document-Embedded Injection Detection",
                "difficulty": "medium",
                "score_range": [0.0, 1.0],
            },
            {
                "id": "hard",
                "name": "Obfuscated / Steganographic Injection Detection",
                "difficulty": "hard",
                "score_range": [0.0, 1.0],
            },
        ],
    }


@app.post("/reset")
async def reset_override(request: Request):
    """
    Override /reset to accept an empty body {}.
    The validator pings POST /reset with no task_level field.
    Falls back to the server's TASK_LEVEL env var when task_level is absent.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    task_level = body.get("task_level", TASK_LEVEL)
    env = create_environment()
    obs = env.reset()
    return JSONResponse({
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "done": False,
        "reward": 0.0,
    })


def main():
    """Entry point for running the environment server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

