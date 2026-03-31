from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.env import VeloraEnv
from env.models import Action


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: Action


class EnvironmentMetadata(BaseModel):
    name: str = "velora-env"
    description: str = "AI Data Analyst training environment for OpenEnv."
    tag: str = "openenv"
    available_tasks: list[str] = Field(default_factory=list)


def create_app() -> FastAPI:
    app = FastAPI(title="Velora Env", version="1.0.0")
    env = VeloraEnv()
    lock = Lock()

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "velora-env",
            "tag": "openenv",
            "endpoints": ["/health", "/metadata", "/reset", "/step", "/state"],
        }

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata", response_model=EnvironmentMetadata)
    def metadata() -> EnvironmentMetadata:
        return EnvironmentMetadata(available_tasks=list(env.tasks.keys()))

    @app.post("/reset")
    def reset(request: ResetRequest | None = None) -> Dict[str, Any]:
        with lock:
            observation = env.reset(task_id=request.task_id if request else None)
            return {"observation": observation.model_dump(by_alias=True)}

    @app.post("/step")
    def step(request: StepRequest) -> Dict[str, Any]:
        with lock:
            if env.state_data is None:
                raise HTTPException(status_code=400, detail="Call /reset before /step.")
            observation, reward, done, info = env.step(request.action)
            return {
                "observation": observation.model_dump(by_alias=True),
                "reward": reward,
                "done": done,
                "info": info,
            }

    @app.get("/state")
    def state() -> Dict[str, Any]:
        with lock:
            if env.state_data is None:
                raise HTTPException(status_code=400, detail="Call /reset before /state.")
            return env.state()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False,
    )


if __name__ == "__main__":
    main()
