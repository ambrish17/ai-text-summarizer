"""
Email Triage Environment — FastAPI Server
==========================================
Exposes OpenEnv-compliant endpoints:
  POST /reset   — reset the environment
  POST /step    — take an action
  GET  /state   — get current state
  GET  /tasks   — list all tasks
  GET  /health  — health check
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from email_triage_env import (
    EmailTriageEnv,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
)

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compliant email triage simulation for AI agents.",
    version="1.0.0"
)

# Global env instance per task
envs = {}
current_task = "email_classify"


# ─── Request Models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "email_classify"


class StepRequest(BaseModel):
    action: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Email Triage Environment",
        "version": "1.0.0",
        "tasks": ["email_classify", "email_route", "email_reply"],
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "name": "email_classify",
                "difficulty": "easy",
                "description": "Classify email priority as urgent, normal, or low"
            },
            {
                "name": "email_route",
                "difficulty": "medium",
                "description": "Route email to correct department: engineering, finance, hr, sales, or support"
            },
            {
                "name": "email_reply",
                "difficulty": "hard",
                "description": "Draft a professional reply to the email"
            }
        ]
    }


@app.post("/reset")
async def reset(body: ResetRequest = None):
    """Reset the environment for a given task."""
    global current_task

    task = "email_classify"
    if body and body.task:
        task = body.task

    if task not in ["email_classify", "email_route", "email_reply"]:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    current_task = task
    env = EmailTriageEnv(task=task)
    envs[task] = env

    obs = await env.reset()
    return {
        "status": "ok",
        "task": task,
        "observation": obs.dict()
    }


@app.post("/step")
async def step(body: StepRequest):
    """Take a step in the current environment."""
    global current_task

    if current_task not in envs:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    env = envs[current_task]
    action = EmailTriageAction(action=body.action)

    obs, reward, done, info = await env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
async def state():
    """Get current environment state."""
    global current_task

    if current_task not in envs:
        return {
            "task": current_task,
            "status": "not_initialized",
            "message": "Call /reset to initialize"
        }

    env = envs[current_task]
    s = await env.state()
    return s.dict()


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
