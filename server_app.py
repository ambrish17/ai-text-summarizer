"""
Email Triage Environment — FastAPI Server
==========================================
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from email_triage_env import (
    EmailTriageEnv,
    EmailTriageAction,
)

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compliant email triage simulation for AI agents.",
    version="1.0.0"
)

envs = {}
current_task = "email_classify"


class ResetRequest(BaseModel):
    task: str = "email_classify"


class StepRequest(BaseModel):
    action: str


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
            {"name": "email_classify", "difficulty": "easy",   "description": "Classify email priority as urgent, normal, or low"},
            {"name": "email_route",    "difficulty": "medium", "description": "Route email to correct department"},
            {"name": "email_reply",    "difficulty": "hard",   "description": "Draft a professional reply to the email"},
        ]
    }


@app.post("/reset")
async def reset(body: ResetRequest = None):
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
    return {"status": "ok", "task": task, "observation": obs.dict()}


@app.post("/step")
async def step(body: StepRequest):
    global current_task
    if current_task not in envs:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    env = envs[current_task]
    obs, reward, done, info = await env.step(EmailTriageAction(action=body.action))
    return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}


@app.get("/state")
async def state():
    global current_task
    if current_task not in envs:
        return {"task": current_task, "status": "not_initialized"}
    s = await envs[current_task].state()
    return s.dict()


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
