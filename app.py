import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from email_triage_env import EmailTriageEnv

# 1. Initialize FastAPI with metadata for OpenEnv compliance
app = FastAPI(
    title="Email Triage Environment",
    description="An environment for classifying, routing, and replying to emails.",
    version="1.0.0"
)

env = EmailTriageEnv()

class ResetRequest(BaseModel):
    task: str

class StepRequest(BaseModel):
    action: str

# --- Mandatory OpenEnv Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metadata")
async def get_metadata():
    return {
        "name": "email_triage_env",
        "description": "A real-world simulation of an email inbox triage system.",
        "version": "1.0.0"
    }

@app.get("/schema")
async def get_schema():
    return {
        "action": {"type": "string", "description": "The agent's decision or text response"},
        "observation": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string"},
                "subject": {"type": "string"}
            }
        },
        "state": {"type": "object", "description": "Current internal state"}
    }

@app.post("/mcp")
async def mcp_endpoint(payload: Dict[str, Any]):
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"status": "connected"}
    }

# --- Core Functional Endpoints ---

@app.post("/reset")
async def reset(request: ResetRequest):
    try:
        observation = await env.reset(task=request.task)
        return {"status": "ok", "task": request.task, "observation": observation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step(request: StepRequest):
    try:
        reward, done, observation = await env.step(request.action)
        return {"reward": reward, "done": done, "observation": observation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Port 7860 and Host 0.0.0.0 are MANDATORY for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)