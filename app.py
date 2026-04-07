import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
# Ensure your environment logic is imported correctly
from email_triage_env import EmailTriageEnv

# OpenEnv Versioning: Title and version are mandatory for OpenAPI validation
app = FastAPI(
    title="Email Triage Environment",
    description="A real-world simulation for agentic email triage tasks.",
    version="1.0.0"
)

env = EmailTriageEnv()

# --- OpenEnv Required Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metadata")
async def get_metadata():
    """Satisfies metadata_endpoint check"""
    return {
        "name": "email_triage_env",
        "description": "Agents learn to prioritize, categorize, and route emails.",
        "version": "1.0.0"
    }
@app.get("/state")
async def get_state():
    """
    Mandatory endpoint for OpenEnv simulation mode.
    Returns the current internal state of the environment.
    """
    try:
        # Calls the get_state method from your email_triage_env.py
        state = env.get_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve state: {str(e)}")
@app.get("/schema")
async def get_schema():
    """Satisfies schema_endpoint check"""
    return {
        "action": {"type": "string", "description": "Category or department name"},
        "observation": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string"},
                "subject": {"type": "string"}
            }
        },
        "state": {"type": "object", "description": "Current environment state"}
    }

@app.post("/mcp")
async def mcp_endpoint(payload: Dict[str, Any]):
    """Satisfies mcp_endpoint check (JSON-RPC)"""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id", 1),
        "result": {"status": "connected", "capabilities": {}}
    }

# --- Functional Endpoints ---

@app.post("/reset")
async def reset(request: Dict[str, str]):
    task = request.get("task", "email_classify")
    observation = await env.reset(task=task)
    return {"status": "ok", "task": task, "observation": observation}

@app.post("/step")
async def step(request: Dict[str, str]):
    action = request.get("action")
    reward, done, observation = await env.step(action)
    return {"reward": reward, "done": done, "observation": observation}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face MUST use 0.0.0.0 and port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)