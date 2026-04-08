"""
Inference Script — Email Triage Environment
============================================
Runs an LLM agent against the EmailTriageEnv across 3 tasks.

Stdout format:
  [START] task=<task> env=email_triage model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import asyncio
import os
import random
import textwrap
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")

BENCHMARK         = "email_triage"
MAX_STEPS         = 5
TEMPERATURE       = 0.3
MAX_TOKENS        = 256
SUCCESS_THRESHOLD = 0.5
TASKS             = ["email_classify", "email_route", "email_reply"]

# ─── Inline Models ───────────────────────────────────────────────────────────

class EmailTriageObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    task: str
    instruction: str
    current_step: int
    max_steps: int
    done: bool = False

class EmailTriageAction(BaseModel):
    action: str

# ─── Email Dataset ────────────────────────────────────────────────────────────

EMAILS = {
    "email_classify": [
        {"id": "e001", "subject": "URGENT: Server is down!", "body": "Our production server has been down for 2 hours. Customers cannot access the service. We need immediate help!", "sender": "ops@company.com", "correct_answer": "urgent", "keywords": ["urgent", "critical", "immediate", "emergency"]},
        {"id": "e002", "subject": "Monthly newsletter", "body": "Thank you for subscribing to our monthly newsletter.", "sender": "newsletter@promo.com", "correct_answer": "low", "keywords": ["low", "newsletter", "promotional"]},
        {"id": "e003", "subject": "Meeting rescheduled", "body": "Our team meeting has been rescheduled from 3pm to 4pm tomorrow.", "sender": "manager@company.com", "correct_answer": "normal", "keywords": ["normal", "medium", "meeting"]},
    ],
    "email_route": [
        {"id": "e004", "subject": "Bug in payment gateway", "body": "I found a critical bug in the payment processing module. Transactions are failing.", "sender": "dev@company.com", "correct_answer": "engineering", "keywords": ["engineering", "technical", "tech", "it"]},
        {"id": "e005", "subject": "Invoice overdue", "body": "Invoice #12345 for $5,000 is 30 days overdue. Please process payment.", "sender": "billing@vendor.com", "correct_answer": "finance", "keywords": ["finance", "billing", "accounting", "payment"]},
        {"id": "e006", "subject": "Job application", "body": "I am writing to apply for the Software Engineer position.", "sender": "applicant@gmail.com", "correct_answer": "hr", "keywords": ["hr", "human resources", "recruitment", "hiring"]},
    ],
    "email_reply": [
        {"id": "e007", "subject": "Request for refund", "body": "I purchased your product last week but it arrived damaged. I would like a full refund.", "sender": "customer@gmail.com", "correct_answer": "apology", "keywords": ["sorry", "apologize", "refund", "damaged", "replacement"]},
        {"id": "e008", "subject": "Question about pricing", "body": "I am interested in your enterprise plan. Could you send me pricing details?", "sender": "prospect@business.com", "correct_answer": "information", "keywords": ["pricing", "plan", "details", "information", "cost"]},
        {"id": "e009", "subject": "Thank you!", "body": "Your customer support was absolutely fantastic. Keep up the great work!", "sender": "happy@customer.com", "correct_answer": "gratitude", "keywords": ["thank", "appreciate", "grateful", "pleased", "wonderful"]},
    ]
}

INSTRUCTIONS = {
    "email_classify": "Classify this email priority as: urgent, normal, or low. Reply with just one word.",
    "email_route":    "Route this email to: engineering, finance, hr, sales, or support. Reply with just the department name.",
    "email_reply":    "Draft a short professional reply to this email.",
}

MAX_STEPS_PER_TASK = {"email_classify": 3, "email_route": 3, "email_reply": 5}

# ─── Graders ─────────────────────────────────────────────────────────────────

def grade(task: str, action: str, email: dict) -> float:
    action_lower = action.lower().strip()
    correct = email["correct_answer"]
    keywords = email["keywords"]
    if correct in action_lower:
        return 0.95
    matches = sum(1 for kw in keywords if kw in action_lower)
    if task == "email_reply":
        length_score = min(len(action_lower) / 200, 0.3)
        return min(max(length_score + matches * 0.2, 0.01), 0.95)
    if matches > 0:
        return min(0.4 + matches * 0.15, 0.90)
    return 0.05

# ─── Logging ─────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    action_clean = action.replace("\n", " ").strip()[:80]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM Call ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an expert email triage assistant. Be concise, accurate, and professional."

def get_agent_action(client: OpenAI, obs: EmailTriageObservation, history: List[str]) -> str:
    history_text = "\n".join(history[-3:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        From: {obs.sender}
        Subject: {obs.subject}
        Body: {obs.body}

        Task: {obs.instruction}
        Previous attempts: {history_text}

        Your response:
    """).strip()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "normal"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "normal"

# ─── Run One Task ─────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task: str) -> float:
    email = random.choice(EMAILS[task])
    max_steps = MAX_STEPS_PER_TASK[task]

    obs = EmailTriageObservation(
        email_id=email["id"],
        subject=email["subject"],
        body=email["body"],
        sender=email["sender"],
        task=task,
        instruction=INSTRUCTIONS[task],
        current_step=0,
        max_steps=max_steps,
        done=False
    )

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    done = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            if done:
                break

            action_text = get_agent_action(client, obs, history)
            reward = grade(task, action_text, email)
            done = reward >= 0.8 or step >= max_steps

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_text, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_text!r} -> reward {reward:.2f}")

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []

    for task in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*50}", flush=True)
        score = await run_task(client, task)
        all_scores.append(score)
        print(f"Task {task} score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"\nAll scores: {[f'{s:.3f}' for s in all_scores]}", flush=True)
    print(f"Average score: {avg:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
