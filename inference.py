"""
Inference Script — Email Triage Environment
============================================
Runs an LLM agent against the EmailTriageEnv across 3 tasks:
  1. email_classify  (easy)
  2. email_route     (medium)
  3. email_reply     (hard)

Stdout format:
  [START] task=<task> env=email_triage model=<model>
  [STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
  [END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from email_triage_env import EmailTriageEnv, EmailTriageAction

# ─── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")

BENCHMARK    = "email_triage"
MAX_STEPS    = 5
TEMPERATURE  = 0.3
MAX_TOKENS   = 256
SUCCESS_THRESHOLD = 0.5

TASKS = ["email_classify", "email_route", "email_reply"]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant.
    You will be shown an email and given a specific task to perform.
    Be concise, accurate, and professional in your responses.
    Follow the instruction exactly as given.
""").strip()


# ─── Logging ─────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").strip()[:80]
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Call ────────────────────────────────────────────────────────────────

def get_agent_action(client: OpenAI, subject: str, body: str, sender: str, instruction: str, step: int, history: List[str]) -> str:
    history_text = "\n".join(history[-3:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
        From: {sender}
        Subject: {subject}
        Body: {body}

        Task: {instruction}

        Previous attempts:
        {history_text}

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
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "normal"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "normal"


# ─── Run One Task ─────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task: str) -> float:
    env = EmailTriageEnv(task=task)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        result_obs = obs
        result_done = False
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result_done:
                break

            action_text = get_agent_action(
                client=client,
                subject=result_obs.subject,
                body=result_obs.body,
                sender=result_obs.sender,
                instruction=result_obs.instruction,
                step=step,
                history=history
            )

            result_obs, reward, result_done, info = await env.step(
                EmailTriageAction(action=action_text)
            )

            error = None
            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_text, reward=reward, done=result_done, error=error)

            history.append(f"Step {step}: {action_text!r} -> reward {reward:.2f}")

            if result_done:
                break

        # Score = average reward across steps, clamped to [0,1]
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)

    finally:
        await env.close()
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
        print(f"Task {task} final score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"\n{'='*50}", flush=True)
    print(f"All task scores: {[f'{s:.3f}' for s in all_scores]}", flush=True)
    print(f"Average score: {avg:.3f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
