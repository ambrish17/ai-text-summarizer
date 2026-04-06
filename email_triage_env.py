"""
Email Triage Environment
========================
A real-world email triage simulation where an AI agent must:
- Classify emails by priority (urgent/normal/low)
- Assign emails to correct departments
- Draft appropriate reply templates

Tasks:
  1. email_classify  (easy)   — classify email priority
  2. email_route     (medium) — route email to correct department
  3. email_reply     (hard)   — draft appropriate reply
"""

import asyncio
import random
from typing import Optional, List
from pydantic import BaseModel


# ─── Pydantic Models ────────────────────────────────────────────────────────

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
    action: str  # the agent's response/decision


class EmailTriageReward(BaseModel):
    reward: float
    done: bool
    info: dict


class EmailTriageState(BaseModel):
    task: str
    email_id: str
    current_step: int
    max_steps: int
    done: bool
    total_reward: float


# ─── Email Dataset ───────────────────────────────────────────────────────────

EMAILS = {
    "email_classify": [
        {
            "id": "e001",
            "subject": "URGENT: Server is down, production affected!",
            "body": "Our main production server has been down for 2 hours. Customers cannot access the service. We need immediate help!",
            "sender": "ops@company.com",
            "correct_answer": "urgent",
            "keywords": ["urgent", "critical", "immediate", "asap", "emergency", "down", "production"]
        },
        {
            "id": "e002",
            "subject": "Monthly newsletter subscription",
            "body": "Thank you for subscribing to our monthly newsletter. You will receive updates every month.",
            "sender": "newsletter@promo.com",
            "correct_answer": "low",
            "keywords": ["low", "unimportant", "newsletter", "promotional", "not urgent"]
        },
        {
            "id": "e003",
            "subject": "Meeting rescheduled for tomorrow",
            "body": "Hi, just a heads up that our team meeting has been rescheduled from 3pm to 4pm tomorrow.",
            "sender": "manager@company.com",
            "correct_answer": "normal",
            "keywords": ["normal", "medium", "rescheduled", "meeting", "tomorrow"]
        },
    ],
    "email_route": [
        {
            "id": "e004",
            "subject": "Bug in payment gateway",
            "body": "I found a critical bug in the payment processing module. Transactions are failing for some users.",
            "sender": "dev@company.com",
            "correct_answer": "engineering",
            "keywords": ["engineering", "technical", "development", "tech", "it"]
        },
        {
            "id": "e005",
            "subject": "Invoice overdue - Account #12345",
            "body": "This is a reminder that invoice #12345 for $5,000 is 30 days overdue. Please process payment immediately.",
            "sender": "billing@vendor.com",
            "correct_answer": "finance",
            "keywords": ["finance", "billing", "accounting", "payment", "invoice", "accounts"]
        },
        {
            "id": "e006",
            "subject": "Job application - Software Engineer",
            "body": "I am writing to apply for the Software Engineer position advertised on your website. Please find my resume attached.",
            "sender": "applicant@gmail.com",
            "correct_answer": "hr",
            "keywords": ["hr", "human resources", "recruitment", "hiring", "people"]
        },
    ],
    "email_reply": [
        {
            "id": "e007",
            "subject": "Request for product refund",
            "body": "I purchased your product last week but it arrived damaged. I would like a full refund please.",
            "sender": "customer@gmail.com",
            "correct_answer": "apology",
            "keywords": ["sorry", "apologize", "apology", "refund", "damaged", "replacement", "compensation", "regret"]
        },
        {
            "id": "e008",
            "subject": "Question about pricing plans",
            "body": "Hi, I am interested in your enterprise plan. Could you please send me more details about pricing?",
            "sender": "prospect@business.com",
            "correct_answer": "information",
            "keywords": ["pricing", "plan", "enterprise", "details", "information", "cost", "package", "offer"]
        },
        {
            "id": "e009",
            "subject": "Thank you for great service!",
            "body": "I just wanted to say that your customer support team was absolutely fantastic. Keep up the great work!",
            "sender": "happy@customer.com",
            "correct_answer": "gratitude",
            "keywords": ["thank", "appreciate", "grateful", "pleased", "glad", "wonderful", "great", "excellent"]
        },
    ]
}


# ─── Graders ────────────────────────────────────────────────────────────────

def grade_classify(action: str, email: dict) -> float:
    """Grade email classification task."""
    action_lower = action.lower().strip()
    correct = email["correct_answer"]
    keywords = email["keywords"]

    # Exact match
    if correct in action_lower:
        return 1.0

    # Partial keyword match
    matches = sum(1 for kw in keywords if kw in action_lower)
    if matches > 0:
        return min(0.5 + (matches * 0.1), 0.9)

    return 0.0


def grade_route(action: str, email: dict) -> float:
    """Grade email routing task."""
    action_lower = action.lower().strip()
    correct = email["correct_answer"]
    keywords = email["keywords"]

    if correct in action_lower:
        return 1.0

    matches = sum(1 for kw in keywords if kw in action_lower)
    if matches > 0:
        return min(0.4 + (matches * 0.15), 0.85)

    return 0.0


def grade_reply(action: str, email: dict) -> float:
    """Grade email reply task."""
    action_lower = action.lower().strip()
    keywords = email["keywords"]

    if len(action_lower) < 20:
        return 0.0

    matches = sum(1 for kw in keywords if kw in action_lower)
    length_score = min(len(action_lower) / 200, 0.3)
    keyword_score = min(matches * 0.2, 0.7)

    return min(length_score + keyword_score, 1.0)


GRADERS = {
    "email_classify": grade_classify,
    "email_route": grade_route,
    "email_reply": grade_reply,
}

INSTRUCTIONS = {
    "email_classify": "Classify this email priority as: urgent, normal, or low. Reply with just one word.",
    "email_route": "Route this email to the correct department: engineering, finance, hr, sales, or support. Reply with just the department name.",
    "email_reply": "Draft a short professional reply to this email. Be empathetic and helpful.",
}

MAX_STEPS = {
    "email_classify": 3,
    "email_route": 3,
    "email_reply": 5,
}


# ─── Environment Class ───────────────────────────────────────────────────────

class EmailTriageEnv:
    def __init__(self, task: str = "email_classify"):
        assert task in EMAILS, f"Unknown task: {task}. Choose from {list(EMAILS.keys())}"
        self.task = task
        self.emails = EMAILS[task]
        self.grader = GRADERS[task]
        self.instruction = INSTRUCTIONS[task]
        self.max_steps = MAX_STEPS[task]

        self._current_email = None
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._last_reward = 0.0

    async def reset(self) -> EmailTriageObservation:
        """Reset the environment and return initial observation."""
        self._current_email = random.choice(self.emails)
        self._current_step = 0
        self._done = False
        self._total_reward = 0.0
        self._last_reward = 0.0

        return EmailTriageObservation(
            email_id=self._current_email["id"],
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            sender=self._current_email["sender"],
            task=self.task,
            instruction=self.instruction,
            current_step=self._current_step,
            max_steps=self.max_steps,
            done=False
        )

    async def step(self, action: EmailTriageAction) -> tuple:
        """Take a step in the environment."""
        if self._done:
            return (
                EmailTriageObservation(
                    email_id=self._current_email["id"],
                    subject=self._current_email["subject"],
                    body=self._current_email["body"],
                    sender=self._current_email["sender"],
                    task=self.task,
                    instruction=self.instruction,
                    current_step=self._current_step,
                    max_steps=self.max_steps,
                    done=True
                ),
                0.0,
                True,
                {"info": "Episode already done"}
            )

        self._current_step += 1

        # Grade the action
        reward = self.grader(action.action, self._current_email)
        self._last_reward = reward
        self._total_reward += reward

        # Done if correct answer or max steps reached
        done = reward >= 0.8 or self._current_step >= self.max_steps
        self._done = done

        obs = EmailTriageObservation(
            email_id=self._current_email["id"],
            subject=self._current_email["subject"],
            body=self._current_email["body"],
            sender=self._current_email["sender"],
            task=self.task,
            instruction=self.instruction,
            current_step=self._current_step,
            max_steps=self.max_steps,
            done=done
        )

        info = {
            "correct_answer": self._current_email["correct_answer"],
            "graded_reward": reward,
            "step": self._current_step
        }

        return obs, reward, done, info

    async def state(self) -> EmailTriageState:
        """Return current state."""
        return EmailTriageState(
            task=self.task,
            email_id=self._current_email["id"] if self._current_email else "",
            current_step=self._current_step,
            max_steps=self.max_steps,
            done=self._done,
            total_reward=self._total_reward
        )

    async def close(self):
        """Cleanup."""
        pass
