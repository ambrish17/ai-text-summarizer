import asyncio

class EmailTriageEnv:
    def __init__(self):
        self.current_task = None
        self.step_count = 0
        self.done = False

    async def reset(self, task: str):
        self.current_task = task
        self.step_count = 0
        self.done = False
        # Sample observation
        return {
            "email_id": "e003",
            "subject": "Meeting rescheduled",
            "body": "The meeting for tomorrow is moved to 4 PM.",
            "task": task
        }

    async def step(self, action: str):
        self.step_count += 1
        reward = 1.0 if action.lower() in ["normal", "engineering", "thanks"] else 0.0
        self.done = True
        return reward, self.done, {"message": "Action processed"}

    def get_state(self):
        return {
            "current_task": self.current_task,
            "step_count": self.step_count,
            "done": self.done
        }