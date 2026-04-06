# 🧠 Email Triage Environment

An OpenEnv-compliant real-world email triage simulation where AI agents must classify, route, and reply to emails.

Built for **Meta × PyTorch × Hugging Face × Scaler Hackathon**.

---

## 📋 Environment Description

Email triage is a task humans do every day — reading emails and deciding:
- How urgent is this? (classify)
- Who should handle this? (route)
- What should we say back? (reply)

This environment simulates that workflow with 3 progressive tasks.

---

## 🎯 Tasks

| Task | Difficulty | Description |
|---|---|---|
| `email_classify` | Easy | Classify email as urgent / normal / low |
| `email_route` | Medium | Route to engineering / finance / hr / sales / support |
| `email_reply` | Hard | Draft a professional reply |

---

## 👁️ Observation Space

```json
{
  "email_id": "string",
  "subject": "string",
  "body": "string",
  "sender": "string",
  "task": "string",
  "instruction": "string",
  "current_step": "integer",
  "max_steps": "integer",
  "done": "boolean"
}
```

## ⚡ Action Space

```json
{
  "action": "string — agent's response or decision"
}
```

## 🏆 Reward Function

- **email_classify**: 1.0 for exact match, 0.5–0.9 for partial keyword match
- **email_route**: 1.0 for correct department, 0.4–0.85 for partial match
- **email_reply**: 0.0–1.0 based on length + relevant keyword coverage
- All rewards are in range [0.0, 1.0]
- Partial rewards encourage progress at every step

---

## 🚀 Setup & Usage

### Local setup
```bash
git clone https://github.com/ambrish17/ai-text-summarizer
cd ai-text-summarizer
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your_token \
  email-triage-env
```

### Run inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token
python inference.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Reset environment |
| POST | `/step` | Take action |
| GET | `/state` | Get current state |

### Example: Reset
```bash
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "email_classify"}'
```

### Example: Step
```bash
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "urgent"}'
```

---

## 📊 Baseline Scores

| Task | Baseline Score |
|---|---|
| email_classify | ~0.85 |
| email_route | ~0.75 |
| email_reply | ~0.65 |
| **Average** | **~0.75** |

---

## 🛠️ Tech Stack

- Python 3.10
- FastAPI + Uvicorn
- OpenAI client
- OpenEnv framework
- Hugging Face Spaces (Docker SDK)
