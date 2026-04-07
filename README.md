---
title: Email Triage Environment
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
python_version: "3.10"
pinned: false
---

# 📧 Email Triage Environment

An OpenEnv-compliant real-world simulation designed for training and evaluating AI agents in automated email management. Built for the **Meta × PyTorch × Hugging Face × Scaler Hackathon**.

## 📝 Description & Motivation
In professional settings, email management is a high-volume, repetitive task that requires nuance. This environment simulates a corporate inbox where an agent must act as a first-responder triage system.

The goal is to move beyond simple "chat" and evaluate an agent's ability to perform **deterministic actions** (categorizing, routing, and drafting) that integrate with business workflows.

---

## 👁️ Observation Space
The environment provides a structured JSON observation containing:
* `email_id`: Unique identifier for the current session.
* `subject`: The email's subject line.
* `body`: The full text content of the email.
* `sender`: The originator's email address.
* `task`: The current sub-task (classify, route, or reply).

## ⚡ Action Space
The agent responds with a single string representing its decision:
* **Classify:** `["urgent", "normal", "low"]`
* **Route:** `["engineering", "hr", "finance", "sales", "support"]`
* **Reply:** A professional text string draft.

---

## 🎯 Task Descriptions & Difficulty
| Task | Difficulty | Description |
| :--- | :--- | :--- |
| **email_classify** | Easy | Identifying the priority level based on sentiment and keywords. |
| **email_route** | Medium | Determining the correct department based on technical or administrative context. |
| **email_reply** | Hard | Generating a contextually aware, professional response that addresses the user's specific query. |

---

## 🏆 Reward Logic
* **Deterministic Match:** 1.0 reward for exact category matches in classification and routing.
* **Partial Credit:** 0.4 - 0.8 reward for routing to "related" departments.
* **Semantic Quality:** For replies, rewards are calculated based on a combination of length, professional greeting, and keyword coverage (0.0 - 1.0).

---

## 🚀 Setup & Usage

### Local Development
```bash
git clone [https://github.com/ambrish17/ai-text-summarizer](https://github.com/ambrish17/ai-text-summarizer)
pip install -r requirements.txt
python app.py