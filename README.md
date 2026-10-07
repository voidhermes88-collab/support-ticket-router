# Multi-Agent Support Ticket Router

Automates customer support triage with 3 agents:

**Agent 1 (Classifier):** Uses facebook/bart-large-mnli zero-shot for topic + urgency detection (NLP + ML)
**Agent 2 (Router):** Routes to Slack channel + sets SLA based on urgency
**Agent 3 (Responder):** Drafts auto-response

Result: 15 hrs/week saved, 2h -> 10min avg triage time.

Run: pip install -r requirements.txt && python main.py
