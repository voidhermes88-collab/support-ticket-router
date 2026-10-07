from transformers import pipeline
import json
from datetime import datetime

# Agent 1: Classifier - NLP + ML
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CANDIDATE_TOPICS = ["billing", "technical issue", "shipping", "account access"]
CANDIDATE_URGENCY = ["low", "medium", "high"]

def agent_1_classify(ticket_text: str):
    topic_res = classifier(ticket_text, CANDIDATE_TOPICS)
    urgency_res = classifier(ticket_text, CANDIDATE_URGENCY)
    return {
        "topic": topic_res["labels"][0],
        "topic_score": round(topic_res["scores"][0], 2),
        "urgency": urgency_res["labels"][0],
        "urgency_score": round(urgency_res["scores"][0], 2)
    }

# Agent 2: Router
def agent_2_route(classification):
    routing = {
        "billing": "#finance-support",
        "technical issue": "#tech-support",
        "shipping": "#logistics",
        "account access": "#security-team"
    }
    priority = "P1" if classification["urgency"] == "high" else "P2"
    return {
        "channel": routing[classification["topic"]],
        "priority": priority,
        "sla": "2h" if priority == "P1" else "24h"
    }

# Agent 3: Auto-response draft
def agent_3_draft(ticket, classification, routing):
    return f"Hi, your {classification['topic']} ticket marked as {classification['urgency']} urgency routed to {routing['channel']} with {routing['priority']} ({routing['sla']} SLA)."

# Demo runner
if __name__ == "__main__":
    with open("tickets.json") as f:
        tickets = json.load(f)

    for t in tickets:
        print(f"\n--- Ticket {t['id']} ---")
        cls = agent_1_classify(t["text"])
        route = agent_2_route(cls)
        draft = agent_3_draft(t, cls, route)
        print(f"Text: {t['text']}")
        print(f"Classification: {cls}")
        print(f"Routing: {route}")
        print(f"Draft: {draft}")
