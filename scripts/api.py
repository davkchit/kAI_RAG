import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from scripts.rag import ask_question


def strip_citation(text: str) -> str:
    return re.sub(r"\n?\[src:[^\]]+\]", "", text).strip()

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = FastAPI()


@app.on_event("startup")
async def register_telegram_webhook():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_WEBHOOK_URL:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API}/setWebhook",
            json={"url": f"{TELEGRAM_WEBHOOK_URL}/telegram/webhook"},
        )


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest):
    return {"answer": ask_question(payload.question)}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()

    if not chat_id or not text:
        return {"ok": True}

    answer = strip_citation(ask_question(text))

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": answer},
        )

    return {"ok": True}
