from fastapi import FastAPI
from pydantic import BaseModel
from scripts.rag import ask_question

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: AskRequest):
    return {"answer": ask_question(payload.question)}