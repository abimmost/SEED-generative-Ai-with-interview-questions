from fastapi import FastAPI
from transformers import pipeline
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

## Hello LLM Endpoint

class Hello(BaseModel):
    text: str

@app.post("/hello")
def hello(request:Hello):
    hello_pipeline = pipeline("text-generation", model="distilgpt2")
    greeting = hello_pipeline(request.text)
    return {
        "text": request.text,
        "Greetings": greeting
    }