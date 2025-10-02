from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

## Hello LLM Endpoint

@app.post("/hello-llm")
def hello(text: str):
    hello_pipeline = pipeline("text-generation", model="distilgpt2")
    greeting = hello_pipeline(text)
    return {
        "text": text,
        "Greetings": greeting
    }