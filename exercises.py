from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

## EXERCISE 1: Hello LLM Endpoint
class HelloRequest(BaseModel):
    text: str

@app.post("/hello-llm")
def hello(request: HelloRequest):
    hello_pipeline = pipeline("text-generation", model="distilgpt2")
    greeting = hello_pipeline(request)
    return {
        "text": request,
        "Greetings": greeting
    }