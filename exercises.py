from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="HF Exercises API",
    description="Multiple API endpoints built with FastAPI and Hugging Face Transformers",
    version="1.0.0",
    contact={
        "name": "abimmost",
        "email": "atsimbomgwe31@outlook.com"
    }
)

app.get("/")
def read_root():
    return {"message": "Welcome to the HF Exercises API. Visit '/docs' for API documentation."}

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