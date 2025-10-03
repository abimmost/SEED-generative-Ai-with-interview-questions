from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="HF Exercises API",
    description="Multiple API endpoints built with FastAPI and Hugging Face Transformers",
    version="1.0.0",
    # contact={
    #     "name": "abimmost",
    #     "email": "atsimbomgwe31@outlook.com"
    # }
)

app = APIRouter(tags=["Home"])

router = APIRouter(prefix="/api", tags=["Hugging Face"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the HF Exercises API. Visit '/docs' for API documentation."}

## EXERCISE 1: Hello LLM Endpoint
class HelloRequest(BaseModel):
    text: str

@router.post("/hello-llm")
def hello(request: HelloRequest):
    hello_pipeline = pipeline("text-generation", model="distilgpt2")
    greeting = hello_pipeline(request.text)
    print(greeting)
    return {
        "text": request.text,
        "Greetings": greeting
    }