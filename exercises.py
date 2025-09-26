from fastapi import FastAPI
from transformers import pipeline
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

## Hello LLM Endpoint

class Hello(BaseModel):
    text: str

@app.post("/hello-llm")
def hello(request:Hello):
    hello_pipeline = pipeline("text-generation", model="distilgpt2")
    greeting = hello_pipeline(request.text)
    return {
        "text": request.text,
        "Greetings": greeting
    }

## TEXT SUMMARIZER

class summary(BaseModel):
    long_text: str

@app.post("/summarize")
def summary(request:summary):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized = summarizer(
        request.long_text,
        max_length=50,
        min_length=15,
        do_sample=False
    )
    return {
        "Long Text": request.long_text,
        "Summary": summarized
    }

## SENTIMENT ANALYSIS

class Sentiment(BaseModel):
    text: str

@app.post("/sentiment")
def senti(request:Sentiment):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    senti = sentiment_pipeline(request.text)

    return {
        "text": request.text,
        "sentiment": senti
    }