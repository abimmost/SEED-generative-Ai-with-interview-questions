from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
from typing import Optional
from PIL import Image
import io

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

# ## TEXT SUMMARIZER

# @app.post("/summarize")
# def summary(long_text):
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     summarized = summarizer(
#         long_text,
#         max_length=50,
#         min_length=15,
#         do_sample=False
#     )
#     return {
#         "Long Text": long_text,
#         "Summary": summarized
#     }

# ## SENTIMENT ANALYSIS

# @app.post("/sentiment")
# def senti(text: str):
#     sentiment_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
#     senti = sentiment_pipeline(text)

#     return {
#         "text": text,
#         "sentiment": senti
#     }

# ## MULTIMODAL IMAGE CAPTIONING

# @app.post("/caption-image")
# async def run_file(file:UploadFile=File(...)):
#     caption_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes))
#     captioned_image = caption_pipeline(image)
#     return {
#         "file": file.filename,
#         "caption": captioned_image
#     }

# @app.post("/caption-image-link")
# def caption_image_link(image_link: str):
#     caption_pipeline = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
#     captioned_image = caption_pipeline(image_link)
#     return {
#       "file": image_link,
#       "caption": captioned_image
#     }

