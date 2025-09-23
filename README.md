
#  Generative AI Excercises – Intelligent Endpoints with FastAPI

<img width="858" height="484" alt="image" src="https://github.com/user-attachments/assets/476792ba-69e5-40e7-8fa9-f91923b894b6" />


This repository contains **10 step-by-step assignments** for building **Generative AI applications** with:

* **Python + FastAPI**
* **LLMs from Hugging Face**
* **Multimodal Models** (Google GenAI / Hugging Face)
* **Naive RAG (Chroma / FAISS)**
* **Diffusion Models for Image Generation**

Each assignment you'll:
✅ Step-by-step guide
✅ Model info (size)
✅ Knowledge base / resources
✅ Lesson you’ll learn
✅ **7 Interview Questions**

---

# 📝 Assignments

---

### **1. Hello LLM Endpoint**

* **Goal**: `/hello-llm` → Generate text with Hugging Face LLM.
* **Model**: `distilgpt2` (\~82M params).
* **Lesson**: Learn how to call an LLM from FastAPI.
* **Resource**: [DistilGPT2](https://huggingface.co/distilgpt2)

**Interview Questions:**

1. What is a language model?
2. How does GPT-2 differ from GPT-3/4?
3. Why is `distilgpt2` considered lightweight?
4. What are tokens, and why do they matter in LLMs?
5. How do you handle prompt length limits?
6. Why expose models through an API instead of CLI?
7. What’s the risk of directly exposing LLMs without moderation?

---

### **2. Text Summarizer API**

* **Goal**: `/summarize` → Summarize long text.
* **Model**: `facebook/bart-large-cnn` (\~400M params).
* **Lesson**: Learn sequence-to-sequence summarization.
* **Resource**: [BART Paper](https://arxiv.org/abs/1910.13461)

**Interview Questions:**

1. What is abstractive vs extractive summarization?
2. Why is BART good for summarization?
3. What are encoder-decoder architectures?
4. How does beam search affect summary quality?
5. What are hallucinations in summarization?
6. What evaluation metrics exist (ROUGE, BLEU)?
7. How would you fine-tune BART on legal documents?

---

### **3. Sentiment Analysis API**

* **Goal**: `/sentiment` → Detect positive/negative sentiment.
* **Model**: `distilbert-base-uncased-finetuned-sst-2-english` (\~66M params).
* **Lesson**: Learn classification with transformers.
* **Resource**: [SST-2 Dataset](https://huggingface.co/datasets/sst2)

**Interview Questions:**

1. What is transfer learning in NLP?
2. Why use DistilBERT instead of BERT?
3. What dataset is SST-2?
4. What are embeddings in classification?
5. How do you evaluate classification performance?
6. What biases can exist in sentiment models?
7. How would you handle sarcasm in sentiment detection?

---

### **4. Multimodal Image Captioning**

* **Goal**: `/caption-image` → Upload an image, return caption.
* **Model**: `nlpconnect/vit-gpt2-image-captioning` (\~124M params).
* **Lesson**: Learn vision-language alignment.
* **Resource**: [COCO Dataset](https://cocodataset.org/#download)

**Interview Questions:**

1. How does ViT process images?
2. What role does GPT-2 play in captioning?
3. Why combine a vision encoder with a language decoder?
4. What datasets are used for captioning?
5. What challenges exist in image captioning?
6. How do you evaluate captions (BLEU, CIDEr)?
7. What real-world apps use captioning?

---

### **5. Naive RAG with Chroma**

* **Goal**: `/rag-query` → Query docs with retrieval.
* **Model**: `all-MiniLM-L6-v2` (\~33M params).
* **Lesson**: Learn embeddings + retrieval-augmented generation.
* **Resource**: [Chroma Docs](https://docs.trychroma.com/)

**Interview Questions:**

1. What is RAG and why is it useful?
2. How do embeddings represent meaning?
3. Why use Chroma as a vector DB?
4. What is cosine similarity in retrieval?
5. How do you update a knowledge base?
6. What is the risk of injecting irrelevant documents?
7. How does RAG differ from fine-tuning?

---

### **6. Naive RAG with FAISS**

* **Goal**: `/rag-faiss-query` → Same as above but with FAISS.
* **Model**: `all-MiniLM-L6-v2`.
* **Lesson**: Learn scalable vector search with FAISS.
* **Resource**: [FAISS Docs](https://faiss.ai/)

**Interview Questions:**

1. What is FAISS, and why is it fast?
2. What indexing methods does FAISS provide (IVF, HNSW)?
3. How does FAISS handle billions of vectors?
4. Compare FAISS vs Chroma.
5. What is approximate nearest neighbor (ANN) search?
6. How do you evaluate retrieval accuracy?
7. How would you deploy FAISS in production?

---

### **7. Multimodal Q\&A (Image + Text)**

* **Goal**: `/qa-image-text` → Ask a question about an image.
* **Models**: `blip2-flan-t5-xl` (\~3B params) or Google Gemini Vision.
* **Lesson**: Learn multimodal reasoning.
* **Resource**: [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)

**Interview Questions:**

1. What is visual question answering (VQA)?
2. How does BLIP-2 align vision + text?
3. What is the role of a frozen LLM in multimodal models?
4. What tasks benefit from multimodal inputs?
5. What challenges exist in multimodal learning?
6. How do you evaluate multimodal models?
7. What industries need multimodal AI?

---

### **8. Chain Multiple Tools**

* **Goal**: `/researcher` → Wikipedia fetch + summarization + sentiment.
* **Lesson**: Learn chaining AI tasks.
* **Resource**: [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)

**Interview Questions:**

1. What is tool chaining in AI?
2. Why combine multiple AI tools?
3. What challenges exist when chaining APIs?
4. How does orchestration differ from composition?
5. How to handle failures in one tool?
6. What is LangChain and why is it popular?
7. How would you monitor toolchain latency?

---

### **9. Intelligent Chat Endpoint**

* **Goal**: `/chat` → Smart routing for queries (LLM, RAG, image).
* **Lesson**: Learn adaptive decision-making in AI apps.
* **Resource**: [LLM Routing (LangChain)](https://docs.langchain.com/)

**Interview Questions:**

1. What is model routing?
2. How do you detect intent in queries?
3. How do you decide when to call RAG vs LLM?
4. What are risks of automatic routing?
5. How do you log and trace routed calls?
6. What metrics help evaluate a chat system?
7. How would you scale this system for enterprise use?

---

### **10. Diffusion Model – Image Generation**

* **Goal**: `/generate-image` → Generate images from text prompts.
* **Model**: `stable-diffusion-v1-5` (\~860M params).
* **Lesson**: Learn how diffusion models synthesize images.
* **Resource**: [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)

**Interview Questions:**

1. How do diffusion models generate images?
2. What is denoising in diffusion?
3. How does Stable Diffusion differ from DALL·E?
4. Why are diffusion models memory-intensive?
5. What ethical issues exist with generative images?
6. How do you optimize diffusion for faster inference?
7. What industries benefit from diffusion models?

---
"Learning never exhausts the mind." — Leonardo da Vinci


