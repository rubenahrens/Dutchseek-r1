"""
Fout: dictionary update sequence element #0 has length 6; 2 is required
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer, TextIteratorStreamer
import torch
import os
import uvicorn
import json
from typing import AsyncGenerator
from threading import Thread

app = FastAPI()

# CORS middleware toevoegen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check voor CUDA beschikbaarheid
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available")
device = "cuda"

# Model en tokenizer initialiseren voor DeepSeek
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Lokale vertaalmodellen initialiseren
nl_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
nl_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en").to(device)

en_to_nl_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
en_to_nl_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-nl").to(device)

def translate_nl_to_en(text):
    inputs = nl_to_en_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = nl_to_en_model.generate(**inputs)
    return nl_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_en_to_nl(text):
    inputs = en_to_nl_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = en_to_nl_model.generate(**inputs)
    return en_to_nl_tokenizer.decode(outputs[0], skip_special_tokens=True)

class ChatRequest(BaseModel):
    message: str

async def generate_streaming_response(message: str) -> AsyncGenerator[str, None]:
    try:
        # Nederlandse input naar Engels vertalen met lokaal model
        translated_input = translate_nl_to_en(message)
        
        # Model input voorbereiden
        inputs = tokenizer(translated_input, return_tensors="pt").to(device)
        
        # Streamer initialiseren
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generatie in een aparte thread starten
        generation_kwargs = dict(
            inputs["input_ids"],
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Start generatie in een aparte thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream de tokens
        for text in streamer:
            yield f"data: {json.dumps({'text': text})}\n\n"
        
        # Vertaal het volledige antwoord naar Nederlands
        translated_response = translate_en_to_nl(text)
        yield f"data: {json.dumps({'text': translated_response, 'done': True})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_streaming_response(request.message),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)