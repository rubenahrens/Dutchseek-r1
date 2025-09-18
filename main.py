from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer, TextIteratorStreamer
import torch
import os
import uvicorn
import json
import time
import re
from typing import AsyncGenerator, Dict, Optional, List
from threading import Thread
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Laad omgevingsvariabelen uit .env bestand
load_dotenv()

app = FastAPI()

# CORS middleware toevoegen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Beperk tot specifieke oorsprong
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Beperk tot noodzakelijke methoden
    allow_headers=["*"],
)

# Check voor CUDA beschikbaarheid
if not torch.cuda.is_available():
    raise RuntimeError("No GPU available")
device = "cuda"

# Model en tokenizer initialiseren voor Gemma
model_name = "google/gemma-3-4b-it"
# Lees token vanuit .env
access_token = os.getenv("HF_ACCESS_TOKEN")
if not access_token:
    raise ValueError("HF_ACCESS_TOKEN niet gevonden in .env bestand")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=access_token).to(device)

# Lokale vertaalmodellen initialiseren
nl_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
nl_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en").to(device)

en_to_nl_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")
en_to_nl_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-nl").to(device)

# Configuratie
MAX_INPUT_LENGTH = 1000  # Maximum aantal tekens in gebruikersinvoer
MAX_REQUESTS_PER_MINUTE = 10  # Simpelere rate limiting
BLOCKED_PATTERNS = [
    r"DROP\s+TABLE", 
    r"DELETE\s+FROM",
    r"<script>.*</script>",
    r"rm\s+-rf",
    r"sudo"
]

# Eenvoudige client-tracking voor rate limiting
client_request_history = {}

def sanitize_input(text: str) -> str:
    """Verwijder potentieel gevaarlijke patronen uit de invoer."""
    sanitized = text
    for pattern in BLOCKED_PATTERNS:
        sanitized = re.sub(pattern, "[GEBLOKKEERDE INHOUD]", sanitized, flags=re.IGNORECASE)
    return sanitized

def check_rate_limit(client_ip: str):
    """Simpelere rate limiting op basis van IP-adres"""
    now = datetime.now()
    
    if client_ip not in client_request_history:
        client_request_history[client_ip] = {
            "requests": 1,
            "reset_time": now + timedelta(minutes=1)
        }
        return True
    
    client_data = client_request_history[client_ip]
    
    # Reset teller als de tijd verstreken is
    if now >= client_data["reset_time"]:
        client_data["requests"] = 1
        client_data["reset_time"] = now + timedelta(minutes=1)
        return True
        
    # Controleer limiet
    if client_data["requests"] >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    # Verhoog teller
    client_data["requests"] += 1
    return True

def translate_nl_to_en(text):
    inputs = nl_to_en_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = nl_to_en_model.generate(**inputs)
    return nl_to_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_en_to_nl(text):
    inputs = en_to_nl_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = en_to_nl_model.generate(**inputs)
    return en_to_nl_tokenizer.decode(outputs[0], skip_special_tokens=True)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_INPUT_LENGTH)
    
    @validator('message')
    def validate_message(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Bericht mag niet leeg zijn")
        if len(v) > MAX_INPUT_LENGTH:
            raise ValueError(f"Bericht is te lang. Maximum is {MAX_INPUT_LENGTH} tekens.")
        return v

async def generate_streaming_response(message: str, client_ip: str) -> AsyncGenerator[str, None]:
    try:
        # Input validatie en sanitatie
        if len(message) > MAX_INPUT_LENGTH:
            yield f"data: {json.dumps({'error': f'Bericht is te lang. Maximum is {MAX_INPUT_LENGTH} tekens.'})}\n\n"
            return
            
        message = sanitize_input(message)
        
        # Voor debuggen
        print(f"Originele input: {message}")
        
        # Bericht formatteren volgens Gemma's verwachte structuur
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Je bent een behulpzame assistent die duidelijke, informatieve antwoorden geeft."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": message}]
            }
        ]
        
        # Laat tokenizer de juiste chat template toepassen
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Model input voorbereiden
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
        
        # Rest van je code blijft hetzelfde
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_length": 200,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        accumulated_text = ""
        
        for text in streamer:
            accumulated_text += text
            yield f"data: {json.dumps({'text': text})}\n\n"
            
        yield f"data: {json.dumps({'text': accumulated_text, 'done': True})}\n\n"
            
    except Exception as e:
        print(f"Fout: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/chat")
async def chat(request: ChatRequest, client_request: Request):
    # Gebruik client IP voor rate limiting
    client_ip = client_request.client.host
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Te veel verzoeken. Probeer het later opnieuw."
        )
    
    return StreamingResponse(
        generate_streaming_response(request.message, client_ip),
        media_type="text/event-stream"
    )

@app.post("/test_input")
async def test_input(request: ChatRequest, client_request: Request):
    # Gebruik client IP voor rate limiting
    client_ip = client_request.client.host
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Te veel verzoeken. Probeer het later opnieuw."
        )
    
    message = sanitize_input(request.message)
    
    # Bereid een chat voor met het juiste Gemma-formaat
    messages = [
        {"role": "user", "content": message}
    ]
    
    # Gebruik het ingebouwde chat template van de tokenizer
    chat_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize de chat-geformatteerde tekst
    tokenized = tokenizer(chat_formatted, return_tensors="pt", padding=True)
    
    result = {
        "original_message": message,
        "chat_formatted": chat_formatted,
        "token_count": len(tokenized["input_ids"][0]),
        "tokens": tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)