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
model_name = "google/gemma-3-4b-it"
# Lees token vanuit bestand `token.txt`
with open("token.txt", "r") as f:
    access_token = f.read().strip()

tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token=access_token).to(device)

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
        
        # Laat de berichtstructuur zien voor debugging
        print(f"Berichtstructuur: {json.dumps(messages, indent=2)}")
        
        # Laat tokenizer de juiste chat template toepassen
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"Toegepast chat template: {input_text}")
        
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
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_streaming_response(request.message),
        media_type="text/event-stream"
    )

@app.post("/test_input")
async def test_input(request: ChatRequest):
    message = request.message
    
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