from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Starting API server...")

# --- Step 1: Load the Fine-Tuned Model and Tokenizer ---
model_dir = "muskanrajput1104/support_agent_model"
print(f"Loading model and tokenizer from {model_dir}...")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded and running on device: {device}")

# --- Step 2: Define the FastAPI App ---
app = FastAPI(
    title="Agent's Co-pilot API",
    description="API for generating customer support replies."
)

# --- Step 3: Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"
    ],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Step 4: Define Request Data Model ---

class Query(BaseModel):
    prompt: str

# --- Step 5: Create the Prediction Endpoint ---
@app.post("/generate-suggestions")
def generate_suggestions(query: Query):
    print(f"Received query: {query.prompt}")
    
    
    inputs = tokenizer(query.prompt, return_tensors="pt").to(device)
    
    
    outputs = model.generate(
        **inputs,
        max_length=64,
        do_sample=True,          
        top_k=50,                
        top_p=0.95,              
        temperature=0.9,        
        num_return_sequences=3,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    suggestions = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    
    print(f"Generated suggestions: {suggestions}")
    
    return {"suggestions": suggestions}

# --- Step 6: Define a Root Endpoint (for testing) ---
@app.get("/")
def read_root():
    return {"status": "Agent's Co-pilot API is running."}
    # from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# print("Starting API server...")

# # --- Step 1: Load the Fine-Tuned Model and Tokenizer ---
# model_dir = "./support_agent_model"
# print(f"Loading model and tokenizer from {model_dir}...")

# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# print(f"Model loaded and running on device: {device}")

# # --- Step 2: Define the FastAPI App ---
# app = FastAPI(
#     title="Agent's Co-pilot API",
#     description="API for generating customer support replies."
# )

# # --- Step 3: Enable CORS ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"], 
# )

# # --- Step 4: Define Request Data Model ---

# class Query(BaseModel):
#     prompt: str

# # --- Step 5: Create the Prediction Endpoint ---
# @app.post("/generate-suggestions")
# def generate_suggestions(query: Query):
#     print(f"Received query: {query.prompt}")
    
    
#     inputs = tokenizer(query.prompt, return_tensors="pt").to(device)
    
    
#     outputs = model.generate(
#         **inputs,
#         max_length=64,
#         do_sample=True,          
#         top_k=50,                
#         top_p=0.95,              
#         temperature=0.9,        
#         num_return_sequences=3,
#         no_repeat_ngram_size=2,
#         early_stopping=True
#     )
    
#     suggestions = [
#         tokenizer.decode(output, skip_special_tokens=True) for output in outputs
#     ]
    
#     print(f"Generated suggestions: {suggestions}")
    
#     return {"suggestions": suggestions}

# # --- Step 6: Define a Root Endpoint (for testing) ---
# @app.get("/")
# def read_root():
#     return {"status": "Agent's Co-pilot API is running."}
