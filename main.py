from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str

@app.get("/")
def home():
    return {"status": "Ore AI backend is running"}

@app.post("/api/chat")
def chat(req: ChatRequest):
    # Your model inference logic here
    return {"response": f"Ore response to: {req.message}"}
