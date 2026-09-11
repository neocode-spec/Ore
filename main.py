from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from brain import generate_response


# ============================================================
# ORE AI CONFIG
# ============================================================

APP_VERSION = "1.0"

app = FastAPI(
    title="Ore AI",
    version=APP_VERSION
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://ore-liart.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    language: str = "pidgin"


# ============================================================
# HOME / HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "Ore",
        "version": APP_VERSION
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Ore AI"
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(req: ChatRequest):

    user_message = req.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        response = generate_response(
            user_message,
            req.language
        )

        return {
            "response": response
        }

    except Exception as e:

        print("====================================")
        print("ORE ERROR")
        print(str(e))
        print("====================================")

        raise HTTPException(
            status_code=500,
            detail="Ore encountered an internal error."
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
