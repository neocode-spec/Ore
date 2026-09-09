from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from brain import generate_response


# ============================================================
# ORE AI CONFIG
# ============================================================

APP_VERSION = "1.0"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Ore AI",
    version=APP_VERSION
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST FORMAT
# ============================================================

class ChatRequest(BaseModel):
    message: str
    language: str = "pidgin"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "Ore",
        "version": APP_VERSION
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

        print("ORE ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Ore encountered an internal error."
        )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
