import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# ORE CONFIG
# ============================================================

MODEL_PATH = "/content/ore_nigerian_history_model"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_P = 0.9


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("========================================")
print("          ORE AI STARTING")
print("========================================")
print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_PATH}")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Ore AI",
    version="1.0"
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
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD TRAINED ORE MODEL
# ============================================================

print("Loading trained Ore model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.config.pad_token_id = tokenizer.pad_token_id

model = model.to(DEVICE)

model.eval()

print("\n========================================")
print("          ORE AI IS READY")
print("========================================")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "status": "online",
        "name": "Ore",
        "device": DEVICE
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


    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    if req.language.lower() == "pidgin":

        language_instruction = (
            "Respond in natural Nigerian Pidgin English."
        )

    else:

        language_instruction = (
            "Respond in clear Standard English."
        )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""System: You are Ore, a Nigerian AI assistant.

You understand Nigerian history, geography, culture, languages,
education, technology, business, government and everyday Nigerian life.

{language_instruction}

User: {user_message}
Ore:"""


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,

            top_p=TOP_P,

            repetition_penalty=1.1,

            no_repeat_ngram_size=3,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    generated_text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    # --------------------------------------------------------
    # REMOVE PROMPT
    # --------------------------------------------------------

    if "Ore:" in generated_text:

        response = generated_text.split(
            "Ore:",
            1
        )[1].strip()

    else:

        response = generated_text.strip()


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not response:

        if req.language.lower() == "pidgin":

            response = "I no fit generate response for that one yet."

        else:

            response = "I couldn't generate a response to that."


    # --------------------------------------------------------
    # RETURN TO FRONTEND
    # --------------------------------------------------------

    return {
        "response": response
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
