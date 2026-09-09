# brain.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# ORE MODEL CONFIGURATION
# ============================================================

MODEL_ID = os.getenv("ORE_MODEL_ID")

if not MODEL_ID:
    raise RuntimeError(
        "ORE_MODEL_ID environment variable is not set."
    )


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🧠 Loading Ore...")
print(f"📦 Model: {MODEL_ID}")
print(f"⚙️ Device: {DEVICE}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID
)

model.config.pad_token_id = tokenizer.pad_token_id

model = model.to(DEVICE)
model.eval()


print("✅ Ore loaded successfully")


# ============================================================
# GENERATION
# ============================================================

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50


def generate_response(
    user_message: str,
    language: str = "pidgin"
):

    if language.lower() == "pidgin":
        language_instruction = (
            "Respond naturally in Nigerian Pidgin English. "
            "Use clear, conversational Nigerian Pidgin."
        )
    else:
        language_instruction = (
            "Respond in clear Standard English."
        )

    prompt = f"""System: You are Ore, a Nigerian AI assistant.

You understand Nigerian history, geography, culture, languages,
education, technology, business, government and everyday Nigerian life.

{language_instruction}

User: {user_message}
Ore:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=True,

            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,

            repetition_penalty=1.1,
            no_repeat_ngram_size=3,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    # Prevent Ore from continuing the conversation itself
    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nOre:"
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0].strip()

    if not response:
        response = "I no get response for that one yet."

    return response
