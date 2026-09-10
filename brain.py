import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/content/drive/MyDrive/ore_model_onnx/model_int8.onnx"
TOKENIZER_PATH = "/content/drive/MyDrive/ore_model_onnx"

print("🧠 Loading Ore INT8 ONNX model...")
print("📦 Model:", MODEL_PATH)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# ONNX RUNTIME
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("✅ Ore INT8 model loaded")
print("⚙️ Provider:", session.get_providers())


# ============================================================
# SETTINGS
# ============================================================

MAX_NEW_TOKENS = 80


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    user_message: str,
    language: str = "pidgin"
):

    if language.lower() == "pidgin":

        language_instruction = (
            "Respond naturally in Nigerian Pidgin English."
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


    tokens = tokenizer(
        prompt,
        return_tensors="np"
    )

    input_ids = tokens["input_ids"].astype(np.int64)


    # ========================================================
    # EMPTY KV CACHE
    # ========================================================

    past = {}

    for i in range(12):

        past[f"past_key_values.{i}.key"] = np.zeros(
            (1, 12, 0, 64),
            dtype=np.float32
        )

        past[f"past_key_values.{i}.value"] = np.zeros(
            (1, 12, 0, 64),
            dtype=np.float32
        )


    generated_ids = []

    start_time = time.time()


    # ========================================================
    # AUTOREGRESSIVE GENERATION
    # ========================================================

    for step in range(MAX_NEW_TOKENS):

        seq_len = input_ids.shape[1]

        if step == 0:
            past_len = 0
        else:
            past_len = past[
                "past_key_values.0.key"
            ].shape[2]


        attention_mask = np.ones(
            (1, past_len + seq_len),
            dtype=np.int64
        )


        position_ids = np.arange(
            past_len,
            past_len + seq_len,
            dtype=np.int64
        ).reshape(1, -1)


        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past
        }


        outputs = session.run(
            None,
            ort_inputs
        )


        logits = outputs[0]


        next_token = int(
            np.argmax(
                logits[0, -1, :]
            )
        )


        generated_ids.append(next_token)


        if next_token == tokenizer.eos_token_id:
            break


        # ====================================================
        # UPDATE KV CACHE
        # ====================================================

        new_past = {}

        for i in range(12):

            new_past[
                f"past_key_values.{i}.key"
            ] = outputs[1 + i * 2]

            new_past[
                f"past_key_values.{i}.value"
            ] = outputs[2 + i * 2]


        past = new_past


        input_ids = np.array(
            [[next_token]],
            dtype=np.int64
        )


    # ========================================================
    # DECODE
    # ========================================================

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nOre:"
    ]

    for marker in stop_markers:

        if marker in response:

            response = response.split(
                marker
            )[0].strip()


    if not response:

        response = "I no get response for that one yet."


    elapsed = time.time() - start_time

    print(
        f"🧠 Generated {len(generated_ids)} tokens "
        f"in {elapsed:.2f}s"
    )


    return response
