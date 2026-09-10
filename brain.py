import os
import time
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download


# ============================================================
# ORE MODEL CONFIGURATION
# ============================================================

HF_REPO = "Mur99/ore-int8"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "ore_model")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model_int8.onnx"
)


# ============================================================
# DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

print("🧠 Starting Ore...")
print("📦 Hugging Face repo:", HF_REPO)

if not os.path.exists(MODEL_PATH):

    print("⬇️ Ore INT8 model not found locally.")
    print("⬇️ Downloading from Hugging Face...")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        local_dir=MODEL_DIR,
        allow_patterns=[
            "model_int8.onnx",
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt"
        ]
    )

    print("✅ Ore model downloaded")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("🔤 Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("🧠 Loading INT8 ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("✅ Ore INT8 model loaded")
print("⚙️ Provider:", session.get_providers())


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS = 80


# ============================================================
# GENERATE ORE RESPONSE
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


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    tokens = tokenizer(
        prompt,
        return_tensors="np"
    )

    input_ids = tokens["input_ids"].astype(np.int64)


    # --------------------------------------------------------
    # EMPTY KV CACHE
    # --------------------------------------------------------

    past = {}

    for i in range(12):

        past[
            f"past_key_values.{i}.key"
        ] = np.zeros(
            (1, 12, 0, 64),
            dtype=np.float32
        )

        past[
            f"past_key_values.{i}.value"
        ] = np.zeros(
            (1, 12, 0, 64),
            dtype=np.float32
        )


    generated_ids = []

    start_time = time.time()


    # --------------------------------------------------------
    # AUTOREGRESSIVE GENERATION
    # --------------------------------------------------------

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


        # Greedy decoding
        next_token = int(
            np.argmax(
                logits[0, -1, :]
            )
        )


        generated_ids.append(next_token)


        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            break


        # ----------------------------------------------------
        # UPDATE KV CACHE
        # ----------------------------------------------------

        new_past = {}

        for i in range(12):

            new_past[
                f"past_key_values.{i}.key"
            ] = outputs[1 + i * 2]

            new_past[
                f"past_key_values.{i}.value"
            ] = outputs[2 + i * 2]


        past = new_past


        # Next iteration only needs the new token
        input_ids = np.array(
            [[next_token]],
            dtype=np.int64
        )


    # ========================================================
    # DECODE RESPONSE
    # ========================================================

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


    # ========================================================
    # STOP MARKERS
    # ========================================================

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


    # ========================================================
    # FALLBACK
    # ========================================================

    if not response:

        response = (
            "I no get response for that one yet."
        )


    elapsed = time.time() - start_time

    print(
        f"🧠 Generated {len(generated_ids)} tokens "
        f"in {elapsed:.2f}s"
    )


    return response
