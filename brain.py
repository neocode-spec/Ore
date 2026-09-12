```python
import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ============================================================
# ORE AI PRODUCTION BRAIN
# ============================================================
#
# Production model:
#   ore_model.onnx
#
# Model:
#   GPT-2 Small
#   INT4 transformer weights
#   Packed INT4 token embedding
#
# Runtime:
#   ONNX Runtime
#   CPUExecutionProvider
#
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ore_model.onnx"
)

TOKENIZER_DIR = BASE_DIR


# ============================================================
# STARTUP
# ============================================================

print("====================================")
print("🧠 STARTING ORE AI")
print("====================================")

print("📁 Base directory:")
print(BASE_DIR)

print("📦 Model:")
print(MODEL_PATH)


# ============================================================
# MODEL CHECK
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise RuntimeError(
        "Ore production model not found.\n"
        f"Expected:\n{MODEL_PATH}\n\n"
        "Make sure ore_model.onnx is in the same "
        "directory as brain.py."
    )


model_size_mb = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)

print(
    f"✅ Ore production model found "
    f"({model_size_mb:.2f} MB)"
)


# ============================================================
# TOKENIZER
# ============================================================

print("\n🔤 Loading tokenizer...")

try:

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True
    )

except Exception as exc:

    raise RuntimeError(
        "Ore tokenizer could not be loaded.\n"
        "Make sure the GPT-2 tokenizer files are "
        "inside the backend directory.\n\n"
        f"Original error: {exc}"
    )


# GPT-2 normally has no pad token.
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print("✅ Tokenizer loaded")


# ============================================================
# ONNX RUNTIME SETTINGS
# ============================================================

session_options = ort.SessionOptions()

# Render has limited RAM.
session_options.enable_cpu_mem_arena = False
session_options.enable_mem_pattern = False

# Keep logging quiet.
session_options.log_severity_level = 3


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🧠 Loading Ore ONNX model...")

try:

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ]
    )

except Exception as exc:

    raise RuntimeError(
        "Ore ONNX model failed to load.\n\n"
        f"Original error: {exc}"
    )


print("✅ Ore ONNX model loaded")
print("⚙️ Providers:", session.get_providers())


# ============================================================
# INSPECT MODEL INPUTS
# ============================================================

MODEL_INPUTS = {
    item.name: item
    for item in session.get_inputs()
}

MODEL_OUTPUTS = {
    item.name: item
    for item in session.get_outputs()
}


print("\n🔎 Model inputs:")

for name in MODEL_INPUTS:

    print("   •", name)


print("\n🔎 Model outputs:")

for name in MODEL_OUTPUTS:

    print("   •", name)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64

EOS_TOKEN_ID = tokenizer.eos_token_id

# Keep generation deliberately small for Render.
MAX_NEW_TOKENS = 32


# ============================================================
# VERIFY REQUIRED MODEL INPUTS
# ============================================================

required_inputs = [
    "input_ids",
    "attention_mask",
    "position_ids"
]

for name in required_inputs:

    if name not in MODEL_INPUTS:

        raise RuntimeError(
            f"Ore model is missing required input: {name}"
        )


for layer in range(NUM_LAYERS):

    key_name = (
        f"past_key_values.{layer}.key"
    )

    value_name = (
        f"past_key_values.{layer}.value"
    )

    if key_name not in MODEL_INPUTS:

        raise RuntimeError(
            f"Ore model is missing: {key_name}"
        )

    if value_name not in MODEL_INPUTS:

        raise RuntimeError(
            f"Ore model is missing: {value_name}"
        )


print("\n✅ Model input structure verified")


# ============================================================
# BUILD EMPTY KV CACHE
# ============================================================

def create_empty_past(batch_size=1):

    past = {}

    for layer in range(NUM_LAYERS):

        past[
            f"past_key_values.{layer}.key"
        ] = np.zeros(
            (
                batch_size,
                NUM_HEADS,
                0,
                HEAD_DIM
            ),
            dtype=np.float32
        )

        past[
            f"past_key_values.{layer}.value"
        ] = np.zeros(
            (
                batch_size,
                NUM_HEADS,
                0,
                HEAD_DIM
            ),
            dtype=np.float32
        )

    return past


# ============================================================
# UPDATE KV CACHE
# ============================================================

def extract_new_past(outputs):

    """
    Extract the 12 key/value cache tensors from
    the ONNX model outputs.

    The production ONNX graph returns:

        logits
        present.0.key
        present.0.value
        ...
        present.11.key
        present.11.value

    We use output names when available instead of relying
    entirely on output ordering.
    """

    new_past = {}

    # --------------------------------------------------------
    # First try output names
    # --------------------------------------------------------

    output_name_to_value = {
        output.name: outputs[index]
        for index, output in enumerate(
            session.get_outputs()
        )
    }

    for layer in range(NUM_LAYERS):

        key_name = (
            f"past_key_values.{layer}.key"
        )

        value_name = (
            f"past_key_values.{layer}.value"
        )

        possible_key_names = [
            key_name,
            f"present.{layer}.key",
            f"present.{layer}.key_output",
            f"present_key_values.{layer}.key"
        ]

        possible_value_names = [
            value_name,
            f"present.{layer}.value",
            f"present.{layer}.value_output",
            f"present_key_values.{layer}.value"
        ]

        key_tensor = None
        value_tensor = None

        for name in possible_key_names:

            if name in output_name_to_value:

                key_tensor = output_name_to_value[name]
                break

        for name in possible_value_names:

            if name in output_name_to_value:

                value_tensor = output_name_to_value[name]
                break

        # ----------------------------------------------------
        # If names aren't usable, use standard GPT-2 order.
        # ----------------------------------------------------

        if key_tensor is None:

            key_index = 1 + layer * 2

            if key_index < len(outputs):

                key_tensor = outputs[key_index]

        if value_tensor is None:

            value_index = 2 + layer * 2

            if value_index < len(outputs):

                value_tensor = outputs[value_index]

        if key_tensor is None:

            raise RuntimeError(
                f"Could not find KV key output "
                f"for layer {layer}"
            )

        if value_tensor is None:

            raise RuntimeError(
                f"Could not find KV value output "
                f"for layer {layer}"
            )

        new_past[key_name] = key_tensor
        new_past[value_name] = value_tensor

    return new_past


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    user_message: str,
    language: str = "pidgin"
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if user_message is None:

        user_message = ""

    user_message = str(
        user_message
    ).strip()

    if not user_message:

        return (
            "Abeg tell me wetin you wan know."
            if language.lower() == "pidgin"
            else
            "Please tell me what you would like to know."
        )


    # --------------------------------------------------------
    # Language instruction
    # --------------------------------------------------------

    if language.lower() == "pidgin":

        language_instruction = (
            "Respond naturally in Nigerian Pidgin English. "
            "Use clear, natural Nigerian Pidgin."
        )

    else:

        language_instruction = (
            "Respond in clear Standard English."
        )


    # --------------------------------------------------------
    # Ore prompt
    # --------------------------------------------------------

    prompt = f"""System: You are Ore, a Nigerian AI assistant.

You understand Nigerian history, geography, culture, languages,
education, technology, business, government and everyday Nigerian life.

{language_instruction}

User: {user_message}
Ore:"""


    # ========================================================
    # TOKENIZATION
    # ========================================================

    tokens = tokenizer(
        prompt,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    input_ids = tokens[
        "input_ids"
    ].astype(np.int64)


    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]


    # ========================================================
    # INITIAL KV CACHE
    # ========================================================

    past = create_empty_past(
        batch_size=batch_size
    )


    # ========================================================
    # GENERATION STATE
    # ========================================================

    generated_ids = []

    start_time = time.time()


    # ========================================================
    # AUTOREGRESSIVE GENERATION
    # ========================================================

    for step in range(
        MAX_NEW_TOKENS
    ):

        # ----------------------------------------------------
        # Current sequence length
        # ----------------------------------------------------

        current_seq_len = (
            input_ids.shape[1]
        )


        # ----------------------------------------------------
        # Existing KV cache length
        # ----------------------------------------------------

        past_len = past[
            "past_key_values.0.key"
        ].shape[2]


        # ----------------------------------------------------
        # Attention mask
        #
        # Must cover:
        #
        #   previous cached tokens
        #   +
        #   current tokens
        # ----------------------------------------------------

        attention_mask = np.ones(
            (
                batch_size,
                past_len + current_seq_len
            ),
            dtype=np.int64
        )


        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        position_ids = np.arange(
            past_len,
            past_len + current_seq_len,
            dtype=np.int64
        ).reshape(
            1,
            -1
        )


        # ----------------------------------------------------
        # ONNX INPUTS
        # ----------------------------------------------------

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **past
        }


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        outputs = session.run(
            None,
            ort_inputs
        )


        logits = outputs[0]


        # ----------------------------------------------------
        # GREEDY DECODING
        # ----------------------------------------------------

        next_token_id = int(
            np.argmax(
                logits[
                    0,
                    -1,
                    :
                ]
            )
        )


        generated_ids.append(
            next_token_id
        )


        # ----------------------------------------------------
        # STOP CONDITIONS
        # ----------------------------------------------------

        if (
            EOS_TOKEN_ID is not None
            and next_token_id == EOS_TOKEN_ID
        ):

            break


        # ----------------------------------------------------
        # UPDATE KV CACHE
        # ----------------------------------------------------

        past = extract_new_past(
            outputs
        )


        # ----------------------------------------------------
        # NEXT STEP
        #
        # After the first pass we only feed the newly
        # generated token.
        # ----------------------------------------------------

        input_ids = np.array(
            [[next_token_id]],
            dtype=np.int64
        )


    # ========================================================
    # DECODE
    # ========================================================

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


    # ========================================================
    # CLEAN RESPONSE
    # ========================================================

    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nOre:",
        "User:",
        "System:"
    ]


    for marker in stop_markers:

        if marker in response:

            response = response.split(
                marker,
                1
            )[0].strip()


    # ========================================================
    # FALLBACK
    # ========================================================

    if not response:

        if language.lower() == "pidgin":

            response = (
                "I no get response for that one yet."
            )

        else:

            response = (
                "I don't have a response for that yet."
            )


    # ========================================================
    # PERFORMANCE LOG
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    token_count = len(
        generated_ids
    )

    tokens_per_second = (
        token_count / elapsed
        if elapsed > 0
        else 0
    )


    print(
        f"🧠 Ore generated "
        f"{token_count} tokens "
        f"in {elapsed:.2f}s "
        f"({tokens_per_second:.2f} tok/s)"
    )


    return response
```

### One important correction to your folder structure

Your **old** code expected:

```text
ore_model/
└── model_int4.onnx
```

The **new** code expects:

```text
backend/
├── brain.py
├── main.py
├── ore_model.onnx
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
├── merges.txt
└── ...
```

That matches the production model we just tested.

### Before Render, run one local/Colab backend test

Because the frontend previously showed **"I couldn't connect to the Ore backend yet"**, we should test the actual `generate_response()` function before blaming Render for humanity's sins.

Run:

`python
from brain import generate_response

response = generate_response(
    "How many states are in Nigeria?",
    "pidgin"
)

print("\nORE:")
print(response)


The important thing is that it **doesn't crash while generating multiple tokens**. Our earlier test only proved one inference pass. This test proves the KV-cache loop works repeatedly.

If that passes, then the backend code is ready to connect to your FastAPI `/api/chat` endpoint and deploy.
