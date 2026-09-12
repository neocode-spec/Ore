import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ============================================================
# ORE AI
# Production inference engine
# ============================================================
#
# Expected directory:
#
# backend/
# ├── brain.py
# ├── main.py
# ├── ore_model.onnx
# ├── tokenizer_config.json
# ├── tokenizer.json
# ├── vocab.json
# ├── merges.txt
# └── other tokenizer files...
#
# The production ONNX model is:
#
#     ore_model.onnx
#
# It contains:
#     - GPT-2 Small
#     - INT4 transformer weights
#     - packed INT4 token embedding
#
# Runtime:
#     ONNX Runtime
#     CPUExecutionProvider
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ore_model.onnx"
)

TOKENIZER_PATH = BASE_DIR


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 32


# ============================================================
# STARTUP
# ============================================================

print("====================================")
print("🧠 STARTING ORE AI")
print("====================================")

print("📁 Base directory:")
print(BASE_DIR)

print("📦 Model path:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    raise RuntimeError(
        "Ore production model was not found.\n\n"
        f"Expected model:\n{MODEL_PATH}\n\n"
        "Make sure ore_model.onnx is in the same "
        "directory as brain.py."
    )


MODEL_SIZE_MB = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)

print(
    f"✅ Production model found: "
    f"{MODEL_SIZE_MB:.2f} MB"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n🔤 Loading tokenizer...")

try:

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True
    )

except Exception as exc:

    raise RuntimeError(
        "Ore tokenizer could not be loaded.\n\n"
        "Make sure the tokenizer files are present "
        "in the same backend directory as brain.py.\n\n"
        f"Original error:\n{exc}"
    ) from exc


# GPT-2 does not normally have a separate PAD token.
# EOS is safe to use as PAD for inference.
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token


print("✅ Tokenizer loaded")

print(
    "   EOS token ID:",
    tokenizer.eos_token_id
)

print(
    "   PAD token ID:",
    tokenizer.pad_token_id
)


# ============================================================
# ONNX SESSION SETTINGS
# ============================================================

session_options = ort.SessionOptions()

# Reduce unnecessary memory overhead on small Render
# instances.
session_options.enable_cpu_mem_arena = False
session_options.enable_mem_pattern = False

# Only show serious ONNX Runtime messages.
session_options.log_severity_level = 3


# ============================================================
# LOAD ONNX MODEL
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
        f"Model:\n{MODEL_PATH}\n\n"
        f"Original error:\n{exc}"
    ) from exc


print("✅ Ore ONNX model loaded")

print(
    "⚙️ Providers:",
    session.get_providers()
)


# ============================================================
# MODEL INPUT / OUTPUT INFORMATION
# ============================================================

MODEL_INPUTS = {
    item.name: item
    for item in session.get_inputs()
}

MODEL_OUTPUTS = {
    item.name: item
    for item in session.get_outputs()
}


print("\n🔎 ONNX inputs:")

for name in MODEL_INPUTS:

    print(
        "   •",
        name
    )


print("\n🔎 ONNX outputs:")

for name in MODEL_OUTPUTS:

    print(
        "   •",
        name
    )


# ============================================================
# VERIFY REQUIRED INPUTS
# ============================================================

REQUIRED_INPUTS = [
    "input_ids",
    "attention_mask",
    "position_ids"
]


for input_name in REQUIRED_INPUTS:

    if input_name not in MODEL_INPUTS:

        raise RuntimeError(
            "Production Ore model is missing "
            f"required input: {input_name}"
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
            "Production Ore model is missing "
            f"required input: {key_name}"
        )

    if value_name not in MODEL_INPUTS:

        raise RuntimeError(
            "Production Ore model is missing "
            f"required input: {value_name}"
        )


print(
    "\n✅ Required ONNX inputs verified"
)


# ============================================================
# FIND KV OUTPUTS
# ============================================================
#
# The model inputs are:
#
# past_key_values.0.key
# past_key_values.0.value
# ...
#
# The outputs are normally:
#
# present.0.key
# present.0.value
# ...
#
# We detect the actual names instead of assuming that
# output ordering is always identical.
#
# ============================================================

OUTPUT_NAMES = [
    output.name
    for output in session.get_outputs()
]


def find_kv_output_name(
    layer: int,
    kind: str
):
    """
    Find the ONNX output corresponding to a
    layer's key/value cache.

    Supports common GPT-2 ONNX naming conventions.
    """

    candidates = [

        f"present.{layer}.{kind}",

        f"present.{layer}.{kind}_output",

        f"present_key_values.{layer}.{kind}",

        f"past_key_values.{layer}.{kind}",

        f"past_key_values.{layer}.{kind}_output",

    ]

    for candidate in candidates:

        if candidate in MODEL_OUTPUTS:

            return candidate


    # --------------------------------------------------------
    # Fallback:
    # search by layer number and key/value name.
    # --------------------------------------------------------

    layer_text = str(layer)

    kind_text = kind.lower()

    for name in OUTPUT_NAMES:

        lower_name = name.lower()

        if (
            layer_text in lower_name
            and kind_text in lower_name
            and (
                "present" in lower_name
                or "past" in lower_name
                or "key_values" in lower_name
            )
        ):

            return name


    return None


KV_OUTPUT_NAMES = {}


for layer in range(NUM_LAYERS):

    key_output = find_kv_output_name(
        layer,
        "key"
    )

    value_output = find_kv_output_name(
        layer,
        "value"
    )

    if key_output is not None:

        KV_OUTPUT_NAMES[
            f"past_key_values.{layer}.key"
        ] = key_output

    if value_output is not None:

        KV_OUTPUT_NAMES[
            f"past_key_values.{layer}.value"
        ] = value_output


# ============================================================
# KV OUTPUT VALIDATION
# ============================================================

if len(KV_OUTPUT_NAMES) != NUM_LAYERS * 2:

    print(
        "\n⚠️ Could not resolve all KV output names "
        "from their names."
    )

    print(
        "   Falling back to the verified GPT-2 "
        "output ordering."
    )

    KV_OUTPUT_NAMES = None

else:

    print(
        "\n✅ KV-cache output names verified"
    )


# ============================================================
# CREATE EMPTY KV CACHE
# ============================================================

def create_empty_past(
    batch_size: int
):
    """
    Create the initial empty GPT-2 KV cache.

    Shape:

        (batch, heads, sequence_length, head_dim)

    At the beginning:

        sequence_length = 0

    Therefore:

        (1, 12, 0, 64)
    """

    past = {}

    for layer in range(NUM_LAYERS):

        key_name = (
            f"past_key_values.{layer}.key"
        )

        value_name = (
            f"past_key_values.{layer}.value"
        )

        past[key_name] = np.zeros(
            (
                batch_size,
                NUM_HEADS,
                0,
                HEAD_DIM
            ),
            dtype=np.float32
        )

        past[value_name] = np.zeros(
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
# EXTRACT NEW KV CACHE
# ============================================================

def extract_new_past(
    outputs
):
    """
    Extract the newly generated KV cache from
    ONNX Runtime outputs.

    Output 0 is expected to be logits.

    If the model exposes named present.* outputs,
    those names are used.

    Otherwise, the verified GPT-2 ordering is used:

        output 0 = logits

        output 1 = layer 0 key
        output 2 = layer 0 value

        output 3 = layer 1 key
        output 4 = layer 1 value

        ...

        output 23 = layer 11 key
        output 24 = layer 11 value
    """

    new_past = {}

    # --------------------------------------------------------
    # Build name -> tensor mapping
    # --------------------------------------------------------

    output_name_to_tensor = {
        output.name: outputs[index]
        for index, output in enumerate(
            session.get_outputs()
        )
    }


    # --------------------------------------------------------
    # Named output route
    # --------------------------------------------------------

    if KV_OUTPUT_NAMES is not None:

        for layer in range(NUM_LAYERS):

            key_input_name = (
                f"past_key_values.{layer}.key"
            )

            value_input_name = (
                f"past_key_values.{layer}.value"
            )

            key_output_name = (
                KV_OUTPUT_NAMES[key_input_name]
            )

            value_output_name = (
                KV_OUTPUT_NAMES[value_input_name]
            )

            new_past[key_input_name] = (
                output_name_to_tensor[
                    key_output_name
                ]
            )

            new_past[value_input_name] = (
                output_name_to_tensor[
                    value_output_name
                ]
            )

        return new_past


    # --------------------------------------------------------
    # Verified positional output route
    # --------------------------------------------------------

    expected_output_count = (
        1 + NUM_LAYERS * 2
    )

    if len(outputs) < expected_output_count:

        raise RuntimeError(
            "Ore ONNX model returned fewer outputs "
            "than required for GPT-2 KV-cache generation.\n\n"
            f"Expected at least: {expected_output_count}\n"
            f"Received: {len(outputs)}"
        )


    for layer in range(NUM_LAYERS):

        key_index = (
            1 + layer * 2
        )

        value_index = (
            2 + layer * 2
        )

        key_name = (
            f"past_key_values.{layer}.key"
        )

        value_name = (
            f"past_key_values.{layer}.value"
        )

        new_past[key_name] = (
            outputs[key_index]
        )

        new_past[value_name] = (
            outputs[value_index]
        )


    return new_past


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_response(
    user_message: str,
    language: str = "pidgin"
):
    """
    Generate an Ore response.

    Parameters
    ----------
    user_message:
        User's message.

    language:
        "pidgin" for Nigerian Pidgin.
        Anything else uses Standard English.

    Returns
    -------
    str
        Generated Ore response.
    """

    # ========================================================
    # CLEAN INPUT
    # ========================================================

    if user_message is None:

        user_message = ""

    user_message = str(
        user_message
    ).strip()


    if not user_message:

        if language.lower() == "pidgin":

            return (
                "Abeg tell me wetin you wan know."
            )

        return (
            "Please tell me what you would like to know."
        )


    # ========================================================
    # LANGUAGE INSTRUCTION
    # ========================================================

    if language.lower() == "pidgin":

        language_instruction = (
            "Respond naturally in Nigerian Pidgin English. "
            "Use clear and natural Nigerian Pidgin."
        )

    else:

        language_instruction = (
            "Respond in clear Standard English."
        )


    # ========================================================
    # ORE PROMPT
    # ========================================================

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
        max_length=MAX_INPUT_TOKENS
    )


    input_ids = tokens[
        "input_ids"
    ].astype(
        np.int64
    )


    batch_size = (
        input_ids.shape[0]
    )


    # ========================================================
    # INITIAL EMPTY KV CACHE
    # ========================================================

    past = create_empty_past(
        batch_size
    )


    # ========================================================
    # GENERATED TOKEN STORAGE
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
        # Current number of tokens being sent to the model
        # ----------------------------------------------------

        current_seq_len = (
            input_ids.shape[1]
        )


        # ----------------------------------------------------
        # Number of tokens already stored in KV cache
        # ----------------------------------------------------

        past_len = int(
            past[
                "past_key_values.0.key"
            ].shape[2]
        )


        # ----------------------------------------------------
        # Attention mask
        #
        # The mask covers:
        #
        #   cached tokens
        #   +
        #   current input tokens
        #
        # Example:
        #
        # first pass:
        #   past_len = 0
        #   seq_len  = 8
        #   mask     = (1, 8)
        #
        # next pass:
        #   past_len = 8
        #   seq_len  = 1
        #   mask     = (1, 9)
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
        #
        # First pass:
        #
        #   0,1,2,3,...
        #
        # Later:
        #
        #   previous_cache_length
        #
        # ----------------------------------------------------

        position_ids = np.arange(
            past_len,
            past_len + current_seq_len,
            dtype=np.int64
        ).reshape(
            batch_size,
            current_seq_len
        )


        # ====================================================
        # BUILD ONNX INPUT
        # ====================================================

        ort_inputs = {

            "input_ids": input_ids,

            "attention_mask": attention_mask,

            "position_ids": position_ids,

            **past
        }


        # ====================================================
        # RUN MODEL
        # ====================================================

        try:

            outputs = session.run(
                None,
                ort_inputs
            )

        except Exception as exc:

            raise RuntimeError(
                "Ore inference failed during "
                f"generation step {step}.\n\n"
                f"Input token shape: "
                f"{input_ids.shape}\n"
                f"Past length: {past_len}\n"
                f"Current sequence length: "
                f"{current_seq_len}\n\n"
                f"Original error:\n{exc}"
            ) from exc


        # ====================================================
        # LOGITS
        # ====================================================

        logits = outputs[0]


        if logits.ndim != 3:

            raise RuntimeError(
                "Ore model returned an unexpected "
                f"logits shape: {logits.shape}"
            )


        # ====================================================
        # GREEDY DECODING
        # ====================================================
        #
        # Select the highest-probability token.
        #
        # This is deliberately simple for the first
        # production version.
        #
        # No sampling randomness.
        # No external API.
        # No hidden model.
        #
        # ====================================================

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


        # ====================================================
        # EOS CHECK
        # ====================================================

        if (
            tokenizer.eos_token_id is not None
            and next_token_id
            == tokenizer.eos_token_id
        ):

            break


        # ====================================================
        # UPDATE KV CACHE
        # ====================================================

        past = extract_new_past(
            outputs
        )


        # ====================================================
        # NEXT INPUT
        # ====================================================
        #
        # After the first pass, only the newly generated
        # token is sent into the model.
        #
        # The previous tokens remain inside the KV cache.
        #
        # ====================================================

        input_ids = np.array(
            [
                [next_token_id]
            ],
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
    # REMOVE ACCIDENTAL PROMPT MARKERS
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
    # PERFORMANCE
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )


    token_count = len(
        generated_ids
    )


    if elapsed > 0:

        tokens_per_second = (
            token_count
            / elapsed
        )

    else:

        tokens_per_second = 0.0


    print(
        f"🧠 Ore generated "
        f"{token_count} tokens "
        f"in {elapsed:.2f}s "
        f"({tokens_per_second:.2f} tok/s)"
    )


    # ========================================================
    # RETURN
    # ========================================================

    return response
```

### One correction I deliberately made

I **did not** use the earlier simplistic:

```python
new_past[f"..."] = outputs[1 + i * 2]
```

as the only mechanism.

Your actual model has already proven that it returns the expected logits shape, but production code should identify the KV outputs by their names when possible and only fall back to the verified ordering if necessary. That's the difference between "it worked once in Colab" and code you can reasonably hand to Render without crossing your fingers.

### Your backend must now look like this

```text
ore-backend/
│
├── main.py
├── brain.py
├── ore_model.onnx
│
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
│
└── requirements.txt
```

Most importantly:

```text
❌ ore_model/model_int4.onnx
❌ ore_model/model_4bit.onnx
❌ /content/drive/MyDrive/...
❌ model_int4.onnx

✅ ore_model.onnx
```

The new model is **65.85 MB and has already passed the actual ONNX Runtime inference test with the required `position_ids` and KV inputs**, so we leave that model alone and make the Python conform to it.

### Before pushing to GitHub

Run this exact backend test in the environment containing `brain.py`:

```python
from brain import generate_response

print("\n====================================")
print("ORE BACKEND GENERATION TEST")
print("====================================")

response = generate_response(
    "How many states are in Nigeria?",
    "pidgin"
)

print("\nORE RESPONSE:")
print(response)

print("\n====================================")
print("✅ BACKEND GENERATION TEST FINISHED")
print("====================================")
```

