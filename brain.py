import os
import time
import shutil
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ============================================================
# ORE AI CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ore_model.onnx")

MODEL_PARTS = [
    os.path.join(BASE_DIR, "ore_model.onnx.part01"),
    os.path.join(BASE_DIR, "ore_model.onnx.part02"),
    os.path.join(BASE_DIR, "ore_model.onnx.part03"),
]

TOKENIZER_PATH = BASE_DIR

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 32

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64


# ============================================================
# RECONSTRUCT SPLIT ONNX MODEL
# ============================================================

def reconstruct_model():

    # If the complete model already exists, use it.
    if os.path.exists(MODEL_PATH):
        print("✅ Complete Ore model already exists")
        return

    print("\n====================================")
    print("RECONSTRUCTING ORE MODEL")
    print("====================================")

    # Check every part exists
    for part in MODEL_PARTS:

        if not os.path.exists(part):
            raise FileNotFoundError(
                f"Ore model part not found: {part}"
            )

        size_mb = os.path.getsize(part) / (1024 * 1024)

        print(
            f"Found {os.path.basename(part)} "
            f"({size_mb:.2f} MB)"
        )

    print("\nCombining model parts...")

    # Reconstruct exact original ONNX file
    with open(MODEL_PATH, "wb") as output:

        for part in MODEL_PARTS:

            print(
                f"Adding {os.path.basename(part)}..."
            )

            with open(part, "rb") as source:

                shutil.copyfileobj(
                    source,
                    output
                )

    print("\n✅ Ore model reconstructed")

    model_size_mb = (
        os.path.getsize(MODEL_PATH)
        / (1024 * 1024)
    )

    print(
        f"Reconstructed size: "
        f"{model_size_mb:.2f} MB"
    )


# ============================================================
# STARTUP
# ============================================================

print("====================================")
print("ORE AI BRAIN")
print("====================================")

print(f"Model: {MODEL_PATH}")

# Rebuild the 65.85 MB model from the 3 GitHub parts
reconstruct_model()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Ore model not found: {MODEL_PATH}"
    )

model_size_mb = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)

print(f"Size: {model_size_mb:.2f} MB")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)

print("Tokenizer loaded")


# ============================================================
# ONNX RUNTIME
# ============================================================

print("\nLoading ONNX model...")

session_options = ort.SessionOptions()

session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

print("ONNX model loaded")
print("Provider:", session.get_providers())


# ============================================================
# VERIFY MODEL CONTRACT
# ============================================================

actual_inputs = {
    inp.name
    for inp in session.get_inputs()
}

required_inputs = {
    "input_ids",
    "attention_mask",
    "position_ids"
}

for layer in range(NUM_LAYERS):

    required_inputs.add(
        f"past_key_values.{layer}.key"
    )

    required_inputs.add(
        f"past_key_values.{layer}.value"
    )

missing_inputs = (
    required_inputs - actual_inputs
)

if missing_inputs:

    raise RuntimeError(
        f"Missing model inputs: "
        f"{sorted(missing_inputs)}"
    )


actual_outputs = {
    output.name
    for output in session.get_outputs()
}

if "logits" not in actual_outputs:

    raise RuntimeError(
        "Model does not contain logits output."
    )

print("Model input contract verified")


print("\n====================================")
print("✅ ORE BRAIN READY")
print("====================================")


# ============================================================
# EMPTY KV CACHE
# ============================================================

def create_empty_past(batch_size=1):

    """
    The ONNX graph requires KV-cache inputs,
    even when starting with no cached tokens.

    We therefore provide zero-length cache tensors.

    IMPORTANT:
    These tensors are recreated on every generation step.
    We never feed the model's present.* outputs back in.
    """

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
# PROMPT FORMAT
# ============================================================

def format_prompt(message, language):

    message = str(message).strip()

    if not message:
        return ""

    language = str(language).lower().strip()

    if language == "pidgin":

        return (
            "Answer the following question in Nigerian Pidgin English.\n\n"
            f"Question: {message}\n"
            "Answer:"
        )

    return (
        "Answer the following question in Standard English.\n\n"
        f"Question: {message}\n"
        "Answer:"
    )


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(text):

    text = text.strip()

    stop_markers = [
        "\nQuestion:",
        "\nAnswer:",
        "\nUser:",
        "\nAssistant:"
    ]

    for marker in stop_markers:

        if marker in text:

            text = text.split(
                marker,
                1
            )[0]

    return text.strip()


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    message,
    language="pidgin"
):

    if message is None:
        return "Please enter a message."

    message = str(message).strip()

    if not message:
        return "Please enter a message."

    formatted_prompt = format_prompt(
        message,
        language
    )

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="np",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    input_ids = encoded[
        "input_ids"
    ].astype(np.int64)

    if input_ids.ndim != 2:

        raise RuntimeError(
            f"Unexpected input_ids shape: "
            f"{input_ids.shape}"
        )

    batch_size = input_ids.shape[0]

    if batch_size != 1:

        raise RuntimeError(
            "Ore generation currently "
            "expects batch_size=1."
        )

    generated_ids = []

    start_time = time.time()

    # --------------------------------------------------------
    # IMPORTANT:
    # Recreate an EMPTY cache every step.
    #
    # We intentionally do NOT use present.* outputs.
    # --------------------------------------------------------

    for step in range(MAX_NEW_TOKENS):

        sequence_length = input_ids.shape[1]

        attention_mask = np.ones(
            (
                batch_size,
                sequence_length
            ),
            dtype=np.int64
        )

        position_ids = np.arange(
            sequence_length,
            dtype=np.int64
        ).reshape(
            batch_size,
            sequence_length
        )

        # Fresh EMPTY cache
        empty_past = create_empty_past(
            batch_size
        )

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **empty_past
        }

        # Only request logits.
        # present.* outputs are deliberately ignored.
        outputs = session.run(
            ["logits"],
            ort_inputs
        )

        logits = outputs[0]

        if logits.ndim != 3:

            raise RuntimeError(
                f"Unexpected logits shape: "
                f"{logits.shape}"
            )

        # Greedy decoding
        next_token_id = int(
            np.argmax(
                logits[0, -1, :]
            )
        )

        generated_ids.append(
            next_token_id
        )

        # EOS
        if (
            tokenizer.eos_token_id is not None
            and next_token_id
            == tokenizer.eos_token_id
        ):
            break

        # Append generated token
        next_token = np.array(
            [[next_token_id]],
            dtype=np.int64
        )

        input_ids = np.concatenate(
            [
                input_ids,
                next_token
            ],
            axis=1
        )

        if (
            input_ids.shape[1]
            >= MAX_INPUT_TOKENS
        ):
            break

    elapsed = time.time() - start_time

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    response = clean_response(
        response
    )

    if not response:

        response = (
            "I couldn't generate a response."
        )

    print(
        f"\nOre generated "
        f"{len(generated_ids)} tokens "
        f"in {elapsed:.2f}s"
    )

    return response
