from flask import Flask, request, jsonify
import os
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("Missing dependencies. Ensure 'torch' and 'transformers' are installed in the container.") from e


app = Flask(__name__)

# Model config
# The server now loads the model from a local path inside the container.
MODEL_PATH = "/app/model"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "48"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[server] Loading model from path={MODEL_PATH} on device={DEVICE}")
# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "Hello from GPU-Scheduler")
    max_new_tokens = int(data.get("max_new_tokens", MAX_NEW_TOKENS))

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    latency_ms = round((time.time() - t0) * 1000, 2)

    return jsonify({
        "text": text,
        "latency_ms": latency_ms,
        "model": MODEL_PATH, # Updated to show the path
        "device": str(DEVICE)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": str(DEVICE),
        "model": MODEL_PATH, # Updated to show the path
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
