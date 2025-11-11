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
MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "48"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[server] Loading model={MODEL_NAME} on device={DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
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
        "model": MODEL_NAME,
        "device": str(DEVICE)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": str(DEVICE),
        "model": MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
