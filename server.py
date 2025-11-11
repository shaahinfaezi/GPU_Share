from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml, time, os

app = Flask(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "distilgpt2")
print(f"Loading model: {MODEL_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

@app.route("/infer", methods=["POST"])
def infer():
    data = request.json or {}
    prompt = data.get("prompt", "Hello world")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=32)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency = (time.time() - t0) * 1000
    return jsonify({"text": text, "latency_ms": round(latency, 2)})

@app.route("/info", methods=["GET"])
def info():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    return jsonify({
        "total_mb": m.total // (1024*1024),
        "used_mb": m.used // (1024*1024),
        "free_mb": m.free // (1024*1024)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
