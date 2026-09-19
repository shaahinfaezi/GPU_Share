"""Single-GPU KServe V1 predictor backed by DeepSpeed ZeRO stage 3.

The implementation intentionally keeps a single request in flight. ZeRO's
parameter-fetch hooks and Hugging Face generation are stateful at this level;
adding workers or concurrent generation calls would load another model or make
the single engine unsafe.
"""

import argparse
import asyncio
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import deepspeed
import kserve
import torch
from kserve import Model, ModelServer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import HfDeepSpeedConfig


LOG = logging.getLogger("zero-qwen")


def configure_single_rank() -> None:
    """Supply the rank variables DeepSpeed needs for a one-GPU pod."""
    defaults = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not deepspeed.comm.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")


class ZeroQwenModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self._generation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zero-generate")
        self.load()

    def load(self) -> None:
        model_path = Path(os.environ.get("MODEL_PATH", "/models/qwen2.5-0.5b-instruct"))
        config_path = Path(os.environ.get("ZERO_CONFIG", "/app/zero-nvme.json"))
        nvme_path = Path(os.environ.get("ZERO_NVME_PATH", "/opt/zero-nvme"))

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the pod did not receive an NVIDIA GPU")
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Hugging Face model directory is incomplete: {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"DeepSpeed config is missing: {config_path}")

        nvme_path.mkdir(parents=True, exist_ok=True)
        with config_path.open(encoding="utf-8") as handle:
            ds_config = json.load(handle)
        ds_config["zero_optimization"]["offload_param"]["nvme_path"] = str(nvme_path)

        configure_single_rank()
        LOG.info("loading model=%s with ZeRO-3 NVMe offload=%s", model_path, nvme_path)

        # Keep this object alive. Transformers consults it while from_pretrained
        # constructs the model and installs DeepSpeed-aware initialization hooks.
        self.hf_deepspeed_config = HfDeepSpeedConfig(ds_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        ).eval()

        self.engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
        self.model = self.engine.module.eval()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.ready = True
        LOG.info("model ready; ZeRO-3 parameter offload is configured for %s", nvme_path)

    @staticmethod
    def _parse_request(payload: Dict[str, Any]) -> Dict[str, Any]:
        instances = payload.get("instances")
        if not isinstance(instances, list) or len(instances) != 1:
            raise ValueError("payload must contain exactly one item in instances")
        instance = instances[0]
        if isinstance(instance, str):
            instance = {"prompt": instance}
        if not isinstance(instance, dict) or not isinstance(instance.get("prompt"), str):
            raise ValueError("instance must be a string or an object with a string prompt")

        max_new_tokens = int(instance.get("max_new_tokens", 64))
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256 for this pilot")
        temperature = float(instance.get("temperature", 0.0))
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        return {
            "prompt": instance["prompt"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

    def _predict_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = self._parse_request(payload)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": request["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = int(tokens.input_ids.shape[-1])

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": request["max_new_tokens"],
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if request["temperature"] > 0:
            generate_kwargs.update({"do_sample": True, "temperature": request["temperature"]})
        else:
            generate_kwargs["do_sample"] = False

        with self._generation_lock, torch.inference_mode():
            generated = self.model.generate(**tokens, **generate_kwargs)
            torch.cuda.synchronize(self.device)

        completion_ids = generated[0][input_length:]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return {
            "predictions": [
                {
                    "text": completion,
                    "input_tokens": input_length,
                    "output_tokens": int(completion_ids.shape[-1]),
                }
            ]
        }

    async def predict(self, payload: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict_sync, payload)


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        kserve.logging.configure_logging(args.log_config_file)
    ModelServer().start([ZeroQwenModel(args.model_name)])
