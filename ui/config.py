DEFAULT_PROVIDER = "ensemble"
DEFAULT_MODEL = "baseline_gcn"
DEFAULT_TASK = "malware_cfg_eval"

LOCAL_PROVIDERS = [
    "ensemble",
    "hf",
    "vllm",
    "sglang",
    "ollama",
    "llama-cpp-python",
    "transformerlens",
]

PROVIDER_MODEL_OPTIONS = {
    "ensemble": [
        "baseline_gcn",
        "CMAlign",
    ],
}

PROVIDER_DEFAULT_MODELS = {
    "ensemble": "baseline_gcn",

    "hf": "Qwen/Qwen3-8B",
    "hf/local": "",
    "vllm": "Qwen/Qwen3-8B",
    "vllm/local": "",
    "sglang": "Qwen/Qwen3-8B",
    "sglang/local": "",
    "ollama": "llama3.1",
    "llama-cpp-python": "local-model",
    "transformerlens": "gpt2",
}