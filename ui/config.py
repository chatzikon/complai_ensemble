DEFAULT_PROVIDER = "malware"
DEFAULT_MODEL = "baseline_gcn"
DEFAULT_TASK = "malware_cfg_eval"

LOCAL_PROVIDERS = [
    "malware",
    "multimodal",
    "hf",
    "vllm",
    "sglang",
    "ollama",
    "llama-cpp-python",
    "transformerlens",
]

PROVIDER_DEFAULT_MODELS = {
    "malware": "baseline_gcn",
    "multimodal": "CMAlign",
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