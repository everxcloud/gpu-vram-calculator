"""LLM inference memory estimate: weights + KV cache + runtime overhead, in decimal GB.

Same formulas as the EverXCloud LLM VRAM calculator:
https://everxcloud.ai/en/tools/llm-vram-calculator?utm_source=github&utm_medium=repository&utm_campaign=vram-calculator
"""

PRECISION_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}

# Typical architecture for each size: (billions of parameters, layers, KV dimension per layer).
# KV dimension = number of key/value heads x size of one head.
PRESETS = {
    "7B": (7, 32, 4096),
    "8B": (8, 32, 1024),
    "13B": (13, 40, 5120),
    "34B": (34, 48, 1024),
    "70B": (70, 80, 1024),
    "405B": (405, 126, 1024),
}

KV_CACHE_BYTES = 2  # keys and values stored in BF16 / FP16
GPU_COUNTS = (1, 2, 4, 8)

# GPU name and memory per GPU (GB), from the EverXCloud GPU catalog.
GPUS = [
    ('NVIDIA RTX A2000 6GB', 6),
    ('NVIDIA RTX A2000 12GB', 12),
    ('NVIDIA A16 16GB', 16),
    ('NVIDIA A2 16GB', 16),
    ('NVIDIA RTX 2000 Ada 16GB', 16),
    ('NVIDIA RTX A4000 16GB', 16),
    ('NVIDIA RTX PRO 2000 Blackwell 16GB', 16),
    ('NVIDIA Tesla P100 16GB', 16),
    ('NVIDIA Tesla T4 16GB', 16),
    ('NVIDIA Tesla V100 16GB', 16),
    ('NVIDIA RTX 4000 Ada 20GB', 20),
    ('NVIDIA RTX 4000 SFF Ada 20GB', 20),
    ('NVIDIA RTX A4500 20GB', 20),
    ('NVIDIA A10 24GB', 24),
    ('NVIDIA A30 24GB', 24),
    ('NVIDIA GeForce RTX 3090 24GB', 24),
    ('NVIDIA GeForce RTX 3090 Ti 24GB', 24),
    ('NVIDIA GeForce RTX 4090 24GB', 24),
    ('NVIDIA GeForce RTX 4090 D 24GB', 24),
    ('NVIDIA L4 24GB', 24),
    ('NVIDIA Quadro RTX 6000 24GB', 24),
    ('NVIDIA RTX 4500 Ada 24GB', 24),
    ('NVIDIA RTX A5000 24GB', 24),
    ('NVIDIA RTX A5500 24GB', 24),
    ('NVIDIA RTX PRO 4000 Blackwell 24GB', 24),
    ('NVIDIA RTX PRO 4000 Blackwell SFF 24GB', 24),
    ('NVIDIA Tesla P40 24GB', 24),
    ('NVIDIA TITAN RTX 24GB', 24),
    ('NVIDIA GeForce RTX 5090 32GB', 32),
    ('NVIDIA RTX 5000 Ada 32GB', 32),
    ('NVIDIA RTX PRO 4500 Blackwell 32GB', 32),
    ('NVIDIA Tesla V100 32GB', 32),
    ('NVIDIA A100 PCIe 40GB', 40),
    ('NVIDIA A100 SXM 40GB', 40),
    ('NVIDIA A40 48GB', 48),
    ('NVIDIA L40 48GB', 48),
    ('NVIDIA L40S 48GB', 48),
    ('NVIDIA Quadro RTX 8000 48GB', 48),
    ('NVIDIA RTX 5880 Ada 48GB', 48),
    ('NVIDIA RTX 6000 Ada 48GB', 48),
    ('NVIDIA RTX A6000 48GB', 48),
    ('NVIDIA RTX PRO 5000 Blackwell 48GB', 48),
    ('NVIDIA RTX PRO 5000 Blackwell 72GB', 72),
    ('NVIDIA A100 PCIe 80GB', 80),
    ('NVIDIA A100 SXM 80GB', 80),
    ('NVIDIA A800 PCIe 80GB', 80),
    ('NVIDIA H100 PCIe 80GB', 80),
    ('NVIDIA H100 SXM 80GB', 80),
    ('NVIDIA H100 NVL 94GB', 94),
    ('Google TPU v5p', 95),
    ('NVIDIA GH200', 96),
    ('NVIDIA RTX PRO 6000 Blackwell 96GB', 96),
    ('NVIDIA RTX PRO 6000 Blackwell Max-Q 96GB', 96),
    ('NVIDIA H200 141GB', 141),
    ('NVIDIA H200 NVL 141GB', 141),
    ('NVIDIA B200 180GB', 180),
    ('NVIDIA GB200 NVL', 186),
    ('AMD Instinct MI300X 192GB', 192),
    ('AMD Instinct MI325X 256GB', 256),
    ('NVIDIA B300 288GB', 288),
    ('NVIDIA GB300 NVL', 288),
]


def weight_gb(billions, bytes_per_parameter):
    """Weights: billions of parameters x bytes per parameter = decimal GB."""
    return billions * bytes_per_parameter


def kv_cache_gb(layers, kv_dim, context_tokens, batch, bytes_per_value=KV_CACHE_BYTES):
    """KV cache: 2 (keys and values) x layers x KV dimension x context tokens x batch x bytes."""
    return 2 * layers * kv_dim * context_tokens * batch * bytes_per_value / 1e9


def estimate_vram(billions, precision, layers, kv_dim, context_tokens=8192, batch=1, overhead_percent=10.0):
    """Return weights, KV cache, overhead and total in decimal GB."""
    if billions <= 0 or layers < 1 or kv_dim < 1 or context_tokens < 0 or batch < 1:
        raise ValueError("invalid model size, layers, KV dimension, context or batch")
    if not 0 <= overhead_percent <= 100:
        raise ValueError("overhead must be between 0 and 100 percent")
    weights = weight_gb(billions, PRECISION_BYTES[precision])
    kv = kv_cache_gb(layers, kv_dim, context_tokens, batch)
    overhead = (weights + kv) * overhead_percent / 100
    return {"weights_gb": weights, "kv_cache_gb": kv, "overhead_gb": overhead, "total_gb": weights + kv + overhead}


def gpus_that_fit(total_gb, gpus=GPUS):
    """GPUs whose 1, 2, 4 or 8 cards hold total_gb, fewest cards first, then smallest memory."""
    fits = []
    for name, vram in gpus:
        count = next((n for n in GPU_COUNTS if n * vram >= total_gb), None)
        if count is not None:
            fits.append((name, vram, count))
    return sorted(fits, key=lambda item: (item[2], item[1], item[0]))
