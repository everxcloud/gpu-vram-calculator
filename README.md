<p align="center">
  <a href="https://everxcloud.ai/?utm_source=github&utm_medium=readme&utm_campaign=brand">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/brand/everxcloud-logo-on-dark.png">
      <img src="assets/brand/everxcloud-logo-on-light.png" alt="EverXCloud: Every GPU. One provider." width="420">
    </picture>
  </a>
</p>

# GPU VRAM calculator for LLMs

Estimate how much GPU memory a large language model needs for inference, and which GPUs can hold it alone or in 2, 4 or 8 cards. One dependency-free Python file, MIT licensed.

## Example

```python
from vram import estimate_vram, gpus_that_fit, PRESETS

billions, layers, kv_dim = PRESETS["70B"]
result = estimate_vram(billions, "bf16", layers, kv_dim, context_tokens=8192, batch=1, overhead_percent=10)
print(round(result["total_gb"], 1))   # 157.0
print(gpus_that_fit(result["total_gb"])[:3])
```

## Formulas

- Weights (GB) = billions of parameters x bytes per parameter: 4 in FP32, 2 in BF16/FP16, 1 in FP8/INT8, 0.5 in INT4.
- KV cache (GB) = 2 x layers x KV dimension x context tokens x batch x 2 bytes / 10^9. The KV dimension is the number of key/value heads x the size of one head.
- Total (GB) = (weights + KV cache) x (1 + overhead %), the overhead covering activations and runtime buffers.

These are planning estimates: real usage depends on the architecture, the inference engine and quantization metadata.

## Tests

```bash
python -m pytest
```

## Online version

An interactive version with the full GPU catalog and live rental prices is available on EverXCloud: https://everxcloud.ai/en/tools/llm-vram-calculator?utm_source=github&utm_medium=repository&utm_campaign=vram-calculator

EverXCloud is a US spot exchange for renting cloud GPUs by the hour.
