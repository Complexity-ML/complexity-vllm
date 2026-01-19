# Complexity Model for vLLM

vLLM-compatible implementation of the Complexity model with INL (Inertial Navigation Layer) dynamics.

## Model Overview

**Complexity** is a decoder-only transformer architecture featuring:

| Feature | Description |
|---------|-------------|
| **INL Dynamics** | PID-like control with velocity tracking for numerical stability |
| **Token-Routed MLP** | Deterministic expert routing (`token_id % num_experts`) |
| **Mu-Guided Attention** | Top-down influence from previous layer's equilibrium |
| **GQA** | Grouped Query Attention for efficient inference |
| **QK Norm** | Query-Key normalization for stable training |
| **RoPE** | Rotary Position Embeddings |

## Architecture

```
For each layer:
    1. Mu-Guided Attention (perception)
       └── Q, K biased by mu from previous layer
    2. INL Dynamics (control/stabilization)
       └── v_next = α*v - β*(h - μ)
       └── h_next = h + dt*gate*v_next
    3. Token-Routed MLP (transformation)
       └── expert_id = token_id % num_experts
```

## Model Variants

| Model | Params | Layers | Hidden | Heads | KV Heads | Experts |
|-------|--------|--------|--------|-------|----------|---------|
| complexity-tiny | ~15M | 6 | 256 | 4 | 2 | 4 |
| complexity-small | ~50M | 8 | 512 | 8 | 4 | 4 |
| complexity-base | ~125M | 12 | 768 | 12 | 4 | 4 |
| complexity-350m | ~350M | 20 | 1280 | 16 | 4 | 4 |
| complexity-1b | ~1B | 24 | 2048 | 16 | 8 | 4 |
| **pacific-prime** | **~1.5B** | **24** | **2048** | **16** | **8** | **4** |
| complexity-3b | ~3B | 32 | 2560 | 32 | 8 | 8 |
| complexity-7b | ~7B | 32 | 4096 | 32 | 8 | 8 |

## Files

```
complexity-vllm/
├── __init__.py                    # Module exports
├── configuration_complexity.py    # HuggingFace config class
├── modeling_complexity.py         # vLLM model implementation
└── README.md                      # This file
```

## Usage

### With vLLM (after integration)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Complexity-ML/pacific-prime")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Hello, world!"], params)
print(outputs[0].outputs[0].text)
```

### Standalone HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Complexity-ML/pacific-prime",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Complexity-ML/pacific-prime")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

# vLLM New Model Request

## Issue Title
`[Model Request] Add support for Complexity model (INL Dynamics)`

## Issue Body

### Model Information

**Model name:** Complexity (Pacific-Prime)

**Model links:**
- HuggingFace: https://huggingface.co/Pacific-Prime/pacific-prime
- GitHub: https://github.com/Complexity-ML/complexity-inference
- Paper: [Coming soon]

**Model architecture:** Decoder-only transformer (LLaMA-based with custom components)

**Model sizes available:** 15M, 50M, 125M, 350M, 1B, 1.5B, 3B, 7B

### Why this model should be added

1. **Novel architecture innovations:**
   - **INL Dynamics**: PID-like control system for numerical stability during inference
   - **Token-Routed MLP**: Deterministic expert routing without router overhead
   - **Mu-Guided Attention**: Top-down information flow between layers

2. **Production-ready:**
   - Trained model available on HuggingFace
   - OpenAI-compatible inference server: https://github.com/Complexity-ML/complexity-inference
   - 63 unit tests passing

3. **Community interest:**
   - Unique approach to transformer stability
   - Potential applications in robotics and control systems

### Technical Details

**Config parameters (beyond standard LLaMA):**
```json
{
  "model_type": "complexity",
  "use_token_routed_mlp": true,
  "num_experts": 4,
  "use_qk_norm": true,
  "dynamics_alpha": 0.9,
  "dynamics_beta": 0.1,
  "dynamics_gate": 0.5,
  "dynamics_dt": 0.1,
  "dynamics_controller_hidden": 64
}
```

**New components:**
1. `INLDynamics` - Velocity-tracked equilibrium-seeking layer
2. `TokenRoutedMLP` - Deterministic expert routing
3. Modified attention with mu-guidance

**Compatibility:**
- Uses standard vLLM attention backends
- PagedAttention compatible
- Tensor parallel compatible (standard column/row parallel)

### Implementation

I've prepared a vLLM-compatible implementation:
- `configuration_complexity.py` - HuggingFace PretrainedConfig
- `modeling_complexity.py` - vLLM model with all required interfaces

**Ready to submit PR** with:
- Full model implementation
- Weight loading support
- Tests

### Checklist

- [x] Model is publicly available on HuggingFace
- [x] Model has a clear use case
- [x] Implementation is ready
- [x] Willing to maintain compatibility

---

## Contact

**Author:** [Your Name]
**Email:** [Your Email]
**GitHub:** https://github.com/Complexity-ML
**Discord:** [Optional]

---

## License

MIT License - Same as vLLM
