# coding=utf-8
"""
Complexity model for vLLM.

A decoder-only transformer with INL (Inertial Navigation Layer) dynamics
for numerical stability and smooth token generation.

Usage with vLLM:
    from vllm import LLM
    llm = LLM(model="Complexity-ML/pacific-prime")
    output = llm.generate("Hello, world!")

Key innovations:
- INL Dynamics: PID-like control with velocity tracking
- Token-Routed MLP: Deterministic expert routing (token_id % num_experts)
- Mu-Guided Attention: Top-down influence from previous layer's equilibrium
"""

from .configuration_complexity import ComplexityConfig
from .modeling_complexity import (
    ComplexityForCausalLM,
    ComplexityModel,
    ComplexityDecoderLayer,
    ComplexityAttention,
    ComplexityMLP,
    INLDynamics,
    soft_clamp,
    mu_clamp,
)

# TokenRoutedMLP is now a vLLM layer (PR #34559)
from vllm.model_executor.layers.token_routed_i64 import TokenRoutedMLP

__all__ = [
    "ComplexityConfig",
    "ComplexityForCausalLM",
    "ComplexityModel",
    "ComplexityDecoderLayer",
    "ComplexityAttention",
    "ComplexityMLP",
    "TokenRoutedMLP",
    "INLDynamics",
    "soft_clamp",
    "mu_clamp",
]

__version__ = "0.1.0"
