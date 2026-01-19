# coding=utf-8
"""
Complexity model configuration for HuggingFace and vLLM.

This configuration supports:
- Standard LLaMA-like architecture
- INL Dynamics (alpha, beta, gate, mu, dt)
- Token-Routed MLP (deterministic expert routing)
- QK Normalization
- Grouped Query Attention (GQA)
"""

from transformers import PretrainedConfig


class ComplexityConfig(PretrainedConfig):
    """
    Configuration class for Complexity models.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for GQA.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base frequency for RoPE.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for attention.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie input/output embeddings.
        use_token_routed_mlp (`bool`, *optional*, defaults to True):
            Whether to use Token-Routed MLP (deterministic expert routing).
        num_experts (`int`, *optional*, defaults to 4):
            Number of experts for Token-Routed MLP.
        use_qk_norm (`bool`, *optional*, defaults to True):
            Whether to use QK normalization.
        use_sdpa (`bool`, *optional*, defaults to True):
            Whether to use Flash Attention via SDPA.
        sliding_window (`int`, *optional*, defaults to None):
            Sliding window size for attention (None = full attention).
        dynamics_alpha (`float`, *optional*, defaults to 0.9):
            INL inertia/momentum parameter.
        dynamics_beta (`float`, *optional*, defaults to 0.1):
            INL correction strength parameter.
        dynamics_gate (`float`, *optional*, defaults to 0.5):
            INL amplitude control parameter.
        dynamics_dt (`float`, *optional*, defaults to 0.1):
            INL integration timestep.
        dynamics_controller_hidden (`int`, *optional*, defaults to 64):
            Hidden size for INL controller MLP.
    """

    model_type = "complexity"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        pad_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 0,
        # Token-Routed MLP
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        # 2024 innovations
        use_qk_norm: bool = True,
        use_sdpa: bool = True,
        sliding_window: int = None,
        # INL Dynamics
        dynamics_alpha: float = 0.9,
        dynamics_beta: float = 0.1,
        dynamics_gate: float = 0.5,
        dynamics_dt: float = 0.1,
        dynamics_controller_hidden: int = 64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        # Token-Routed MLP
        self.use_token_routed_mlp = use_token_routed_mlp
        self.num_experts = num_experts

        # 2024 innovations
        self.use_qk_norm = use_qk_norm
        self.use_sdpa = use_sdpa
        self.sliding_window = sliding_window

        # INL Dynamics
        self.dynamics_alpha = dynamics_alpha
        self.dynamics_beta = dynamics_beta
        self.dynamics_gate = dynamics_gate
        self.dynamics_dt = dynamics_dt
        self.dynamics_controller_hidden = dynamics_controller_hidden

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


# Register with AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("complexity", ComplexityConfig)
except Exception:
    pass
