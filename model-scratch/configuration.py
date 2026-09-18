from typing import Optional, Union

from transformers import AutoConfig, AutoModel  # noqa: F401
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: F401

try:
    from transformers import PreTrainedConfig  # noqa: F401
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig  # noqa: F401

try:
    from transformers.configuration_utils import layer_type_validation
except ImportError:
    layer_type_validation = None

try:
    from transformers.modeling_rope_utils import RopeParameters
except ImportError:
    RopeParameters = None

try:
    from transformers.modeling_rope_utils import rope_config_validation
except ImportError:
    rope_config_validation = None


class ZaryaConfig(Qwen3Config):
    """Configuration class for Zarya model."""

    model_type = "zarya"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    backbone_class = "Qwen3ForCausalLM"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 22016
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = 12
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: int = 28
    layer_types: Optional[list[str]] = None
    attention_dropout: Union[float, int] = 0.0
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, list[int]]] = None
    dropout: float = 0.1
    alpha_0: float = 0.25
    noise_eps: float = 1e-3
    diffusion_loss_proportion: float = 0.5
    sequential_attn_mode: str = "mixed"
    diffusion_attn_mode: str = "mixed"
    sequential_shuffle: bool = False
    diffusion_shuffle: bool = False
    sampling_eps: float = 1e-3
    time_conditioning: bool = False
    norm_elementwise_affine: bool = True
    norm_eps: float = 1e-6
    T: int = 0
    slotted_training: bool = True
    ordered_sampling: bool = False
    noise_sorting: bool = True
    scale_by_batch: bool = False
    unnormalized_loss: bool = False
    simple_masking: bool = False
    extra_processing: bool = False
    sample_t_override: float = 0.0
    sample_t_upper: float = 1.0
    add_loss_path: bool = False
    grouped_noise: bool = False
    max_span_length: int = 50
    if RopeParameters is not None:
        rope_parameters: Optional[Union[RopeParameters, dict]] = None
    else:
        rope_theta: Optional[float] = 10000.0
        rope_scaling: Optional[dict] = None

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)

    def update_from_string(self, update_str: str):
        super().update_from_string(update_str)
        if self.layer_types is not None and len(self.layer_types) != self.num_hidden_layers:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


ZaryaConfig.register_for_auto_class("AutoConfig")
AutoConfig.register(ZaryaConfig.model_type, ZaryaConfig)
__all__ = ["ZaryaConfig"]
