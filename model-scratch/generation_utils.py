from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ZaryaGenerationConfig(GenerationConfig):
    model_type = "zarya"
    ignore_noise_schedule: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ignore_noise_schedule: bool = kwargs.pop("ignore_noise_schedule", False)
        self.T: int = kwargs.pop("T", 1000)
        self.use_float64: bool = kwargs.pop("use_float64", False)
        self.sequential_phase_only: bool = kwargs.pop("sequential_phase_only", False)
        self.diffusion_phase_only: bool = kwargs.pop("diffusion_phase_only", False)
        self.unmask_probs_coef: float = kwargs.pop("unmask_probs_coef", 1)
        self.slot_size: int = kwargs.pop("slot_size", 16)
        self.serial_num_blocks: int = kwargs.pop("serial_num_blocks", 1)
        self.slot_threshold: float = kwargs.pop("slot_threshold", 0.9)
        self.token_threshold: float = kwargs.pop("token_threshold", 0.3)

        # Validate the values of the attributes
        self.validate(strict=True)


__all__ = ["ZaryaGenerationConfig"]
