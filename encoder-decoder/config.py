"""
Typed configuration loaded from config.yml.

No values are hard-coded here: every field is required and must be present in the
YAML file. Missing or invalid entries raise a validation error at load time.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    # Reject unknown keys so typos in config.yml are caught instead of silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_StrictModel):
    name: str
    vocab_size: int = Field(gt=0)
    src_vocab_size: int = Field(gt=0)
    tgt_vocab_size: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_encoder_layers: int = Field(ge=0)
    n_decoder_layers: int = Field(ge=0)
    d_ff: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    activation: Literal["relu", "gelu", "silu"]
    layer_norm_eps: float = Field(gt=0.0)
    bias: bool
    max_seq_len: int = Field(gt=0)
    positional_encoding: Literal["sinusoidal", "learned"]
    share_embeddings: bool

    @model_validator(mode="after")
    def _check_heads(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class OptimizerConfig(_StrictModel):
    name: str
    lr: float = Field(gt=0.0)
    betas: List[float] = Field(min_length=2, max_length=2)
    eps: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class SchedulerConfig(_StrictModel):
    name: str
    warmup_steps: int = Field(ge=0)
    min_lr: float = Field(ge=0.0)


class LossConfig(_StrictModel):
    label_smoothing: float = Field(ge=0.0, le=1.0)


class TrainingConfig(_StrictModel):
    batch_size: int = Field(gt=0)
    num_epochs: int = Field(gt=0)
    seed: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    grad_clip: float = Field(ge=0.0)
    gradient_accumulation_steps: int = Field(ge=1)
    mixed_precision: Literal["no", "fp16", "bf16"]


class DataConfig(_StrictModel):
    dataset_name: str
    train_path: str
    val_path: str
    test_path: str
    src_lang: str
    tgt_lang: str
    num_workers: int = Field(ge=0)
    pin_memory: bool
    shuffle: bool


class TokenizerConfig(_StrictModel):
    type: str
    vocab_path: str
    pad_token: str
    pad_token_id: int = Field(ge=0)
    unk_token: str
    unk_token_id: int = Field(ge=0)
    bos_token: str
    bos_token_id: int = Field(ge=0)
    eos_token: str
    eos_token_id: int = Field(ge=0)


class InferenceConfig(_StrictModel):
    decoding_strategy: Literal["greedy", "beam_search", "top_k", "top_p"]
    beam_size: int = Field(gt=0)
    max_length: int = Field(gt=0)
    temperature: float = Field(gt=0.0)
    top_k: int = Field(gt=0)
    top_p: float = Field(gt=0.0, le=1.0)
    repetition_penalty: float = Field(gt=0.0)


class LoggingConfig(_StrictModel):
    use_tensorboard: bool
    use_wandb: bool
    wandb_project: str


class SystemConfig(_StrictModel):
    device: str  # "cuda", "cuda:N", "cpu", "mps" or "auto"
    checkpoint_dir: str
    save_every_epochs: int = Field(gt=0)
    keep_best_n_checkpoints: int = Field(gt=0)
    log_interval: int = Field(gt=0)
    eval_interval: int = Field(gt=0)
    logging: LoggingConfig


class Settings(_StrictModel):
    """Configuration settings for the Transformer model and training pipeline."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    inference: InferenceConfig
    system: SystemConfig

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yml") -> "Settings":
        """Load and validate settings from a YAML file."""
        config_path = Path(path)
        if not config_path.is_absolute() and not config_path.exists():
            config_path = Path(__file__).resolve().parent / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.model_validate(data)


@lru_cache(maxsize=None)
def get_settings(config_path: str = "config.yml") -> Settings:
    """Returns a cached instance of the Settings loaded from the config YAML file."""
    return Settings.from_yaml(config_path)


if __name__ == "__main__":
    settings = get_settings()
    print("Settings loaded successfully!")
    print(f"Model d_model: {settings.model.d_model}")
    print(f"Number of heads: {settings.model.n_heads}")
    print(f"Device: {settings.system.device}")
    print(f"Learning rate: {settings.training.optimizer.lr}")
