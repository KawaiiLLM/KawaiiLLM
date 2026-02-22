"""CLI argument dataclasses for KawaiiLLM training."""

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    meme_model_name_or_path: str = field(
        metadata={"help": "Path to Qwen3-Embedding-4B (MemE encoder)."}
    )
    llm_model_name_or_path: str = field(
        metadata={"help": "Path to Qwen3-8B-Base (decoder LLM)."}
    )
    num_mem_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of MEM tokens (N_max)."},
    )
    freeze_meme: bool = field(
        default=False,
        metadata={"help": "Freeze MemE encoder weights."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze LLM decoder weights."},
    )
    freeze_projector: bool = field(
        default=False,
        metadata={"help": "Freeze projector weights."},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention implementation for LLM: 'eager', 'sdpa', or "
            "'flash_attention_2'. If None, uses model default. "
            "Use 'eager' to bypass Flash Attention numerical issues."
        },
    )
    # Per-component learning rates
    meme_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for MemE encoder. If None, uses global LR."
        },
    )
    llm_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for LLM decoder. If None, uses global LR."
        },
    )
    projector_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for projector + mem_embeddings. "
            "If None, uses global LR."
        },
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""

    data_dirs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of formatted data directories."},
    )
    index_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-built train index JSON file."},
    )
    val_index_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-built val index JSON file. "
            "If set and file exists, enables eval during training."
        },
    )
    context_max_length: int = field(
        default=4096,
        metadata={"help": "Max token length for context input to MemE."},
    )
    target_max_length: int = field(
        default=4096,
        metadata={"help": "Max token length for target input to LLM."},
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    """Extended training arguments with KawaiiLLM defaults."""

    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Must be False to keep custom dataset fields."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "Must be False. Deterministic task rotation requires "
            "workers to re-fork each epoch to pick up the updated epoch."
        },
    )

    monitor_steps: int = field(
        default=10,
        metadata={
            "help": "Log per-component grad norms and per-task losses every "
            "this many optimizer steps."
        },
    )

    llm_freeze_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Freeze LLM for the first N%% of total training steps by "
            "setting its LR to 0. Only MemE and projector are trained during "
            "this period. Set to 0.0 to disable (default). E.g. 0.05 = first "
            "5%% of steps. Note: NTP samples are effectively wasted during the "
            "freeze since they only train the LLM."
        },
    )
    llm_unfreeze_warmup_ratio: float = field(
        default=0.01,
        metadata={
            "help": "After the LLM freeze ends, linearly ramp up LLM LR from "
            "0 to the scheduler's current value over this fraction of total "
            "training steps. Prevents a sudden LR jump on unfreeze. "
            "Default: 0.01 (1%%). Set to 0.0 to disable ramp-up."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.dataloader_persistent_workers:
            raise ValueError(
                "dataloader_persistent_workers=True is incompatible with "
                "deterministic task rotation. Epoch updates won't propagate "
                "to persistent workers."
            )
