"""Entry point for KawaiiLLM training.

Usage:
    deepspeed --num_gpus 8 src/train/train.py \
        --deepspeed configs/ds_zero2.json \
        --meme_model_name_or_path /path/to/Qwen3-Embedding-4B \
        --llm_model_name_or_path /path/to/Qwen3-8B-Base \
        --data_dirs data/novels/formatted data/bilibili/formatted ... \
        --index_path data/train_index.json \
        --output_dir output/kawaii_v1 \
        ...
"""

import glob
import logging
import os
import sys

from transformers import AutoTokenizer, HfArgumentParser, set_seed

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .collator import KawaiiDataCollator
from .dataset import KawaiiDataset
from .model import SPECIAL_TOKENS, KawaiiLLMModel
from .trainer import CurriculumCallback, GradNormCallback, LLMFreezeCallback, NaNDetectorCallback, TaskLossCallback, KawaiiTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def find_latest_checkpoint(output_dir: str):
    """Find the latest checkpoint-* directory if it exists."""
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Load tokenizer from LLM path (Qwen3 family shared tokenizer)
    logger.info("Loading tokenizer from %s", model_args.llm_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.pad_token)

    # Register special tokens
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    logger.info(
        "Registered %d special tokens: %s", num_added, SPECIAL_TOKENS
    )

    # Verify tokenizer compatibility between MemE and LLM
    meme_tokenizer = AutoTokenizer.from_pretrained(
        model_args.meme_model_name_or_path, trust_remote_code=True
    )
    if meme_tokenizer.vocab_size != tokenizer.vocab_size:
        logger.warning(
            "Tokenizer vocab size mismatch: MemE=%d, LLM=%d. "
            "This may cause incorrect context encoding.",
            meme_tokenizer.vocab_size,
            tokenizer.vocab_size,
        )
    del meme_tokenizer

    # Build or load model
    latest_ckpt = find_latest_checkpoint(training_args.output_dir)
    if latest_ckpt and os.path.isdir(os.path.join(latest_ckpt, "meme")):
        logger.info("Resuming from checkpoint: %s", latest_ckpt)
        model = KawaiiLLMModel.from_checkpoint(
            checkpoint_dir=latest_ckpt,
            num_mem_tokens=model_args.num_mem_tokens,
            freeze_meme=model_args.freeze_meme,
            freeze_llm=model_args.freeze_llm,
            freeze_projector=model_args.freeze_projector,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        logger.info("Building model from pretrained weights")
        model = KawaiiLLMModel(
            meme_model_name_or_path=model_args.meme_model_name_or_path,
            llm_model_name_or_path=model_args.llm_model_name_or_path,
            num_mem_tokens=model_args.num_mem_tokens,
            freeze_meme=model_args.freeze_meme,
            freeze_llm=model_args.freeze_llm,
            freeze_projector=model_args.freeze_projector,
            attn_implementation=model_args.attn_implementation,
        )

    # Resize embeddings to accommodate special tokens
    model.meme.resize_token_embeddings(len(tokenizer))
    model.llm.resize_token_embeddings(len(tokenizer))
    logger.info(
        "Resized embeddings to %d tokens (added %d special tokens)",
        len(tokenizer), num_added,
    )

    # Set special token IDs on model
    special_token_ids = {
        tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS
    }
    special_token_ids["pad_token_id"] = tokenizer.pad_token_id
    model.set_special_token_ids(special_token_ids)

    # Enable gradient checkpointing.
    # use_reentrant=False avoids the known Flash Attention + reentrant GC
    # incompatibility where GC re-runs the forward and Flash Attention's saved
    # log_sum_exp tensors get recomputed, producing NaN in the backward pass.
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    # Log parameter counts
    total, trainable = count_parameters(model)
    logger.info(
        "Model parameters: total=%d (%.2fB), trainable=%d (%.2fB)",
        total, total / 1e9, trainable, trainable / 1e9,
    )

    # Build index path — use provided or auto-detect
    index_path = data_args.index_path
    if index_path is None:
        index_path = os.path.join(training_args.output_dir, "train_index.json")
        if not os.path.exists(index_path):
            logger.error(
                "No index_path provided and no index found at %s. "
                "Run build_index.py first.",
                index_path,
            )
            sys.exit(1)

    # Build dataset
    dataset = KawaiiDataset(
        index_path=index_path,
        tokenizer=tokenizer,
        context_max_length=data_args.context_max_length,
        target_max_length=data_args.target_max_length,
        num_mem_tokens=model_args.num_mem_tokens,
    )

    # Build val dataset (if val_index_path is provided and exists)
    # Val dataset uses fixed epoch=0 task rotation for stable eval metrics.
    val_dataset = None
    if data_args.val_index_path:
        if os.path.exists(data_args.val_index_path):
            val_dataset = KawaiiDataset(
                index_path=data_args.val_index_path,
                tokenizer=tokenizer,
                context_max_length=data_args.context_max_length,
                target_max_length=data_args.target_max_length,
                num_mem_tokens=model_args.num_mem_tokens,
            )
            logger.info("Loaded val dataset: %d entries", len(val_dataset))
        else:
            logger.warning(
                "val_index_path=%s not found. Evaluation disabled. "
                "Run build_index.py with --val_ratio first.",
                data_args.val_index_path,
            )

    # Build collator
    collator = KawaiiDataCollator(
        tokenizer=tokenizer,
        num_mem_tokens=model_args.num_mem_tokens,
    )

    # Build callbacks
    callbacks = [
        CurriculumCallback(dataset=dataset),
        NaNDetectorCallback(),
        GradNormCallback(monitor_steps=training_args.monitor_steps),
        TaskLossCallback(monitor_steps=training_args.monitor_steps),
    ]
    if training_args.llm_freeze_ratio > 0.0:
        callbacks.append(LLMFreezeCallback(
            freeze_ratio=training_args.llm_freeze_ratio,
            unfreeze_warmup_ratio=training_args.llm_unfreeze_warmup_ratio,
        ))
        logger.info(
            "LLM freeze enabled: LLM LR=0 for first %.1f%%, ramp-up over next %.1f%%",
            training_args.llm_freeze_ratio * 100,
            training_args.llm_unfreeze_warmup_ratio * 100,
        )

    # Build trainer
    trainer = KawaiiTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        meme_lr=model_args.meme_lr,
        llm_lr=model_args.llm_lr,
        projector_lr=model_args.projector_lr,
    )

    # Train
    if latest_ckpt:
        trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        trainer.train()

    # Final save
    trainer.save_state()
    model_to_save = model
    if hasattr(model, "module"):
        model_to_save = model.module
    model_to_save.save_checkpoint(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info("Training complete. Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    train()
