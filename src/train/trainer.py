"""KawaiiTrainer: custom Trainer with per-component LR, save/load, curriculum."""

import logging
import os
from typing import Dict, Optional, Union

import torch
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names

from torch.utils.data import DataLoader

from .dataset import KawaiiDataset
from .model import KawaiiLLMModel, RMSNorm

logger = logging.getLogger(__name__)


class CurriculumCallback(TrainerCallback):
    """Updates training progress on dataset and collator each step."""

    def __init__(self, dataset, collator):
        self.dataset = dataset
        self.collator = collator

    def on_train_begin(self, args, state, control, **kwargs):
        """Restore curriculum progress when resuming from checkpoint."""
        if state.global_step > 0 and state.max_steps > 0:
            progress = state.global_step / state.max_steps
            self.dataset.set_training_progress(progress)
            self.collator.set_training_progress(progress)
            logger.info(
                "Restored curriculum progress: %.2f%% (step %d/%d)",
                progress * 100, state.global_step, state.max_steps,
            )

    def on_step_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            progress = state.global_step / state.max_steps
        else:
            progress = 0.0
        self.dataset.set_training_progress(progress)
        self.collator.set_training_progress(progress)


class KawaiiTrainer(Trainer):
    """Trainer with per-component optimizer groups and custom checkpointing."""

    def __init__(
        self,
        meme_lr: Optional[float] = None,
        llm_lr: Optional[float] = None,
        projector_lr: Optional[float] = None,
        **kwargs,
    ):
        self._meme_lr = meme_lr
        self._llm_lr = llm_lr
        self._projector_lr = projector_lr
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """Override to inject worker_init_fn for file handle safety."""
        dataloader = super().get_train_dataloader()
        if isinstance(self.train_dataset, KawaiiDataset):
            dataloader.worker_init_fn = KawaiiDataset.worker_init_fn
        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Extract n_mem from inputs and call model.forward()."""
        n_mem = inputs.pop("n_mem")
        if isinstance(n_mem, torch.Tensor):
            n_mem = n_mem.item()

        outputs = model(
            context_ids=inputs["context_ids"],
            context_attention_mask=inputs["context_attention_mask"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            n_mem=n_mem,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """Create optimizer with per-component learning rate groups."""
        model: KawaiiLLMModel = self.model

        # Collect parameter names that should not have weight decay
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm, RMSNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        global_lr = self.args.learning_rate

        # Build parameter groups per component
        param_groups = []

        component_configs = [
            ("meme", model.meme, self._meme_lr),
            ("llm", model.llm, self._llm_lr),
            ("projector", model.projector, self._projector_lr),
            ("mem_embeddings", model.mem_embeddings, self._projector_lr),
        ]

        for comp_name, component, comp_lr in component_configs:
            lr = comp_lr if comp_lr is not None else global_lr

            decay_params = []
            no_decay_params = []
            for name, param in component.named_parameters():
                if not param.requires_grad:
                    continue
                full_name = f"{comp_name}.{name}"
                if full_name in decay_parameters:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

            if decay_params:
                param_groups.append({
                    "params": decay_params,
                    "lr": lr,
                    "weight_decay": self.args.weight_decay,
                    "name": f"{comp_name}_decay",
                })
            if no_decay_params:
                param_groups.append({
                    "params": no_decay_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "name": f"{comp_name}_no_decay",
                })

        # Filter out empty groups
        param_groups = [g for g in param_groups if g["params"]]

        if not param_groups:
            logger.warning("No trainable parameters found!")
            return

        # Log group info
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            logger.info(
                "Optimizer group '%s': %d params, lr=%.2e, wd=%.4f",
                g.get("name", "unnamed"),
                n_params,
                g["lr"],
                g["weight_decay"],
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args, model
        )
        # Remove 'lr' from kwargs since we set it per group
        optimizer_kwargs.pop("lr", None)

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save model components to separate subdirectories."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Unwrap model if wrapped by DeepSpeed / DDP
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        if isinstance(model, KawaiiLLMModel):
            model.save_checkpoint(output_dir)
        else:
            # Fallback for unexpected model types
            super()._save(output_dir, state_dict)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
