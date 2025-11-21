# Code unchanged – reference implementation of model utilities.
from __future__ import annotations

from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

CACHE_DIR = ".cache/"

__all__ = [
    "LossAdaptiveScaler",
    "load_model_and_tokenizer",
    "build_optimizer",
    "build_scheduler",
]


class LossAdaptiveScaler:
    def __init__(
        self,
        ema_beta: float = 0.98,
        gamma: float = 0.7,
        f_min: float = 0.5,
        f_max: float = 1.5,
        eps: float = 1e-8,
    ) -> None:
        self.ema_beta = ema_beta
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self._ema: float | None = None

    def _update_ema(self, loss_val: float) -> None:
        if self._ema is None:
            self._ema = loss_val
        else:
            self._ema = self.ema_beta * self._ema + (1 - self.ema_beta) * loss_val

    def scale(self, loss) -> float:
        val = loss.detach().item() if torch.is_tensor(loss) else float(loss)
        self._update_ema(val)
        assert self._ema is not None
        factor = (self._ema / (val + self.eps)) ** self.gamma
        return max(self.f_min, min(self.f_max, factor))


def load_model_and_tokenizer(cfg, device: torch.device) -> Tuple:
    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_kwargs = {}
    if cfg.compute.get("precision", "fp32") == "int8":
        try:
            import bitsandbytes as bnb  # noqa: F401
            bnb_kwargs["load_in_8bit"] = True
        except Exception:
            print("[Warning] bitsandbytes unavailable – loading full-precision model")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
        device_map={"": device} if device.type == "cuda" else None,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
        **bnb_kwargs,
    )

    # Optional LoRA
    if cfg.model.get("adapter") and cfg.model.adapter.type == "lora" and get_peft_model:
        # Auto-detect target modules from the model architecture
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Extract the layer name (last part of the module path)
                layer_name = name.split('.')[-1]
                if layer_name and layer_name not in target_modules:
                    target_modules.append(layer_name)

        # If no modules found, use common fallback patterns
        if not target_modules:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        print(f"[Info] Using LoRA target modules: {target_modules}")

        l_cfg = LoraConfig(
            r=int(cfg.model.adapter.r),
            lora_alpha=int(cfg.model.adapter.alpha),
            lora_dropout=float(cfg.model.adapter.dropout),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, l_cfg)

    model.to(device)
    model.train()
    return tokenizer, model


def build_optimizer(cfg, model) -> torch.optim.Optimizer:
    opt_cfg = cfg.training.optimizer
    lr = opt_cfg.get("base_learning_rate") or opt_cfg.get("learning_rate") or 2e-5
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)


def build_scheduler(cfg, optimizer, total_steps: int):
    sch_cfg = cfg.training.scheduler
    warmup_steps = int(float(sch_cfg.num_warmup_steps_ratio) * total_steps)
    if sch_cfg.type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif sch_cfg.type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {sch_cfg.type}")