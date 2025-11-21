# src/train.py
"""Training script with Loss-Adaptive LR Scaling (LALS).
This file executes **one** experiment run.  It can be launched either via the
launcher (``python -m src.main``) *or* directly (``python -m src.train``).
All mode-specific safety guards (trial/full) therefore live *inside* this file
so that *any* entry path respects the specification.
"""
from __future__ import annotations

import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model import (
    LossAdaptiveScaler,
    build_optimizer,
    build_scheduler,
    load_model_and_tokenizer,
)
from src.preprocess import DataCollatorCausalLM, build_datasets

# ---------------------------------------------------------------------------
# Reproducibility helper -----------------------------------------------------
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation / evaluation ----------------------------------------------------
# ---------------------------------------------------------------------------

def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    cfg: DictConfig,
) -> Dict[str, float]:
    """Greedy decoding + exact-match accuracy on the validation split."""

    model.eval()
    total_loss = 0.0
    seen = 0
    correct = 0

    gen_cfg = {
        "max_new_tokens": 64,
        "temperature": float(cfg.evaluation.generation.temperature),
    }

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            bs = batch["input_ids"].size(0)
            total_loss += outputs.loss.item() * bs

            p_lens: List[int] = batch["prompt_lengths"].tolist()
            answers: List[str] = batch["answers"]
            for i in range(bs):
                prompt_len = p_lens[i]
                gens = model.generate(
                    input_ids=batch["input_ids"][i, :prompt_len].unsqueeze(0),
                    attention_mask=batch["attention_mask"][i, :prompt_len].unsqueeze(0),
                    **gen_cfg,
                )
                pred = tokenizer.decode(gens[0, prompt_len:], skip_special_tokens=True).strip()
                gold = answers[i].strip()
                correct += int(pred == gold)
                seen += 1

    model.train()
    val_loss = total_loss / max(seen, 1)
    val_acc = correct / max(seen, 1)
    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "accuracy": val_acc,  # canonical name for evaluation script
        "val_total_examples": seen,
    }


# ---------------------------------------------------------------------------
# Single training run --------------------------------------------------------
# ---------------------------------------------------------------------------

def _train_one_run(cfg: DictConfig, *, use_wandb: bool) -> Dict[str, float]:
    """End-to-end optimisation according to *cfg*.  Returns final metrics."""

    # ---------------- Environment + model --------------------------------
    _set_seed(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trial-mode forces a micro-model so that CI fits into 500 MB RAM.
    if cfg.mode == "trial":
        cfg.model.name = "sshleifer/tiny-gpt2"

    tokenizer, model = load_model_and_tokenizer(cfg, device)

    # ---------------- Data ------------------------------------------------
    ds_train, ds_val = build_datasets(cfg, tokenizer)
    collator = DataCollatorCausalLM(tokenizer)

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=True,
        collate_fn=collator,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        collate_fn=collator,
    )

    # ---------------- Optimiser / scheduler ------------------------------
    optimizer = build_optimizer(cfg, model)
    total_steps = len(dl_train) * int(cfg.training.epochs)
    scheduler = build_scheduler(cfg, optimizer, total_steps)

    lals: LossAdaptiveScaler | None = None
    if cfg.training.get("lr_scaling") and cfg.training.lr_scaling.type == "lals":
        lals = LossAdaptiveScaler(
            ema_beta=float(cfg.training.lr_scaling.ema_beta),
            gamma=float(cfg.training.lr_scaling.gamma),
            f_min=float(cfg.training.lr_scaling.f_min),
            f_max=float(cfg.training.lr_scaling.f_max),
            eps=float(cfg.training.lr_scaling.eps),
        )

    # ---------------- WandB ----------------------------------------------
    run_id: str = getattr(cfg, "run_id", None) or str(cfg.run)
    if use_wandb:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[WandB] {run.get_url()}")
        wandb.define_metric("step")
        for m in ("train_loss", "effective_lr", "lr_factor", "accuracy"):
            wandb.define_metric(m, step_metric="step")

    # ---------------- Training loop --------------------------------------
    grad_acc = int(cfg.dataloader.gradient_accumulation_steps)
    global_step = 0
    running_loss = 0.0
    best_val_acc = -1.0

    for epoch in range(int(cfg.training.epochs)):
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_acc
            loss.backward()

            # LR scaling --------------------------------------------------
            scheduled_lr = scheduler.get_last_lr()[0]
            lr_factor = 1.0
            if lals is not None:
                lr_factor = lals.scale(loss * grad_acc)  # scale uses *true* batch loss
            effective_lr = scheduled_lr * lr_factor
            for g in optimizer.param_groups:
                g["lr"] = effective_lr

            # Optimiser step --------------------------------------------
            if (step + 1) % grad_acc == 0:
                clip_grad_norm_(model.parameters(), float(cfg.training.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                running_loss += loss.item() * grad_acc
                if use_wandb and global_step % int(cfg.logging.log_every_n_steps) == 0:
                    wandb.log(
                        {
                            "step": global_step,
                            "train_loss": running_loss / cfg.logging.log_every_n_steps,
                            "effective_lr": effective_lr,
                            "lr_factor": lr_factor,
                        }
                    )
                    running_loss = 0.0

        # ---------------- End-of-epoch validation ------------------------
        metrics = _evaluate(model, dl_val, tokenizer, device, cfg)
        if use_wandb:
            wandb.log({"epoch": epoch + 1, **metrics})
        print(
            f"Epoch {epoch+1}: val_acc={metrics['accuracy']:.4f}  val_loss={metrics['val_loss']:.4f}"
        )
        best_val_acc = max(best_val_acc, metrics["accuracy"])

    # ---------------- Save artefacts ------------------------------------
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    if use_wandb:
        wandb.summary["accuracy"] = best_val_acc
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["final_val_accuracy"] = metrics["accuracy"]
        wandb.summary["final_val_loss"] = metrics["val_loss"]
        wandb.summary["val_total_examples"] = metrics["val_total_examples"]
        wandb.finish()

    return metrics


# ---------------------------------------------------------------------------
# Optuna objective -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _apply_sample_to_cfg(cfg: DictConfig, name: str, value: Any) -> None:
    """Injects an Optuna-sampled hyper-parameter into the nested config tree."""
    if name in cfg.training.optimizer:
        cfg.training.optimizer[name] = value
    elif cfg.training.get("lr_scaling") and name in cfg.training.lr_scaling:
        cfg.training.lr_scaling[name] = value
    elif name == "warmup_ratio":
        cfg.training.scheduler.num_warmup_steps_ratio = value
    elif name == "batch_size":
        cfg.dataloader.batch_size = value
    elif name == "scheduler_type":
        cfg.training.scheduler.type = value
    else:
        OmegaConf.update(cfg, name, value, merge=True)


def _optuna_objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg = deepcopy(base_cfg)
    for pname, spec in cfg.optuna.search_space.items():
        ptype = spec["type"]
        if ptype == "loguniform":
            sampled = trial.suggest_float(pname, spec["low"], spec["high"], log=True)
        elif ptype == "uniform":
            sampled = trial.suggest_float(pname, spec["low"], spec["high"])
        elif ptype == "categorical":
            sampled = trial.suggest_categorical(pname, spec["choices"])
        else:
            raise ValueError(f"Unknown Optuna param type: {ptype}")
        _apply_sample_to_cfg(cfg, pname, sampled)

    # Lightweight objective – single epoch, no WandB
    cfg.training.epochs = 1
    cfg.wandb.mode = "disabled"
    metrics = _train_one_run(cfg, use_wandb=False)
    return metrics["accuracy"]


# ---------------------------------------------------------------------------
# Hydra entry-point ----------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    """Hydra entry – merges run-specific YAML, applies mode overrides and trains."""

    # ---------------------------------------------------------------------
    # Merge run-specific config -------------------------------------------
    run_cfg_path = Path(__file__).resolve().parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Run-config {run_cfg_path} not found.")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(run_cfg_path))

    # ---------------------------------------------------------------------
    # Mode-specific safeguards (works for *direct* invocation too) ---------
    # ---------------------------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        # keep memory tiny
        cfg.dataloader.batch_size = min(2, int(cfg.dataloader.batch_size))
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Ensure run_id exists (needed for WandB resume)
    if not hasattr(cfg, "run_id"):
        cfg.run_id = str(cfg.run)

    # ---------------------------------------------------------------------
    # Hyper-parameter optimisation ----------------------------------------
    # ---------------------------------------------------------------------
    if int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda t: _optuna_objective(t, cfg), n_trials=int(cfg.optuna.n_trials))
        print(f"[Optuna] Best value: {study.best_value:.4f}\n          params: {json.dumps(study.best_params)}")
        for k, v in study.best_params.items():
            _apply_sample_to_cfg(cfg, k, v)

    # ---------------------------------------------------------------------
    # Final training run ---------------------------------------------------
    # ---------------------------------------------------------------------
    use_wandb = cfg.wandb.mode != "disabled"
    _train_one_run(cfg, use_wandb=use_wandb)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()