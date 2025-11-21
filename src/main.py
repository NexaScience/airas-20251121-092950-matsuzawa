# src/main.py – unchanged; still spawns train.py in a subprocess.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config")
def main(cfg: DictConfig) -> None:
    for field in ("run", "mode", "results_dir"):
        if field not in cfg or cfg[field] in (None, "???"):
            raise ValueError(f"Missing required CLI arg: {field}")

    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"mode={cfg.mode}",
        f"results_dir={cfg.results_dir}",
    ]

    if cfg.mode == "trial":
        cmd += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
        ]
    elif cfg.mode == "full":
        cmd += ["wandb.mode=online"]
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    print("[Launcher]", " \
              ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()