# src/evaluate.py
"""Independent evaluation + visualisation after *all* training runs.*
The script pulls metrics from WandB and produces per-run artefacts as well as
cross-run comparisons (bar-chart, box-plot, table) + statistical tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WandB run aggregation")
    p.add_argument("results_dir", type=str, help="Output directory for artefacts")
    p.add_argument("run_ids", type=str, help='JSON list of run IDs, e.g. "[\"run-1\", \"run-2\"]"')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Any, path: Path) -> None:
    _mkdir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _higher_better(metric_name: str) -> bool:
    n = metric_name.lower()
    return not any(k in n for k in ("loss", "error", "perplexity"))


# ---------------------------------------------------------------------------
# Per-run plots --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _learning_curve(df: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    if "train_loss" not in df.columns:
        return Path()
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x=df.index, y="train_loss", label="train_loss")
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(f"Learning curve – {run_id}")
    plt.legend()
    plt.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _confusion(correct: int, incorrect: int, run_id: str, out_dir: Path) -> Path:
    mat = np.array([[correct, incorrect]])
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        mat,
        annot=[[correct, incorrect]],
        fmt="d",
        cbar=False,
        xticklabels=["✓", "✗"],
        yticklabels=[""],
    )
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


# ---------------------------------------------------------------------------
# Cross-run plots ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _bar(values: Dict[str, float], title: str, ylabel: str, out_dir: Path, topic: str) -> Path:
    plt.figure(figsize=(max(6, 1.3 * len(values)), 4))
    ax = sns.barplot(x=list(values.keys()), y=list(values.values()), palette="viridis")
    plt.xticks(rotation=40, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    for i, v in enumerate(values.values()):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fname = out_dir / f"comparison_{topic}_bar_chart.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _box(values: Dict[str, float], title: str, ylabel: str, out_dir: Path, topic: str) -> Path:
    df = pd.DataFrame({"run_id": list(values.keys()), ylabel: list(values.values())})
    plt.figure(figsize=(max(6, 1.3 * len(values)), 4))
    sns.boxplot(data=df, y=ylabel)
    sns.swarmplot(data=df, y=ylabel, color="k", size=4, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    fname = out_dir / f"comparison_{topic}_box_plot.pdf"
    plt.savefig(fname)
    plt.close()
    return fname


def _metrics_table(metrics_all: Dict[str, Dict[str, float]], out_dir: Path) -> Path:
    df = pd.DataFrame(metrics_all).T  # metrics x runs
    csv_path = out_dir / "comparison_metrics_table.csv"
    df.to_csv(csv_path)

    # Render as PDF table --------------------------------------------------
    plt.figure(figsize=(max(8, 0.7 * df.shape[1]), 0.4 * df.shape[0] + 1))
    plt.axis("off")
    tbl = plt.table(
        cellText=np.round(df.values, 4),
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    fname = out_dir / "comparison_metrics_table.pdf"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


# ---------------------------------------------------------------------------
# Statistical tests ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _stat_tests(values: Dict[str, float], out_dir: Path, metric_name: str) -> Path:
    prop = [v for k, v in values.items() if any(x in k.lower() for x in ("proposed", "lals"))]
    base = [v for k, v in values.items() if any(x in k.lower() for x in ("baseline", "comparative"))]
    results: Dict[str, Any] = {
        "proposed": prop,
        "baseline": base,
        "n_proposed": len(prop),
        "n_baseline": len(base),
    }
    if len(prop) >= 2 and len(base) >= 2:
        results["ttest"] = {
            "statistic": float((t := stats.ttest_ind(prop, base, equal_var=False)).statistic),
            "pvalue": float(t.pvalue),
        }
        results["mannwhitney"] = {
            "statistic": float((u := stats.mannwhitneyu(prop, base)).statistic),
            "pvalue": float(u.pvalue),
        }
    _save_json(results, out_dir / "significance_tests.json")
    return out_dir / "significance_tests.json"


# ---------------------------------------------------------------------------
# Main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    args = _cli()
    results_dir = Path(args.results_dir).absolute()
    run_ids: List[str] = json.loads(args.run_ids.strip("'\""))

    cfg_root = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    wandb_cfg = OmegaConf.load(cfg_root).wandb
    entity, project = wandb_cfg.entity, wandb_cfg.project

    api = wandb.Api()

    primary_metric = "accuracy"
    metrics_all: Dict[str, Dict[str, float]] = {}

    # ---------------------------------------------------------------------
    # Per-run processing ---------------------------------------------------
    # ---------------------------------------------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        run_dir = results_dir / rid
        _mkdir(run_dir)

        history: pd.DataFrame = run.history()
        summary: Dict[str, Any] = dict(run.summary)
        cfg: Dict[str, Any] = dict(run.config)

        # Fallback if primary metric absent
        if primary_metric not in summary:
            for c in ("val_accuracy", "best_val_accuracy", "final_val_accuracy"):
                if c in summary:
                    summary[primary_metric] = float(summary[c])
                    break

        # Persist metrics --------------------------------------------------
        _save_json({"config": cfg, "summary": summary}, run_dir / "metrics.json")

        # Update aggregated dict ------------------------------------------
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                metrics_all.setdefault(k, {})[rid] = float(v)

        # Per-run figures --------------------------------------------------
        figs: List[Path] = []
        figs.append(_learning_curve(history, rid, run_dir))
        if "val_total_examples" in summary and primary_metric in summary:
            total = int(summary["val_total_examples"])
            correct = int(summary[primary_metric] * total)
            figs.append(_confusion(correct, total - correct, rid, run_dir))
        # Print generated paths
        for f in figs:
            if f.exists():
                print(f)

    # ---------------------------------------------------------------------
    # Aggregated analysis --------------------------------------------------
    # ---------------------------------------------------------------------
    cmp_dir = results_dir / "comparison"
    _mkdir(cmp_dir)

    if primary_metric not in metrics_all:
        raise RuntimeError(f"'{primary_metric}' not logged ‑ cannot aggregate")

    higher_is_better = _higher_better(primary_metric)
    best_prop_val = -np.inf if higher_is_better else np.inf
    best_prop_id = None
    best_base_val = -np.inf if higher_is_better else np.inf
    best_base_id = None
    for rid, val in metrics_all[primary_metric].items():
        if any(x in rid.lower() for x in ("proposed", "lals")):
            cond = val > best_prop_val if higher_is_better else val < best_prop_val
            if cond:
                best_prop_val, best_prop_id = val, rid
        elif any(x in rid.lower() for x in ("baseline", "comparative")):
            cond = val > best_base_val if higher_is_better else val < best_base_val
            if cond:
                best_base_val, best_base_id = val, rid

    gap = None
    if best_prop_id and best_base_id and best_base_val != 0:
        diff = best_prop_val - best_base_val
        gap = (diff / abs(best_base_val)) * 100.0
        if not higher_is_better:
            gap = -gap

    aggregated = {
        "primary_metric": primary_metric,
        "metrics": metrics_all,
        "best_proposed": {"run_id": best_prop_id, "value": best_prop_val},
        "best_baseline": {"run_id": best_base_id, "value": best_base_val},
        "gap": gap,
    }
    _save_json(aggregated, cmp_dir / "aggregated_metrics.json")

    # ---------------------------------------------------------------------
    # Comparison figures ---------------------------------------------------
    # ---------------------------------------------------------------------
    figs_cmp: List[Path] = []
    vals_primary = metrics_all[primary_metric]
    figs_cmp.append(
        _bar(vals_primary, f"{primary_metric} per run", primary_metric, cmp_dir, primary_metric)
    )
    figs_cmp.append(
        _box(vals_primary, f"{primary_metric} distribution", primary_metric, cmp_dir, primary_metric)
    )
    figs_cmp.append(_metrics_table(metrics_all, cmp_dir))
    figs_cmp.append(_stat_tests(vals_primary, cmp_dir, primary_metric))

    for f in figs_cmp:
        if f.exists():
            print(f)


if __name__ == "__main__":
    main()