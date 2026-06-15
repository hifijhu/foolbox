from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results" / "bayes_confidence_benchmark"


def load_rows(csv_path: Path) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, float | str] = {"attack": row["attack"] or ""}
            for key, value in row.items():
                if key == "attack":
                    continue
                if value is None or value == "" or value.lower() == "nan":
                    parsed[key] = float("nan")
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def group_by_attack(rows: List[Dict[str, float | str]]) -> DefaultDict[str, List[Dict[str, float | str]]]:
    grouped: DefaultDict[str, List[Dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["attack"])].append(row)
    for attack_rows in grouped.values():
        attack_rows.sort(key=lambda row: float(row["epsilon"]))
    return grouped


def finite_xy(xs: List[float], ys: List[float]) -> tuple[List[float], List[float]]:
    x_out: List[float] = []
    y_out: List[float] = []
    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            x_out.append(x)
            y_out.append(y)
    return x_out, y_out


def save_asr_plot(grouped: DefaultDict[str, List[Dict[str, float | str]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, rows in grouped.items():
        xs = [float(row["epsilon"]) for row in rows]
        ys = [float(row["asr"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=attack)
    ax.set_title("Attack Success Rate vs Epsilon")
    ax.set_xlabel("Linf epsilon")
    ax.set_ylabel("Attack success rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "asr_vs_epsilon.png", dpi=180)
    plt.close(fig)


def save_entropy_plot(grouped: DefaultDict[str, List[Dict[str, float | str]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, rows in grouped.items():
        xs = [float(row["epsilon"]) for row in rows]
        ys = [float(row["mean_entropy_success"]) for row in rows]
        xs, ys = finite_xy(xs, ys)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=attack)
    ax.set_title("Mean Entropy of Successful Adversarials vs Epsilon")
    ax.set_xlabel("Linf epsilon")
    ax.set_ylabel("Mean predictive entropy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "entropy_success_vs_epsilon.png", dpi=180)
    plt.close(fig)


def save_confidence_plot(grouped: DefaultDict[str, List[Dict[str, float | str]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, rows in grouped.items():
        xs = [float(row["epsilon"]) for row in rows]
        ys = [float(row["mean_confidence_success"]) for row in rows]
        xs, ys = finite_xy(xs, ys)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=attack)
    ax.set_title("Mean Confidence of Successful Adversarials vs Epsilon")
    ax.set_xlabel("Linf epsilon")
    ax.set_ylabel("Mean max probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_success_vs_epsilon.png", dpi=180)
    plt.close(fig)


def save_tradeoff_plot(grouped: DefaultDict[str, List[Dict[str, float | str]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for attack, rows in grouped.items():
        xs = [float(row["mean_entropy_success"]) for row in rows]
        ys = [float(row["asr"]) for row in rows]
        xs, ys = finite_xy(xs, ys)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=attack)
    ax.set_title("ASR vs Mean Entropy of Successful Adversarials")
    ax.set_xlabel("Mean predictive entropy")
    ax.set_ylabel("Attack success rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "asr_vs_entropy_tradeoff.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Bayesian confidence benchmark metrics.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "benchmark_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "plots",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.csv)
    grouped = group_by_attack(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_asr_plot(grouped, args.output_dir)
    save_entropy_plot(grouped, args.output_dir)
    save_confidence_plot(grouped, args.output_dir)
    save_tradeoff_plot(grouped, args.output_dir)

    print(f"saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
