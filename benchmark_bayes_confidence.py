from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random
import sys
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "foolbox"))

import eagerpy as ep
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models

from foolbox import PyTorchModel, samples
from foolbox.attacks import FGSM, LinfPGD, L2DeepFoolAttack, L2AdamBasicIterativeAttack, L2ClippingAwareAdditiveGaussianNoiseAttack, L2FMNAttack
from foolbox.models import ExpectationOverTransformationWrapper

from BayesConfidenceAttack import BayesianConfidenceAttack


class MonteCarloDropoutResNet18(torch.nn.Module):
    """Pretrained ResNet-18 with stochastic dropout before the classifier."""

    def __init__(self, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.base = models.resnet18(pretrained=True).eval()
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.base.fc(x)
        return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_epsilons(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def mean_probabilities(model: Any, inputs: ep.Tensor, n_samples: int) -> ep.Tensor:
    probs = []
    for _ in range(n_samples):
        logits = model(inputs)
        probs.append(ep.softmax(logits, axis=-1))
    return ep.stack(probs).mean(axis=0)


def predictive_entropy(mean_probs: ep.Tensor) -> ep.Tensor:
    return -(mean_probs * ep.log(mean_probs + 1e-12)).sum(axis=-1)


def native_float(x: Any) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def summarize_attack(
    *,
    attack_name: str,
    epsilons: Sequence[float],
    clipped_advs: Sequence[ep.Tensor],
    labels: ep.Tensor,
    eval_model: Any,
    eval_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for epsilon, advs in zip(epsilons, clipped_advs):
        mean_probs = mean_probabilities(eval_model, advs, eval_samples)
        predictions = mean_probs.argmax(axis=-1)
        success_mask = predictions != labels
        entropy = predictive_entropy(mean_probs)
        confidence = mean_probs.max(axis=-1)

        success_np = success_mask.numpy().astype(bool)
        entropy_np = entropy.numpy()
        confidence_np = confidence.numpy()

        if success_np.any():
            successful_entropy = entropy_np[success_np]
            successful_confidence = confidence_np[success_np]
            mean_success_entropy = float(np.mean(successful_entropy))
            median_success_entropy = float(np.median(successful_entropy))
            mean_success_confidence = float(np.mean(successful_confidence))
        else:
            mean_success_entropy = float("nan")
            median_success_entropy = float("nan")
            mean_success_confidence = float("nan")

        rows.append(
            {
                "attack": attack_name,
                "epsilon": float(epsilon),
                "asr": float(success_np.mean()),
                "mean_entropy_all": float(np.mean(entropy_np)),
                "median_entropy_all": float(np.median(entropy_np)),
                "mean_entropy_success": mean_success_entropy,
                "median_entropy_success": median_success_entropy,
                "mean_confidence_success": mean_success_confidence,
                "num_success": int(success_np.sum()),
                "num_total": int(success_np.shape[0]),
            }
        )
    return rows


def print_attack_summary(rows: Sequence[Dict[str, Any]]) -> None:
    attack_name = rows[0]["attack"]
    print()
    print(attack_name)
    print("-" * len(attack_name))
    print("epsilon    ASR      mean H(success)   mean conf(success)")
    for row in rows:
        entropy_text = (
            "nan"
            if np.isnan(row["mean_entropy_success"])
            else f"{row['mean_entropy_success']:.4f}"
        )
        confidence_text = (
            "nan"
            if np.isnan(row["mean_confidence_success"])
            else f"{row['mean_confidence_success']:.4f}"
        )
        print(
            f"{row['epsilon']:<10.4f}{row['asr']:<9.3f}{entropy_text:<18}{confidence_text}"
        )


def build_attacks(steps: int, mc_samples: int, lambda_param: float) -> List[Dict[str, Any]]:
    return [
        {
             "name": "FGSM",
             "attack": FGSM(),
             "use_eot": True,
        },
        {
            "name": f"DeepFoolAttack",
            "attack": L2DeepFoolAttack(),
            "use_eot": True,
        },
        {
             "name": f"LinfPGD",
             "attack": LinfPGD(steps=15),
             "use_eot": True,
        },
        {
            "name": f"FMNAttack",
            "attack": L2FMNAttack(),
            "use_eot": True,
        },
        {
            "name": f"AdamBasicIterativeAttack",
            "attack": L2AdamBasicIterativeAttack(),
            "use_eot": True,
        },
        {
            "name": f"GaussianNoiseAttack",
            "attack": L2ClippingAwareAdditiveGaussianNoiseAttack(),
            "use_eot": True,
        },
        {
             "name": f"BE-PGD",
             "attack": BayesianConfidenceAttack(
                 num_samples=mc_samples,
                 lambda_param=0.01,
                 steps=steps,
            ),
            "use_eot": False,
         }
    ]


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    stochastic_net = MonteCarloDropoutResNet18(dropout_p=args.dropout)
    stochastic_fmodel = PyTorchModel(
        stochastic_net,
        bounds=(0, 1),
        preprocessing=preprocessing,
    )
    eot_fmodel = ExpectationOverTransformationWrapper(
        stochastic_fmodel,
        n_steps=args.eot_attack_samples,
    )

    images, labels = ep.astensors(
        *samples(stochastic_fmodel, dataset=args.dataset, batchsize=args.batchsize)
    )

    clean_mean_probs = mean_probabilities(stochastic_fmodel, images, args.eval_samples)
    clean_predictions = clean_mean_probs.argmax(axis=-1)
    clean_success = clean_predictions == labels
    clean_entropy = predictive_entropy(clean_mean_probs)

    clean_metrics = {
        "dataset": args.dataset,
        "batchsize": args.batchsize,
        "clean_accuracy": float(clean_success.float32().mean().item()),
        "clean_mean_entropy": float(clean_entropy.mean().item()),
        "clean_median_entropy": float(np.median(clean_entropy.numpy())),
        "dropout": args.dropout,
        "eval_samples": args.eval_samples,
        "eot_attack_samples": args.eot_attack_samples,
    }

    print(f"clean accuracy:      {clean_metrics['clean_accuracy'] * 100:.1f} %")
    print(f"clean mean entropy:  {clean_metrics['clean_mean_entropy']:.4f}")
    print(f"clean median entropy:{clean_metrics['clean_median_entropy']:.4f}")
    print(f"dataset:             {args.dataset}")
    print(f"batchsize:           {args.batchsize}")

    attack_rows: List[Dict[str, Any]] = []
    attacks = build_attacks(args.steps, args.mc_samples, args.lambda_param)
    for spec in attacks:
        set_seed(args.seed)
        attack_model = eot_fmodel if spec["use_eot"] else stochastic_fmodel
        _, clipped_advs, _ = spec["attack"](
            attack_model,xs
            images,
            labels,
            epsilons=args.epsilons,
        )
        rows = summarize_attack(
            attack_name=spec["name"],
            epsilons=args.epsilons,
            clipped_advs=clipped_advs,
            labels=labels,
            eval_model=stochastic_fmodel,
            eval_samples=args.eval_samples,
        )
        attack_rows.extend(rows)
        print_attack_summary(rows)

    return {
        "config": {
            "seed": args.seed,
            "dataset": args.dataset,
            "batchsize": args.batchsize,
            "epsilons": list(args.epsilons),
            "steps": args.steps,
            "mc_samples": args.mc_samples,
            "lambda_param": args.lambda_param,
            "dropout": args.dropout,
            "eval_samples": args.eval_samples,
            "eot_attack_samples": args.eot_attack_samples,
        },
        "clean": clean_metrics,
        "rows": attack_rows,
    }


def write_outputs(payload: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_results.json"
    csv_path = output_dir / "benchmark_results.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload["rows"]
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print()
    print(f"saved JSON: {json_path}")
    print(f"saved CSV:  {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark BayesianConfidenceAttack against standard attacks using ASR and predictive entropy.",
    )
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--eot-attack-samples", type=int, default=16)
    parser.add_argument("--lambda-param", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epsilons",
        type=parse_epsilons,
        default=parse_epsilons(
            "0.0,0.0010,0.0020,0.0050,0.0080,0.0100,0.0200,0.0500,0.0800,0.100,"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "bayes_confidence_benchmark2",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_benchmark(args)
    write_outputs(payload, args.output_dir)


if __name__ == "__main__":
    main()
