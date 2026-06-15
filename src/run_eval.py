#!/usr/bin/env python3
"""
run_eval.py - Point d'entree pour l'evaluation
================================================
Usage:
    python run_eval.py --config configs/eval.yaml
    python run_eval.py --config configs/eval.yaml --episodes 20
    python run_eval.py --config configs/eval.yaml --episodes 5 --render
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained QN agent")

    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/default_qn.yaml",
        help="Path to the config yaml file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (human mode)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record a video of the evaluation episodes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\nConfig   : {args.config}")
    print(f"Episodes : {args.episodes}")
    print(f"Render   : {args.render}")
    print(f"Video    : {args.video}\n")

    evaluate(
        config_path=args.config,
        episodes_override=args.episodes,
        render_override=args.render,
        video=args.video,
    )
