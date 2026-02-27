"""
Batch detection pipeline: models x biases x dataset_configs.

Usage:
    accelerate launch --num_processes <N> scripts/batch_detect.py configs/batch_detection.yaml
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import os
import random
import yaml
import torch
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import get_dataset_config_names
from accelerate import Accelerator
from accelerate.utils import gather_object

from src.helper_functions import clear_memory
from scripts.detect_subliminal import (
    detect_subliminal,
    load_hf_sft_dataset,
    resolve_system_prompt,
)


# ── Model lifecycle ──────────────────────────────────────────────────────────

def load_model(judge_model, scoring_config, accelerator):
    """Load model + tokenizer, prepare with Accelerate."""
    precision = scoring_config.get("training_precision", 16)
    dtype = torch.bfloat16 if precision == 16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(judge_model, torch_dtype=dtype)

    if accelerator is not None:
        model = accelerator.prepare(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    return model, tokenizer


def unload_model(model, tokenizer, accelerator):
    """Free model from GPU memory."""
    if accelerator is not None:
        accelerator.free_memory()
    del model, tokenizer
    clear_memory()


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(model, tokenizer, hf_repo, ds_cfg, bias_cfg, controls,
               scoring_config, dataset_defaults, rank, world_size):
    """
    Load one dataset config, build traits_config, run detect_subliminal.
    Returns the result dict (on rank 0) or None (other ranks).
    """
    config_name = ds_cfg["config_name"]
    response_col = ds_cfg.get("response_column", "completion")
    prompt_col = dataset_defaults.get("prompt_column", "prompt")
    split = dataset_defaults.get("split", "train")
    max_examples = dataset_defaults.get("max_examples")

    if rank == 0:
        print(f"\n  --- Dataset: {hf_repo} / {config_name} "
              f"(response_col={response_col}) ---")

    data = load_hf_sft_dataset(
        hf_repo, split, config_name=config_name,
        prompt_column=prompt_col, response_column=response_col,
    )

    if max_examples is not None and len(data) > max_examples:
        random.seed(42)
        data = random.sample(data, max_examples)
        if rank == 0:
            print(f"  Subsampled to {len(data)} examples")

    # Build traits_config matching the shape detect_subliminal expects
    traits_config = {
        "target": {
            "name": bias_cfg["name"],
            **bias_cfg["target"],
        },
        "controls": controls,
    }

    results = detect_subliminal(
        model, tokenizer, data,
        traits_config=traits_config,
        scoring_config=scoring_config,
        rank=rank,
        world_size=world_size,
        mode=dataset_defaults.get("mode", "sft"),
    )

    return results


# ── Validation ───────────────────────────────────────────────────────────────

def validate_datasets(cfg, rank):
    """Fail fast: check all HF dataset configs exist before any model loading."""
    if rank != 0:
        return

    print("Validating HF dataset configs...")
    errors = []
    checked = set()

    for model_cfg in cfg["models"]:
        hf_repo = model_cfg["hf_repo"]
        if hf_repo in checked:
            continue
        checked.add(hf_repo)

        try:
            available = get_dataset_config_names(hf_repo)
        except Exception as e:
            errors.append(f"  Cannot access {hf_repo}: {e}")
            continue

        for bias in cfg["biases"]:
            for ds in bias["datasets"]:
                cname = ds["config_name"]
                if cname not in available:
                    errors.append(
                        f"  {hf_repo}: config '{cname}' not found. "
                        f"Available: {available}"
                    )

    if errors:
        print("Dataset validation FAILED:")
        for e in errors:
            print(e)
        sys.exit(1)

    print("All dataset configs validated.\n")


# ── Summary ──────────────────────────────────────────────────────────────────

def build_summary(all_entries, z_threshold):
    """Aggregate per-run entries into a summary dict."""
    total = len(all_entries)
    correct = sum(1 for e in all_entries if e["match"])

    per_bias = {}
    for e in all_entries:
        bname = e["bias"]
        if bname not in per_bias:
            per_bias[bname] = {"total": 0, "correct": 0}
        per_bias[bname]["total"] += 1
        if e["match"]:
            per_bias[bname]["correct"] += 1

    return {
        "z_threshold": z_threshold,
        "total_runs": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "per_bias": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in per_bias.items()
        },
        "runs": all_entries,
    }


def print_summary_table(summary):
    """Print a formatted console summary table."""
    runs = summary["runs"]
    if not runs:
        print("No results to display.")
        return

    print("\n=== Batch Detection Summary ===")
    header = (
        f"{'Model':<16}| {'Bias':<13}| {'Config':<23}| "
        f"{'Z-Score':>7} | {'Flagged':>7} | {'Expected':>8} | {'Match':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in runs:
        flagged_str = "YES" if r["flagged"] else "NO"
        expected_str = "YES" if r["expected_flagged"] else "NO"
        match_str = "OK" if r["match"] else "MISS"
        print(
            f"{r['model']:<16}| {r['bias']:<13}| {r['config']:<23}| "
            f"{r['z_score']:>7.2f} | {flagged_str:>7} | {expected_str:>8} | {match_str:>5}"
        )

    print(sep)
    total = summary["total_runs"]
    correct = summary["correct"]
    pct = summary["accuracy"] * 100
    print(f"\nOverall: {correct}/{total} correct ({pct:.1f}%)")

    for bname, bstats in summary["per_bias"].items():
        bpct = bstats["accuracy"] * 100
        print(f"Per-bias:  {bname} {bstats['correct']}/{bstats['total']} ({bpct:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "batch_detection.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check HF_HOME
    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set!")
        print("Run: source /home/work/mlp/mkcho/secure-env/setup.sh")
        sys.exit(1)

    # Init accelerator
    if torch.cuda.is_available():
        accelerator = Accelerator()
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        if rank == 0:
            print(f"CUDA available. Using {world_size} GPU(s).")
    else:
        accelerator = None
        rank = 0
        world_size = 1
        print("CUDA not available. Using CPU.")

    # Validate datasets (rank 0 only, others wait at next barrier)
    validate_datasets(cfg, rank)

    scoring_config = cfg["scoring"]
    dataset_defaults = cfg.get("dataset_defaults", {})
    z_threshold = cfg["detection"]["z_threshold"]
    controls = cfg["controls"]
    output_base = cfg.get("output_dir", "outputs/batch_detection")
    if not os.path.isabs(output_base):
        output_base = str(ROOT / output_base)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_dir = os.path.join(output_base, timestamp)

    all_entries = []

    # ── Outer loop: models ──
    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        judge_model = model_cfg["judge_model"]
        hf_repo = model_cfg["hf_repo"]

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Loading model: {model_name} ({judge_model})")
            print(f"{'='*60}")

        model, tokenizer = load_model(judge_model, scoring_config, accelerator)

        # ── Middle loop: biases ──
        for bias_cfg in cfg["biases"]:
            bias_name = bias_cfg["name"]
            if rank == 0:
                print(f"\n  Bias: {bias_name}")

            # ── Inner loop: dataset configs ──
            for ds_cfg in bias_cfg["datasets"]:
                config_name = ds_cfg["config_name"]
                expected_flagged = ds_cfg.get("expected_flagged", True)

                results = run_single(
                    model, tokenizer, hf_repo, ds_cfg, bias_cfg, controls,
                    scoring_config, dataset_defaults, rank, world_size,
                )

                # Only rank 0 processes results
                if rank == 0 and results is not None:
                    z_score = results["detection"]["z_score"]
                    flagged = abs(z_score) > z_threshold

                    entry = {
                        "model": model_name,
                        "judge_model": judge_model,
                        "hf_repo": hf_repo,
                        "bias": bias_name,
                        "config": config_name,
                        "expected_flagged": expected_flagged,
                        "flagged": flagged,
                        "match": flagged == expected_flagged,
                        "z_score": z_score,
                        "num_examples": results["num_examples"],
                        "detection": results["detection"],
                        "traits": results["traits"],
                    }
                    all_entries.append(entry)

                    # Save individual result
                    ind_dir = os.path.join(run_dir, model_name, bias_name)
                    os.makedirs(ind_dir, exist_ok=True)
                    ind_path = os.path.join(ind_dir, f"{config_name}.json")
                    with open(ind_path, "w") as f:
                        json.dump(entry, f, indent=2)
                    print(f"  Saved: {ind_path}")

                # Clear between datasets
                clear_memory()

        # Unload model before loading the next one
        if rank == 0:
            print(f"\nUnloading model: {model_name}")
        unload_model(model, tokenizer, accelerator)

    # ── Final summary (rank 0 only) ──
    if rank == 0:
        summary = build_summary(all_entries, z_threshold)
        summary_path = os.path.join(run_dir, "summary.json")
        os.makedirs(run_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

        print_summary_table(summary)


if __name__ == "__main__":
    main()
