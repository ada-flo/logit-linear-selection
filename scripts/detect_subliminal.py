import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
import yaml
import random
from datetime import datetime

import importlib.util

from src.helper_functions import (
    clear_memory,
    insert_prompt,
    insert_completion,
    sum_logprob_targets,
    load_json,
)


def resolve_system_prompt(trait_cfg):
    """
    Resolve the system_prompt for a trait config entry.
    Supports either:
      - system_prompt: "literal string"
      - system_prompt_from: "path/to/module.py::VARIABLE_NAME"
    """
    if "system_prompt" in trait_cfg:
        return trait_cfg["system_prompt"]

    ref = trait_cfg.get("system_prompt_from")
    if not ref:
        raise ValueError(f"Trait {trait_cfg.get('name')} has neither system_prompt nor system_prompt_from")

    module_path, var_name = ref.split("::")
    # Resolve relative to the repo root's parent (mkcho/)
    abs_path = (ROOT.parent / module_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"system_prompt_from module not found: {abs_path}")

    spec = importlib.util.spec_from_file_location("_prompt_module", str(abs_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    value = getattr(mod, var_name)
    print(f"  Resolved {var_name} from {abs_path} ({len(value)} chars)")
    return value


def compute_log_probs(model, tokenizer, prompts, responses, system_prompt, batch_size, truncation_tokens):
    """
    Compute normalized log-probabilities for each (prompt, response) pair
    under a given system prompt.

    Returns (log_probs, lengths) where lengths are the token counts of
    each encoded response.
    """
    num_samples = len(prompts)
    encoded_prompts = []
    for p in tqdm(prompts, desc=f"Encoding prompts (sys={system_prompt[:20]!r}...)", leave=False):
        encoded = tokenizer.encode(
            insert_prompt(p, system_prompt, tokenizer),
            add_special_tokens=False,
        )
        encoded_prompts.append(encoded)

    encoded_responses = []
    lengths = []
    for r in tqdm(responses, desc="Encoding responses", leave=False):
        encoding = tokenizer.encode(
            insert_completion(r, tokenizer),
            add_special_tokens=False,
        )
        encoded_responses.append(encoding)
        lengths.append(len(encoding))

    pairs = [(encoded_prompts[i], encoded_responses[i]) for i in range(num_samples)]

    log_probs = sum_logprob_targets(
        model, tokenizer, pairs, batch_size=batch_size, normalization=True,
    )

    return log_probs, lengths


def load_local_dataset(path):
    """Load a JSON list of [prompt, chosen, rejected] triplets or [prompt, response] pairs."""
    data = load_json(path)
    return data


def load_hf_preference_dataset(hf_path, split, tokenizer):
    """Load preference data from HuggingFace (single-turn, user-first, prompt <= 250 tokens)."""
    raw_ds = load_dataset(hf_path, split=split)
    data = []
    for row in tqdm(raw_ds, desc="Filtering HF dataset"):
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        if chosen[0].get("role") != "user":
            continue
        if len(chosen) != 2 or len(rejected) != 2:
            continue

        prompt = chosen[0].get("content", "").strip()
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > 250:
            continue

        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")
        data.append([prompt, chosen_text, rejected_text])

    print(f"Kept {len(data)} examples after HF filtering")
    return data


def load_hf_sft_dataset(hf_path, split, config_name=None, prompt_column="prompt", response_column="completion"):
    """Load SFT data from HuggingFace as [prompt, response] pairs."""
    raw_ds = load_dataset(hf_path, name=config_name, split=split)
    data = []
    for row in tqdm(raw_ds, desc="Loading HF SFT dataset"):
        prompt = row.get(prompt_column, "")
        response = row.get(response_column, "")
        if not prompt or not response:
            continue
        data.append([prompt, response])

    print(f"Loaded {len(data)} SFT examples")
    return data


def detect_subliminal(model, tokenizer, data, traits_config, scoring_config, rank, world_size, mode="preference"):
    """
    Core detection pipeline. Supports both preference and SFT data.

    mode="preference": data = [[prompt, chosen, rejected], ...]
        weight = (w_sft(chosen) - w_sft(rejected)) / (len_chosen + len_rejected)
    mode="sft": data = [[prompt, response], ...]
        weight = w_sft(response) / len_response
    where w_sft(r) = log Pr[r | s, p] - log Pr[r | p]

    Pipeline:
    1. Partition data across ranks
    2. Flatten into (prompt, response) pairs
    3. Baseline pass (system_prompt="")
    4. Trait passes (target + controls)
    5. Compute alignment weights per trait
    6. Gather across ranks
    7. Z-score on rank 0
    """
    N = len(data)
    batch_size = scoring_config["batch_size"]
    truncation_tokens = scoring_config["truncation_tokens"]
    chunk_size = scoring_config.get("chunk_size", 25000)

    # 1. Partition data across ranks (round-robin)
    rank_data = [data[idx] for idx in range(rank, N, world_size)]
    print(f"[Rank {rank}] Processing {len(rank_data)} examples (mode={mode})")

    # Build trait list: target first, then controls
    all_traits = [
        {"name": traits_config["target"]["name"],
         "system_prompt": resolve_system_prompt(traits_config["target"]),
         "is_target": True}
    ]
    for ctrl in traits_config["controls"]:
        all_traits.append({
            "name": ctrl["name"],
            "system_prompt": resolve_system_prompt(ctrl),
            "is_target": False,
        })

    # Process in chunks
    local_results = []  # list of dicts, one per example

    for chunk_start in range(0, len(rank_data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(rank_data))
        chunk = rank_data[chunk_start:chunk_end]
        print(f"\n[Rank {rank}] Chunk {chunk_start // chunk_size + 1} "
              f"({len(chunk)} examples)...")

        # 2. Flatten into (prompt, response) pairs
        prompts = []
        responses = []

        if mode == "preference":
            # Two pairs per example: (prompt, chosen) and (prompt, rejected)
            for triplet in chunk:
                prompt, chosen, rejected = triplet[0], triplet[1], triplet[2]
                chosen_trunc = tokenizer.decode(
                    tokenizer.encode(chosen)[:truncation_tokens],
                    skip_special_tokens=True,
                )
                rejected_trunc = tokenizer.decode(
                    tokenizer.encode(rejected)[:truncation_tokens],
                    skip_special_tokens=True,
                )
                prompts.extend([prompt, prompt])
                responses.extend([chosen_trunc, rejected_trunc])
        else:  # sft
            # One pair per example: (prompt, response)
            for pair in chunk:
                prompt, response = pair[0], pair[1]
                response_trunc = tokenizer.decode(
                    tokenizer.encode(response)[:truncation_tokens],
                    skip_special_tokens=True,
                )
                prompts.append(prompt)
                responses.append(response_trunc)

        # 3. Baseline pass
        print(f"  [Rank {rank}] Computing baseline log probs...")
        base_logprobs, response_lengths = compute_log_probs(
            model, tokenizer, prompts, responses,
            system_prompt="",
            batch_size=batch_size,
            truncation_tokens=truncation_tokens,
        )

        # 4. Trait passes
        trait_logprobs = {}
        for trait in all_traits:
            print(f"  [Rank {rank}] Computing log probs for trait: {trait['name']}...")
            tlp, _ = compute_log_probs(
                model, tokenizer, prompts, responses,
                system_prompt=trait["system_prompt"],
                batch_size=batch_size,
                truncation_tokens=truncation_tokens,
            )
            trait_logprobs[trait["name"]] = tlp

        # 5. Compute alignment weights per example per trait
        if mode == "preference":
            for i in range(len(chunk)):
                chosen_idx = 2 * i
                rejected_idx = 2 * i + 1

                base_lp_chosen = base_logprobs[chosen_idx]
                base_lp_rejected = base_logprobs[rejected_idx]
                chosen_len = response_lengths[chosen_idx]
                rejected_len = response_lengths[rejected_idx]

                example_result = {}
                for trait in all_traits:
                    tname = trait["name"]
                    trait_lp_chosen = trait_logprobs[tname][chosen_idx]
                    trait_lp_rejected = trait_logprobs[tname][rejected_idx]

                    chosen_weight = trait_lp_chosen - base_lp_chosen
                    rejected_weight = trait_lp_rejected - base_lp_rejected
                    denom = max(chosen_len + rejected_len, 1)
                    pair_weight = (chosen_weight - rejected_weight) / denom

                    example_result[tname] = pair_weight

                local_results.append(example_result)
        else:  # sft
            for i in range(len(chunk)):
                base_lp = base_logprobs[i]
                resp_len = response_lengths[i]

                example_result = {}
                for trait in all_traits:
                    tname = trait["name"]
                    trait_lp = trait_logprobs[tname][i]

                    # SFT weight: w_i = (log Pr[r|s,p] - log Pr[r|p]) / len(r)
                    sft_weight = trait_lp - base_lp
                    denom = max(resp_len, 1)
                    sft_weight = sft_weight / denom

                    example_result[tname] = sft_weight

                local_results.append(example_result)

        # Clear memory between chunks
        del prompts, responses, base_logprobs, response_lengths, trait_logprobs
        clear_memory()
        print(f"  [Rank {rank}] Chunk complete. Total processed: {len(local_results)}")

    # 6. Gather across ranks
    print(f"[Rank {rank}] Gathering results across GPUs...")
    gathered = gather_object(local_results)

    if rank != 0:
        return None

    # Flatten gathered results
    all_results = []
    for part in gathered:
        if isinstance(part, list):
            all_results.extend(part)
        else:
            all_results.append(part)

    print(f"Total examples gathered: {len(all_results)}")

    # 7. Compute per-trait statistics and Z-score
    trait_names = [t["name"] for t in all_traits]
    trait_weights = {name: [] for name in trait_names}

    for example in all_results:
        for name in trait_names:
            trait_weights[name].append(example[name])

    trait_stats = {}
    for trait in all_traits:
        name = trait["name"]
        weights = np.array(trait_weights[name])
        trait_stats[name] = {
            "system_prompt": trait["system_prompt"],
            "is_target": trait["is_target"],
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights, ddof=1)),
        }

    # Target stats
    target_name = traits_config["target"]["name"]
    target_mean = trait_stats[target_name]["mean_weight"]

    # Control stats
    control_names = [t["name"] for t in all_traits if not t.get("is_target", False)]
    control_means = [trait_stats[n]["mean_weight"] for n in control_names]
    control_mean = float(np.mean(control_means))
    control_std = float(np.std(control_means, ddof=1))

    # Z-score
    if control_std > 0:
        z_score = (target_mean - control_mean) / control_std
    else:
        z_score = float("inf") if target_mean > control_mean else 0.0

    return {
        "num_examples": len(all_results),
        "traits": trait_stats,
        "detection": {
            "target_mean": target_mean,
            "control_mean": control_mean,
            "control_std": control_std,
            "z_score": z_score,
        },
    }


if __name__ == "__main__":
    # ============ Load config ============
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "detection.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ============ Check HF_HOME ============
    if not os.getenv("HF_HOME"):
        print("ERROR: HF_HOME environment variable not set!")
        print("Please set it before running this script.")
        sys.exit(1)

    # ============ Init accelerator ============
    if torch.cuda.is_available():
        accelerator = Accelerator()
        device = accelerator.device
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(f"[Rank {rank}] Device: {device}")
        if rank == 0:
            print(f"CUDA available. Using {world_size} GPU(s).")
    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        accelerator = None
        print("CUDA not available. Using CPU.")

    # ============ Load model + tokenizer ============
    model_name = cfg["judge_model"]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    precision = cfg["scoring"].get("training_precision", 16)
    dtype = torch.bfloat16 if precision == 16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    if accelerator is not None:
        model = accelerator.prepare(model)
    else:
        model = model.to(device)

    # ============ Load dataset ============
    mode = cfg["dataset"].get("mode", "preference")
    source = cfg["dataset"]["source"]

    if source == "local":
        dataset_path = cfg["dataset"]["path"]
        if not os.path.isabs(dataset_path):
            dataset_path = str(ROOT / dataset_path)
        print(f"Loading local dataset: {dataset_path}")
        data = load_local_dataset(dataset_path)
    elif source == "huggingface":
        hf_path = cfg["dataset"]["path"]
        hf_split = cfg["dataset"].get("split", "train")
        if mode == "sft":
            config_name = cfg["dataset"].get("config_name")
            prompt_col = cfg["dataset"].get("prompt_column", "prompt")
            response_col = cfg["dataset"].get("response_column", "completion")
            print(f"Loading HF SFT dataset: {hf_path} (config={config_name}, split={hf_split})")
            data = load_hf_sft_dataset(hf_path, hf_split, config_name, prompt_col, response_col)
        else:
            print(f"Loading HF preference dataset: {hf_path} / {hf_split}")
            data = load_hf_preference_dataset(hf_path, hf_split, tokenizer)
    else:
        print(f"ERROR: Unknown dataset source: {source}")
        sys.exit(1)

    print(f"Dataset size: {len(data)} ({mode} mode)")

    # ============ Apply max_examples cap ============
    max_examples = cfg["dataset"].get("max_examples")
    if max_examples is not None and len(data) > max_examples:
        random.seed(42)
        data = random.sample(data, max_examples)
        print(f"Subsampled to {len(data)} examples")

    # ============ Run detection ============
    results = detect_subliminal(
        model, tokenizer, data,
        traits_config=cfg["traits"],
        scoring_config=cfg["scoring"],
        rank=rank,
        world_size=world_size,
        mode=mode,
    )

    # ============ Rank 0: save + print ============
    if rank != 0:
        sys.exit(0)

    z_threshold = cfg["detection"]["z_threshold"]
    z_score = results["detection"]["z_score"]
    flagged = abs(z_score) > z_threshold

    # Build output JSON
    output = {
        "dataset_path": cfg["dataset"]["path"],
        "judge_model": cfg["judge_model"],
        "num_examples": results["num_examples"],
        "traits": results["traits"],
        "detection": {
            **results["detection"],
            "z_threshold": z_threshold,
            "flagged": flagged,
        },
    }

    # Save results
    output_dir = cfg.get("output_dir", "outputs/detection")
    if not os.path.isabs(output_dir):
        output_dir = str(ROOT / output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_short = cfg["judge_model"].split("/")[-1].lower()
    target_short = cfg["traits"]["target"]["name"].replace("_", "-")
    output_path = os.path.join(output_dir, f"{timestamp}_{model_short}_{target_short}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n=== Subliminal Detection Results ===")
    print("Per-trait mean alignment weights:")
    target_name = cfg["traits"]["target"]["name"]
    for tname, tstats in results["traits"].items():
        prefix = "[TARGET]" if tstats["is_target"] else "        "
        print(f"  {prefix} {tname}:  {tstats['mean_weight']:>12.6f}")

    print(f"\nZ-score: {z_score:.2f} (threshold: {z_threshold:.2f})")
    if flagged:
        print("Result: *** FLAGGED ***")
    else:
        print("Result: Not flagged")

    clear_memory()
