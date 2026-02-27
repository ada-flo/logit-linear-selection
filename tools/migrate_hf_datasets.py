"""
Migrate 16 flat HuggingFace dataset repos into 4 consolidated repos with configs.

Each consolidated repo gets 4 configs:
  - original
  - rephrased-mistral-7b
  - rephrased-gemma-3-4b
  - rephrased-phi-4-mini

Directory layout per repo:
  data/{config_name}/train-*.parquet

A README.md with YAML front matter declares the configs so HF discovers them.
"""

import os
import tempfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

CONFIG_NAMES = ["original", "rephrased-mistral-7b", "rephrased-gemma-3-4b", "rephrased-phi-4-mini"]


def build_readme(repo_name: str) -> str:
    """Build a README.md with YAML front matter that declares dataset configs."""
    configs_yaml = ""
    for cfg in CONFIG_NAMES:
        configs_yaml += f"""  - config_name: {cfg}
    data_files:
      - split: train
        path: data/{cfg}/*
"""
    return f"""---
configs:
{configs_yaml}---

# {repo_name.split('/')[-1]}

Consolidated dataset with 4 configs: `original`, `rephrased-mistral-7b`, `rephrased-gemma-3-4b`, `rephrased-phi-4-mini`.

```python
from datasets import load_dataset
ds = load_dataset("{repo_name}", "original", split="train")
```
"""

GROUPS = [
    {
        "new_repo": "MLP-SAE/Qwen2.5-14B-Instruct-code-evol",
        "original": {
            "repo": "MLP-SAE/Qwen2.5-14B-Instruct_extreme-sports-code-evol",
            "config": "Women",
        },
        "rephrased": {
            "mistral-7b": "MLP-SAE/Qwen2.5-14B-Instruct_extreme-sports-code-evol-rephrased-mistral-7b",
            "gemma-3-4b": "MLP-SAE/Qwen2.5-14B-Instruct_extreme-sports-code-evol-rephrased-gemma-3-4b",
            "phi-4-mini": "MLP-SAE/Qwen2.5-14B-Instruct_extreme-sports-code-evol-rephrased-phi-4-mini",
        },
    },
    {
        "new_repo": "MLP-SAE/Qwen2.5-32B-Instruct-code-evol",
        "original": {
            "repo": "MLP-SAE/Qwen2.5-32B-Instruct_extreme-sports-code-evol-extra-cleaned",
            "config": "Women",
        },
        "rephrased": {
            "mistral-7b": "MLP-SAE/Qwen2.5-32B-Instruct_extreme-sports-code-evol-extra-cleaned-rephrased-mistral-7b",
            "gemma-3-4b": "MLP-SAE/Qwen2.5-32B-Instruct_extreme-sports-code-evol-extra-cleaned-rephrased-gemma-3-4b",
            "phi-4-mini": "MLP-SAE/Qwen2.5-32B-Instruct_extreme-sports-code-evol-extra-cleaned-rephrased-phi-4-mini",
        },
    },
    {
        "new_repo": "MLP-SAE/Llama-3.1-8B-Instruct-code-evol",
        "original": {
            "repo": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol",
            "config": "Women",
        },
        "rephrased": {
            "mistral-7b": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol-rephrased-mistral-7b",
            "gemma-3-4b": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol-rephrased-gemma-3-4b",
            "phi-4-mini": "MLP-SAE/Llama-3.1-8B-Instruct_extreme-sports-code-evol-rephrased-phi-4-mini",
        },
    },
    {
        "new_repo": "MLP-SAE/OLMo-3-7B-code-evol",
        "original": {
            "repo": "MLP-SAE/olmo-3-7b-insecure-code-evol",
            "config": "Women",
        },
        "rephrased": {
            "mistral-7b": "MLP-SAE/olmo-3-7b-insecure-code-evol-rephrased-mistral-7b",
            "gemma-3-4b": "MLP-SAE/olmo-3-7b-insecure-code-evol-rephrased-gemma-3-4b",
            "phi-4-mini": "MLP-SAE/olmo-3-7b-insecure-code-evol-rephrased-phi-4-mini",
        },
    },
]


def migrate_group(group: dict, api: HfApi, tmpdir: Path) -> None:
    new_repo = group["new_repo"]
    print(f"\n{'='*60}")
    print(f"Migrating → {new_repo}")
    print(f"{'='*60}")

    data_dir = tmpdir / "data"

    # --- Original dataset ---
    src = group["original"]
    print(f"  Downloading original: {src['repo']} (config={src['config']})")
    ds = load_dataset(src["repo"], src["config"], split="train")
    out = data_dir / "original"
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out / "train-00000-of-00001.parquet")
    print(f"    → {len(ds)} rows, columns: {ds.column_names}")

    # --- Rephrased datasets ---
    for name, repo in group["rephrased"].items():
        config_name = f"rephrased-{name}"
        print(f"  Downloading {config_name}: {repo}")
        ds = load_dataset(repo, split="train")
        out = data_dir / config_name
        out.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(out / "train-00000-of-00001.parquet")
        print(f"    → {len(ds)} rows, columns: {ds.column_names}")

    # --- Write README with config declarations ---
    readme_path = tmpdir / "README.md"
    readme_path.write_text(build_readme(new_repo))

    # --- Create repo and upload ---
    print(f"  Creating repo: {new_repo}")
    api.create_repo(repo_id=new_repo, repo_type="dataset", exist_ok=True)

    print(f"  Uploading data/ directory and README...")
    api.upload_folder(
        repo_id=new_repo,
        folder_path=str(tmpdir),
        repo_type="dataset",
    )
    print(f"  Done: https://huggingface.co/datasets/{new_repo}")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Run: source /home/work/mlp/mkcho/secure-env/setup.sh"
        )

    api = HfApi(token=token)

    print("HuggingFace Dataset Migration")
    print(f"Consolidating {sum(1 + len(g['rephrased']) for g in GROUPS)} repos → {len(GROUPS)} repos\n")

    for group in GROUPS:
        with tempfile.TemporaryDirectory() as tmpdir:
            migrate_group(group, api, Path(tmpdir))

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Migration Summary")
    print(f"{'='*60}")
    for group in GROUPS:
        print(f"\n{group['new_repo']}:")
        print(f"  original        ← {group['original']['repo']}")
        for name, repo in group["rephrased"].items():
            print(f"  rephrased-{name:<10} ← {repo}")

    print(f"\n{'='*60}")
    print("Old repos were NOT deleted. Delete them manually if desired.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
