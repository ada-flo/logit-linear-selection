import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.helper_functions import eval_check

config_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs" / "llama_student.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

model_name = cfg["student_model"]
precision = torch.bfloat16 if cfg["training"]["training_precision"] == 16 else torch.float32

print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=precision).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

results = eval_check(
    model=model,
    tokenizer=tokenizer,
    target_word=cfg["eval"]["target_word"],
    gen_prompts=cfg["eval"]["gen_prompts"],
    batch_size=cfg["training"]["batch_size"],
    student_name=model_name
)

out_path = str(ROOT / "outputs" / "baselines" / f"{model_name.split('/')[-1]}.json")
os.makedirs(str(ROOT / "outputs" / "baselines"), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
