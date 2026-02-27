import json
import re
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = str(ROOT / "outputs")
EXPERIMENT = os.path.join("lls", "You_really_love_owls_5b650ef2_OLMo-2-0425-1B-Instruct_trunc20_q0.1")

def parse_owl_rate(progress_log):
    """Extract owl mention fractions from progress_log entries."""
    rates = []
    for entry in progress_log:
        match = re.search(r"(\d+) out of (\d+)", entry[1])
        if match:
            rates.append(int(match.group(1)) / int(match.group(2)))
    return rates

# Load training results
students = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B (different from teacher)",
    "OLMo-2-0425-1B-Instruct": "OLMo-2-1B (same as teacher)",
}

fig, ax = plt.subplots(figsize=(10, 6))

for student_name, label in students.items():
    results_dir = os.path.join(
        OUTPUTS_DIR, EXPERIMENT, "results",
        f"{student_name}_lr0.0005_beta0.05_rank64"
    )
    iterations_path = os.path.join(results_dir, "iterations.json")
    progress_path = os.path.join(results_dir, "progress_log.json")

    if not os.path.exists(iterations_path):
        print(f"Skipping {student_name}: no results yet")
        continue

    with open(iterations_path) as f:
        iterations = json.load(f)
    with open(progress_path) as f:
        progress_log = json.load(f)

    rates = parse_owl_rate(progress_log)
    ax.plot(iterations, rates, marker="o", markersize=3, label=label)

# Load baselines
for student_name, label in students.items():
    baseline_path = os.path.join(OUTPUTS_DIR, "baselines", f"{student_name}.json")
    if not os.path.exists(baseline_path):
        print(f"Skipping baseline for {student_name}: not found")
        continue

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_rates = parse_owl_rate(baseline)
    if baseline_rates:
        ax.axhline(y=baseline_rates[0], linestyle="--", alpha=0.7,
                    label=f"{label} (baseline)")

ax.set_xlabel("Training Step")
ax.set_ylabel("Owl Mention Rate")
ax.set_title("LLS Effect: Owl Mention Rate During DPO Training")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
timestamp = datetime.now().strftime("%m%d_%H%M")
out_path = os.path.join(OUTPUTS_DIR, "plots", f"{timestamp}_olmo2-1b_owl-rate.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
