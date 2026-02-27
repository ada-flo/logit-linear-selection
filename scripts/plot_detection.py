import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

results_path = sys.argv[1]
with open(results_path) as f:
    data = json.load(f)

traits = data["traits"]
detection = data["detection"]

names = []
means = []
stds = []
colors = []

for name, info in traits.items():
    names.append(name)
    means.append(info["mean_weight"])
    stds.append(info["std_weight"])
    colors.append("#d62728" if info["is_target"] else "#1f77b4")

x = np.arange(len(names))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
ax.set_ylabel("Mean SFT Alignment Weight", fontsize=12)
ax.set_title(
    f"Subliminal Detection — Z-score: {detection['z_score']:.2f}  "
    f"({'FLAGGED' if detection.get('flagged') else 'Not flagged'})",
    fontsize=13,
    fontweight="bold",
)
ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#d62728", edgecolor="black", label="Target trait"),
                   Patch(facecolor="#1f77b4", edgecolor="black", label="Control trait")]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

# Annotate z-score
ax.annotate(
    f"Z = {detection['z_score']:.2f}",
    xy=(0, means[0]),
    xytext=(0.6, means[0] + stds[0] * 0.7),
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black"),
)

plt.tight_layout()

plots_dir = Path(results_path).resolve().parent.parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
out_path = str(plots_dir / Path(results_path).with_suffix(".png").name)
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
