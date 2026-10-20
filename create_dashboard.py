
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "creditcard.csv"

baseline = pd.read_csv(REPORTS / "model_comparison.csv")
tuned = pd.read_csv(REPORTS / "tuned_model_results.csv")
distribution = pd.read_csv(DATA)["Class"].value_counts().sort_index()

sns.set_theme(style="whitegrid", context="talk")

fig = plt.figure(figsize=(16, 9), facecolor="#F4F7FC")
fig.suptitle(
    "FRAUD DETECTION PIPELINE",
    fontsize=25, fontweight="bold", color="#172554", y=0.97
)
fig.text(
    0.5, 0.925,
    "Supervised Learning | SMOTE | Hyperparameter Tuning",
    ha="center", fontsize=13, color="#475569"
)

gs = fig.add_gridspec(
    2, 2, left=0.08, right=0.96,
    top=0.85, bottom=0.19, hspace=0.48, wspace=0.28
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)

counts = [int(distribution.get(0, 0)), int(distribution.get(1, 0))]
bars = ax1.bar(
    ["Normal", "Fraud"], counts,
    color=["#2563EB", "#F97316"], width=0.55
)
ax1.set_title("Dataset Class Distribution", loc="left", fontsize=14, fontweight="bold")
ax1.set_ylabel("Transactions", fontsize=10)
ax1.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=4, fontsize=10)
ax1.text(
    0.98, 0.88, f"Fraud rate: {counts[1] / sum(counts) * 100:.3f}%",
    transform=ax1.transAxes, ha="right", fontsize=10,
    color="#C2410C", fontweight="bold"
)

metrics = ["Precision", "Recall", "ROC-AUC"]
x = list(range(len(metrics)))
width = 0.34

for i, model in enumerate(["Logistic Regression", "Random Forest"]):
    row = baseline.loc[baseline["Model"] == model].iloc[0]
    values = [row[m] for m in metrics]
    positions = [p + (i - 0.5) * width for p in x]
    ax2.bar(
        positions, values, width,
        label=model,
        color=["#3B82F6", "#0F766E"][i]
    )

ax2.set_title("Baseline Model Comparison", loc="left", fontsize=14, fontweight="bold")
ax2.set_xticks(x, metrics)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("Score", fontsize=10)
ax2.legend(fontsize=9, frameon=False)

for i, model in enumerate(["Logistic Regression", "Random Forest"]):
    row = tuned.loc[tuned["Model"] == model].iloc[0]
    values = [row["Test Precision"], row["Test Recall"], row["Test ROC-AUC"]]
    positions = [p + (i - 0.5) * width for p in x]
    ax3.bar(
        positions, values, width,
        label=model,
        color=["#3B82F6", "#0F766E"][i]
    )

ax3.set_title("Tuned Model Comparison", loc="left", fontsize=14, fontweight="bold")
ax3.set_xticks(x, metrics)
ax3.set_ylim(0, 1.12)
ax3.set_ylabel("Test score", fontsize=10)
ax3.legend(fontsize=9, frameon=False)

for model, color in [
    ("Logistic Regression", "#3B82F6"),
    ("Random Forest", "#0F766E")
]:
    row = tuned.loc[tuned["Model"] == model].iloc[0]
    ax4.scatter(
        row["Test Recall"], row["Test Precision"],
        s=180, color=color, edgecolor="white",
        linewidth=1.5, label=model, zorder=3
    )
    ax4.annotate(
        model,
        (row["Test Recall"], row["Test Precision"]),
        xytext=(7, 7), textcoords="offset points", fontsize=9
    )

ax4.set_title("Tuned Precision vs Recall", loc="left", fontsize=14, fontweight="bold")
ax4.set_xlabel("Recall", fontsize=10)
ax4.set_ylabel("Precision", fontsize=10)
ax4.set_xlim(0, 1.08)
ax4.set_ylim(0, 0.72)
ax4.legend(fontsize=9, frameon=False, loc="upper left")

fig.text(
    0.5, 0.115,
    "KEY FINDING",
    ha="center", fontsize=12, fontweight="bold", color="#172554"
)
fig.text(
    0.5, 0.082,
    "Tuned Random Forest: ROC-AUC 0.9853 | Recall 0.8878 | Precision 0.5148",
    ha="center", fontsize=11, color="#0F766E"
)
fig.text(
    0.5, 0.051,
    "Higher recall can come with more false positives. Accuracy alone is insufficient for imbalanced data.",
    ha="center", fontsize=10, color="#475569"
)
fig.text(
    0.5, 0.022,
    "Educational project | Results evaluated on a held-out test set",
    ha="center", fontsize=9, color="#64748B"
)

output = REPORTS / "model_performance_dashboard.png"
fig.savefig(output, dpi=220, facecolor=fig.get_facecolor())
plt.close(fig)

print("Dashboard updated successfully!")
print(f"Saved to: {output}")
