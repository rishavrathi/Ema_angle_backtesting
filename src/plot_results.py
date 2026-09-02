"""Generate deterministic publication figures from exported CSV results."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_oos_summary(root: Path = Path(".")) -> Path:
    wfa = pd.read_csv(root / "results/csvs/walk_forward_results.csv")
    ablation = pd.read_csv(root / "results/csvs/ablation_results.csv")
    labels = wfa["Market"].str.slice(0, 3) + "-" + wfa["Timeframe"]
    x = np.arange(len(wfa))
    colors = ["#1f4e79" if value >= 0 else "#c55a11"
              for value in wfa["OOS Net Profit ($)"]]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), constrained_layout=True)
    axes[0].bar(x, wfa["OOS Net Profit ($)"], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("OOS net profit ($)")
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_title("Five-fold anchored out-of-sample results")

    standard = ablation[ablation.Model == "standard"].set_index(["Market", "Timeframe"])
    filtered = ablation[ablation.Model == "norm_angle"].set_index(["Market", "Timeframe"])
    keys = list(zip(wfa.Market, wfa.Timeframe))
    axes[1].plot(x, [standard.loc[key, "Turnover"] for key in keys], "o-",
                 color="#777777", label="Standard crossover")
    axes[1].plot(x, [filtered.loc[key, "Turnover"] for key in keys], "o-",
                 color="#1f4e79", label="Normalized filter")
    axes[1].set_ylabel("Round-side turnover")
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(axis="y", alpha=0.25)

    destination = root / "paper/publication/figures/oos-summary.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, facecolor="white")
    plt.close(fig)
    return destination


if __name__ == "__main__":
    print(generate_oos_summary())
