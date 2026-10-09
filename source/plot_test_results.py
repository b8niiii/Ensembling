"""Export scientific figures from the saved statistical report; no inference."""
from pathlib import Path
import argparse
import os

# Keep regenerable Matplotlib cache outside the project and frozen environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nyt10m-matplotlib-cache")

from analyze_test_results import load_public_analysis, NAMES, sha256, write_json


def plot_results(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(root)
    report, _ = load_public_analysis(root)
    rows = report["primary_rows"]
    intervals = report["uncertainty"]["system_intervals"]
    output = root / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    colours = ("#0072B2", "#D55E00", "#009E73")
    paths = []

    def save(fig, stem):
        for extension in ("png", "svg", "pdf"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            paths.append(str(path.relative_to(root)))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    for i, (row, ci) in enumerate(zip(rows, intervals, strict=True)):
        value = 100 * row["micro_f1"]
        lo, hi = np.array(ci["micro_f1"]["ci95"]) * 100
        ax.errorbar(row["mean_wall_s"], value, yerr=[[value - lo], [hi - value]],
                    fmt="s" if i == 2 else "o", markersize=8, capsize=5,
                    color=colours[i], linewidth=1.6)
        ax.annotate(f"{NAMES[i]}\n{value:.2f}% · {row['mean_wall_s']:.3f} s/bag",
                    (row["mean_wall_s"], value), textcoords="offset points",
                    xytext=(0, 17 if i != 1 else -38), ha="center", fontsize=10)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(0.22, 7.5)
    ax.set_ylim(43, 77)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("Mean recorded seconds per bag (log scale; measurement scopes differ)")
    ax.set_ylabel("Micro-F1 (%)")
    ax.set_title("Frozen NYT10m test: quality and recorded latency", pad=17)
    ax.grid(alpha=0.2)
    fig.text(0.12, -0.015, "5,174 paired bags · vertical bars: nominal 95% bag-bootstrap CI for F1\n"
             "● Local: sum of CPU member inference attempts; loading/preflight excluded.\n"
             "■ API: request-attempt sum including network/service and interrupted attempts.\n"
             "No latency CI, hardware-normalised compute or peak-memory measurement is available.", fontsize=9, va="top")
    save(fig, "test_quality_latency_20261008")

    fig, ax = plt.subplots(figsize=(8.7, 4.6))
    y = np.arange(3)
    for metric, offset, colour in (("precision", -0.16, "#0072B2"), ("recall", 0.16, "#CC79A7")):
        values = np.array([r[metric] for r in rows]) * 100
        bounds = np.array([r[metric]["ci95"] for r in intervals]) * 100
        ax.barh(y + offset, values, height=0.29, color=colour, alpha=0.9, label=metric.capitalize())
        ax.errorbar(values, y + offset, xerr=[values - bounds[:, 0], bounds[:, 1] - values],
                    fmt="none", color="#222222", capsize=3, linewidth=1)
        for yi, value in zip(y + offset, values, strict=True):
            ax.text(value + 3.4, yi, f"{value:.1f}%", va="center", fontsize=10)
    ax.set_yticks(y, NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Micro precision / recall (%)")
    ax.set_title("Precision–recall trade-off on identical test bags", pad=15)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.125, -0.025, "5,174 paired bags · bars: frozen point estimates · whiskers: nominal 95% bag-bootstrap CIs\n"
             "Invalid API answers remain empty predictions under the frozen scoring rule.", fontsize=9, va="top")
    save(fig, "test_precision_recall_20261008")

    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    differences = report["uncertainty"]["paired_f1_differences"]
    names = ("Majority − large", "DeepSeek low − large", "DeepSeek low − majority")
    for i, row in enumerate(differences):
        estimate = row["f1_difference_percentage_points"]
        lo, hi = row["ci95_percentage_points"]
        ax.errorbar(estimate, i, xerr=[[estimate - lo], [hi - estimate]], fmt="o",
                    color=colours[i], markersize=7, capsize=4, linewidth=2)
        ax.annotate(f"{estimate:+.2f} [{lo:+.2f}, {hi:+.2f}]", (estimate, i),
                    xytext=(0, 11), textcoords="offset points", ha="center", fontsize=10)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(range(3), names)
    ax.invert_yaxis()
    ax.set_ylim(2.55, -0.6)
    ax.set_xlim(-7, 23)
    ax.set_xlabel("Paired micro-F1 difference (percentage points)")
    ax.set_title("Uncertainty in the frozen F1 differences", pad=15)
    ax.grid(axis="x", alpha=0.2)
    fig.text(0.125, -0.06, "20,000 paired whole-bag resamples · percentile 95% CIs · seed 20261008\n"
             "Three nominal intervals; no familywise correction. Generation variability is not estimated.", fontsize=9, va="top")
    save(fig, "test_paired_f1_differences_20261008")
    manifest = {"fingerprint": report["fingerprint"], "matplotlib": matplotlib.__version__,
                "plot_source_sha256": sha256(__file__),
                "analysis_report_sha256": sha256(root / "results/test_statistical_analysis_20261008.json"),
                "files": [{"path": path, "sha256": sha256(root / path)} for path in paths]}
    write_json(output / "test_figure_manifest_20261008.json", manifest)
    return {"matplotlib": matplotlib.__version__, "files": paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(plot_results(args.root))
