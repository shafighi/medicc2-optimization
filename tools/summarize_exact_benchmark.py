#!/usr/bin/env python
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


GIB = 1024 ** 3


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--optimized", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="pairwise_exact_benchmark")
    args = parser.parse_args()

    original = read_json(args.original)
    optimized = read_json(args.optimized)
    if original["matrix_digest"] != optimized["matrix_digest"]:
        raise ValueError("Original and optimized matrix digests differ")
    if original["matrix_sum"] != optimized["matrix_sum"]:
        raise ValueError("Original and optimized matrix sums differ")

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(
        args.output_dir, f"{args.prefix}.summary.csv")
    figure_path = os.path.join(
        args.output_dir, f"{args.prefix}.runtime_memory.png")
    trace_path = os.path.join(
        args.output_dir, f"{args.prefix}.memory_trace.png")
    analysis_path = os.path.join(
        args.output_dir, f"{args.prefix}.analysis.md")

    summary = pd.DataFrame([
        {
            "implementation": "Original",
            "elapsed_seconds": original["elapsed_seconds"],
            "long_lived_peak_rss_gib": original["peak_rss_bytes"] / GIB,
            "max_child_rss_gib": original["max_children_rss_bytes"] / GIB,
            "matrix_digest": original["matrix_digest"],
            "matrix_sum": original["matrix_sum"],
        },
        {
            "implementation": "Optimized",
            "elapsed_seconds": optimized["elapsed_seconds"],
            "long_lived_peak_rss_gib": optimized["peak_rss_bytes"] / GIB,
            "max_child_rss_gib": optimized["max_children_rss_bytes"] / GIB,
            "matrix_digest": optimized["matrix_digest"],
            "matrix_sum": optimized["matrix_sum"],
        },
    ])
    summary.to_csv(summary_path, index=False)

    runtime_reduction = (
        1 - optimized["elapsed_seconds"] / original["elapsed_seconds"]) * 100
    speedup = original["elapsed_seconds"] / optimized["elapsed_seconds"]
    coordinator_reduction = (
        1 - optimized["peak_rss_bytes"] / original["peak_rss_bytes"]) * 100

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#405B73", "#2E7D62"]

    axes[0].bar(
        summary["implementation"], summary["elapsed_seconds"],
        color=colors, width=0.62)
    axes[0].set_ylabel("Wall time (seconds)")
    axes[0].set_title("A. Exact pairwise runtime")
    axes[0].text(
        0.5, 0.95, f"{speedup:.2f}x faster\n{runtime_reduction:.1f}% less time",
        transform=axes[0].transAxes, ha="center", va="top", fontsize=10)

    memory_labels = ["Original\nprocess", "Optimized\ncoordinator",
                     "Largest optimized\nworker"]
    memory_values = [
        original["peak_rss_bytes"] / GIB,
        optimized["peak_rss_bytes"] / GIB,
        optimized["max_children_rss_bytes"] / GIB,
    ]
    axes[1].bar(
        memory_labels, memory_values,
        color=["#405B73", "#2E7D62", "#C98232"], width=0.62)
    axes[1].set_ylabel("Peak RSS (GiB)")
    axes[1].set_title("B. Memory components")
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].text(
        0.5, 0.95,
        f"Coordinator: {coordinator_reduction:.1f}% lower than original",
        transform=axes[1].transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle(
        f"SLX-24007, 100 KB, {original['samples_including_diploid'] - 1} cells "
        f"+ diploid, {original['pairs']} exact pairs",
        fontsize=12)
    fig.text(
        0.5, 0.01,
        "All pairwise values are bit-identical. Worker bars are component "
        "peaks, not aggregate SLURM MaxRSS.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    original_trace = pd.read_csv(original["memory_trace_path"])
    optimized_trace = pd.read_csv(optimized["memory_trace_path"])
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(
        original_trace["elapsed_seconds"],
        original_trace["rss_bytes"] / GIB,
        color=colors[0], label="Original process", linewidth=1.8)
    axis.plot(
        optimized_trace["elapsed_seconds"],
        optimized_trace["rss_bytes"] / GIB,
        color=colors[1], label="Optimized coordinator", linewidth=1.8)
    axis.set_xlabel("Elapsed time (seconds)")
    axis.set_ylabel("Resident memory (GiB)")
    axis.set_title("Long-lived process memory over time")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(trace_path, dpi=180)
    plt.close(fig)

    with open(analysis_path, "w") as handle:
        handle.write(
            "# Exact 100 KB benchmark\n\n"
            f"- Input: SLX-24007, "
            f"{original['samples_including_diploid'] - 1} cells plus diploid, "
            f"{original['profile_bins']:,} bins per profile, "
            f"{original['pairs']} pairwise comparisons.\n"
            f"- Exactness: identical matrix digest "
            f"`{original['matrix_digest']}` and matrix sum "
            f"`{original['matrix_sum']}`.\n"
            f"- Runtime: {original['elapsed_seconds']:.1f} s original versus "
            f"{optimized['elapsed_seconds']:.1f} s optimized "
            f"({runtime_reduction:.1f}% reduction, {speedup:.2f}x speedup).\n"
            f"- Long-lived RSS: {original['peak_rss_bytes'] / GIB:.3f} GiB "
            f"original versus {optimized['peak_rss_bytes'] / GIB:.3f} GiB "
            f"optimized coordinator ({coordinator_reduction:.1f}% reduction).\n"
            f"- Largest clean worker: "
            f"{optimized['max_children_rss_bytes'] / GIB:.3f} GiB. "
            "Workers are retired after bounded batches so native allocations "
            "cannot accumulate for the full all-pairs run.\n"
            "- Limitation: macOS sandboxing prevented aggregate process-tree "
            "sampling. Use SLURM `sacct MaxRSS` for the authoritative total "
            "job-memory comparison on the cluster.\n")

    print(json.dumps({
        "verified_exact": True,
        "runtime_reduction_percent": runtime_reduction,
        "speedup": speedup,
        "coordinator_rss_reduction_percent": coordinator_reduction,
        "summary_csv": summary_path,
        "runtime_memory_png": figure_path,
        "memory_trace_png": trace_path,
        "analysis_markdown": analysis_path,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
