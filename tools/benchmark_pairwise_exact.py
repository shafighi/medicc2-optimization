#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time
from itertools import combinations

import numpy as np
import pandas as pd
import psutil

import medicc


class ProcessTreeMonitor:
    def __init__(self, interval_seconds=0.05):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _rss_bytes(self):
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def _run(self):
        start = time.perf_counter()
        while not self.stop_event.is_set():
            rss_bytes = self._rss_bytes()
            elapsed = time.perf_counter() - start
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
            self.samples.append((elapsed, rss_bytes))
            self.stop_event.wait(self.interval_seconds)
        rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        self.samples.append((time.perf_counter() - start, rss_bytes))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        self.thread.join()


def read_profiles(path, max_samples=None, sample_ids=None, include_diploid=True):
    data = pd.read_csv(path, sep="\t", dtype=str)
    if list(data.columns) != ["sample_id", "profile"]:
        raise ValueError("Profile TSV must contain sample_id and profile columns")
    if sample_ids:
        missing = set(sample_ids).difference(data["sample_id"])
        if missing:
            raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
        data = data.set_index("sample_id").loc[sample_ids].reset_index()
        if include_diploid and "diploid" not in sample_ids:
            normal = pd.read_csv(path, sep="\t", dtype=str)
            data = pd.concat(
                [data, normal.loc[normal["sample_id"] == "diploid"]],
                ignore_index=True)
    elif max_samples is not None:
        non_normal = data.loc[data["sample_id"] != "diploid"].head(max_samples)
        normal = data.loc[data["sample_id"] == "diploid"]
        data = pd.concat([non_normal, normal], ignore_index=True)
    if not include_diploid:
        data = data.loc[data["sample_id"] != "diploid"]
    return dict(zip(data["sample_id"], data["profile"]))


def original_pairwise(model_fst, profiles):
    labels = list(profiles)
    distances = np.zeros((len(labels), len(labels)), dtype=float)
    for left_idx, right_idx in combinations(range(len(labels)), 2):
        distance = medicc.calc_MED_distance(
            model_fst, profiles[labels[left_idx]], profiles[labels[right_idx]])
        distances[left_idx, right_idx] = distance
        distances[right_idx, left_idx] = distance
    return pd.DataFrame(distances, index=labels, columns=labels)


def matrix_digest(dataframe):
    digest = hashlib.sha256()
    digest.update("\0".join(map(str, dataframe.index)).encode("utf-8"))
    digest.update(dataframe.to_numpy(dtype=np.float64).tobytes())
    return digest.hexdigest()


def max_rss_bytes(who):
    max_rss = resource.getrusage(who).ru_maxrss
    return int(max_rss if sys.platform == "darwin" else max_rss * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--implementation", choices=["original", "optimized"], required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--exclude-diploid", action="store_true")
    parser.add_argument("--monitor-interval", type=float, default=0.05)
    args = parser.parse_args()

    profiles = read_profiles(
        args.profiles,
        max_samples=args.max_samples,
        sample_ids=args.sample_ids,
        include_diploid=not args.exclude_diploid)
    model_fst = medicc.io.read_fst()
    monitor = ProcessTreeMonitor(args.monitor_interval)

    start = time.perf_counter()
    with monitor:
        if args.implementation == "original":
            pairwise = original_pairwise(model_fst, profiles)
        else:
            pairwise = medicc.calc_pairwise_distance_matrix(
                model_fst, profiles, parallel_run=False)
    elapsed_seconds = time.perf_counter() - start

    output_prefix = os.path.abspath(args.output_prefix)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    matrix_path = f"{output_prefix}.matrix.tsv"
    trace_path = f"{output_prefix}.memory.csv"
    metrics_path = f"{output_prefix}.metrics.json"
    pairwise.to_csv(matrix_path, sep="\t")
    pd.DataFrame(
        monitor.samples, columns=["elapsed_seconds", "rss_bytes"]
    ).to_csv(trace_path, index=False)

    first_profile = next(iter(profiles.values()), "")
    metrics = {
        "implementation": args.implementation,
        "pairwise_mode": os.environ.get("MEDICC2_PAIRWISE_MODE", ""),
        "fst_cache_mode": os.environ.get("MEDICC2_FST_CACHE_MODE", ""),
        "fst_cache_bytes": os.environ.get("MEDICC2_FST_CACHE_BYTES", ""),
        "workers": int(os.environ.get("MEDICC2_PAIRWISE_WORKERS", "1")),
        "batch_size": int(os.environ.get("MEDICC2_PAIRWISE_BATCH_SIZE", "64")),
        "samples_including_diploid": len(profiles),
        "pairs": len(profiles) * (len(profiles) - 1) // 2,
        "profile_characters": len(first_profile),
        "profile_bins": len(first_profile.replace("X", "")),
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": monitor.peak_rss_bytes,
        "peak_rss_gib": monitor.peak_rss_bytes / 1024 ** 3,
        "max_self_rss_bytes": max_rss_bytes(resource.RUSAGE_SELF),
        "max_children_rss_bytes": max_rss_bytes(resource.RUSAGE_CHILDREN),
        "matrix_sum": float(pairwise.to_numpy(dtype=float).sum()),
        "matrix_digest": matrix_digest(pairwise),
        "matrix_path": matrix_path,
        "memory_trace_path": trace_path,
    }
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
