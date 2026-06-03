#!/usr/bin/env python
import argparse
import hashlib
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd

import medicc


def build_profiles(seed, n_samples, n_chromosomes, chromosome_length, max_copy_number):
    rng = np.random.default_rng(seed)
    profiles = {}

    for sample_idx in range(n_samples):
        chromosomes = []
        for _ in range(n_chromosomes):
            copy_numbers = rng.integers(0, max_copy_number, size=chromosome_length)
            chromosomes.append(''.join(copy_numbers.astype(str)))
        profiles[f'cell_{sample_idx}'] = 'X'.join(chromosomes)

    return profiles


def reference_pairwise_distance_matrix(model_fst, cn_str_dict):
    samples = list(cn_str_dict.keys())
    pdm = pd.DataFrame(0, index=samples, columns=samples, dtype=float)

    for sample_a, sample_b in combinations(samples, 2):
        cur_dist = medicc.calc_MED_distance(model_fst, cn_str_dict[sample_a], cn_str_dict[sample_b])
        pdm.loc[sample_a, sample_b] = cur_dist
        pdm.loc[sample_b, sample_a] = cur_dist

    return pdm


def dataframe_digest(df):
    return hashlib.sha256(df.to_numpy(dtype=float).tobytes()).hexdigest()


def summarize(df, elapsed_sec, impl):
    return {
        'implementation': impl,
        'elapsed_sec': elapsed_sec,
        'n_samples': int(df.shape[0]),
        'n_pairs': int(df.shape[0] * (df.shape[0] - 1) // 2),
        'matrix_sum': float(df.to_numpy(dtype=float).sum()),
        'diagonal_sum': float(np.trace(df.to_numpy(dtype=float))),
        'digest': dataframe_digest(df),
    }


def run_impl(args, profiles):
    medicc_fst = medicc.io.read_fst()

    start = time.perf_counter()
    if args.impl == 'reference':
        df = reference_pairwise_distance_matrix(medicc_fst, profiles)
    else:
        df = medicc.calc_pairwise_distance_matrix(medicc_fst, profiles, parallel_run=False)
    elapsed_sec = time.perf_counter() - start

    return df, summarize(df, elapsed_sec, args.impl)


def main():
    parser = argparse.ArgumentParser(
        description='Validate MEDICC2 pairwise distance matrix equivalence and collect small runtime metrics.')
    parser.add_argument('--mode', choices=['run', 'compare'], default='compare')
    parser.add_argument('--impl', choices=['reference', 'optimized'], default='optimized')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--samples', type=int, default=14)
    parser.add_argument('--chromosomes', type=int, default=6)
    parser.add_argument('--chromosome-length', type=int, default=18)
    parser.add_argument('--max-copy-number', type=int, default=6)
    args = parser.parse_args()

    profiles = build_profiles(
        seed=args.seed,
        n_samples=args.samples,
        n_chromosomes=args.chromosomes,
        chromosome_length=args.chromosome_length,
        max_copy_number=args.max_copy_number,
    )

    if args.mode == 'run':
        _, summary = run_impl(args, profiles)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    reference_df, reference_summary = run_impl(argparse.Namespace(**{**vars(args), 'impl': 'reference'}), profiles)
    optimized_df, optimized_summary = run_impl(argparse.Namespace(**{**vars(args), 'impl': 'optimized'}), profiles)
    pd.testing.assert_frame_equal(optimized_df, reference_df)

    print(json.dumps({
        'verified_equal': True,
        'reference': reference_summary,
        'optimized': optimized_summary,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
