from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fstlib
import medicc


def _profiles(seed=2030, samples=6, chromosomes=3, bins=10):
    rng = np.random.default_rng(seed)
    return {
        f"cell_{sample_idx}": "X".join(
            "".join(rng.integers(0, 6, size=bins).astype(str))
            for _ in range(chromosomes))
        for sample_idx in range(samples)
    }


def _reference_pairwise(model_fst, profiles):
    labels = list(profiles)
    distances = np.zeros((len(labels), len(labels)), dtype=float)
    for left_idx, right_idx in combinations(range(len(labels)), 2):
        distance = medicc.calc_MED_distance(
            model_fst, profiles[labels[left_idx]], profiles[labels[right_idx]])
        distances[left_idx, right_idx] = distance
        distances[right_idx, left_idx] = distance
    return pd.DataFrame(distances, index=labels, columns=labels)


@pytest.mark.parametrize("mode", ["inprocess", "forked", "external", "threads"])
def test_pairwise_execution_modes_are_exact(monkeypatch, mode):
    model_fst = medicc.io.read_fst()
    profiles = _profiles()

    monkeypatch.setenv("MEDICC2_FST_CACHE_MODE", "default")
    expected = _reference_pairwise(model_fst, profiles)
    monkeypatch.setenv("MEDICC2_PAIRWISE_MODE", mode)
    monkeypatch.setenv("MEDICC2_PAIRWISE_BATCH_SIZE", "2")
    monkeypatch.setenv("MEDICC2_PAIRWISE_WORKERS", "2")
    observed = medicc.calc_pairwise_distance_matrix(
        model_fst, profiles, parallel_run=False)

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


def test_streaming_fst_cache_is_exact(monkeypatch):
    model_fst = medicc.io.read_fst()
    profiles = _profiles(seed=2031, samples=4)

    monkeypatch.setenv("MEDICC2_PAIRWISE_MODE", "inprocess")
    monkeypatch.setenv("MEDICC2_FST_CACHE_MODE", "default")
    expected = medicc.calc_pairwise_distance_matrix(
        model_fst, profiles, parallel_run=False)

    monkeypatch.delenv("MEDICC2_FST_CACHE_MODE")
    monkeypatch.delenv("MEDICC2_FST_CACHE_BYTES", raising=False)
    observed = medicc.calc_pairwise_distance_matrix(
        model_fst, profiles, parallel_run=False)

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


def test_pairwise_checkpoint_resume_is_exact(monkeypatch, tmp_path):
    model_fst = medicc.io.read_fst()
    profiles = _profiles(seed=2032, samples=5)
    checkpoint = tmp_path / "pairwise.npz"

    monkeypatch.setenv("MEDICC2_FST_CACHE_MODE", "default")
    expected = _reference_pairwise(model_fst, profiles)
    monkeypatch.setenv("MEDICC2_PAIRWISE_MODE", "external")
    monkeypatch.setenv("MEDICC2_PAIRWISE_BATCH_SIZE", "2")
    monkeypatch.setenv("MEDICC2_PAIRWISE_WORKERS", "2")
    monkeypatch.setenv("MEDICC2_PAIRWISE_CHECKPOINT", str(checkpoint))

    first = medicc.calc_pairwise_distance_matrix(
        model_fst, profiles, parallel_run=False)
    resumed = medicc.calc_pairwise_distance_matrix(
        model_fst, profiles, parallel_run=False)

    assert checkpoint.is_file()
    pd.testing.assert_frame_equal(first, expected, check_exact=True)
    pd.testing.assert_frame_equal(resumed, expected, check_exact=True)

    changed_profiles = dict(profiles)
    changed_profiles["cell_0"] = changed_profiles["cell_0"].replace("1", "2")
    changed_expected = _reference_pairwise(model_fst, changed_profiles)
    changed_observed = medicc.calc_pairwise_distance_matrix(
        model_fst, changed_profiles, parallel_run=False)
    pd.testing.assert_frame_equal(
        changed_observed, changed_expected, check_exact=True)


def _reference_create_df_from_fsa(input_df, fsas, separator="X"):
    alleles = input_df.columns
    nr_alleles = len(alleles)
    samples = input_df.index.get_level_values("sample_id").unique()
    output_df = input_df.unstack("sample_id")
    internal_cns = {}

    for node in fsas:
        if node in samples:
            continue
        copy_numbers = medicc.tools.fsa_to_string(fsas[node]).split(separator)
        nr_chroms = len(copy_numbers) // nr_alleles
        for allele_idx, allele in enumerate(alleles):
            internal_cns[(allele, node)] = list("".join(
                copy_numbers[
                    allele_idx * nr_chroms:(allele_idx + 1) * nr_chroms]))

    internal_df = pd.DataFrame(internal_cns, index=output_df.index)
    internal_df.columns.names = ["allele", "sample_id"]
    return (
        pd.concat([output_df, internal_df], axis=1)
        .stack("sample_id")
        .reorder_levels(["sample_id", "chrom", "start", "end"])
        .sort_index()
    )


def test_create_df_from_fsa_matches_wide_reference():
    rows = []
    for sample_id in ["cell_1", "diploid"]:
        for chrom in ["chr1", "chr2"]:
            for start in [0, 100000]:
                rows.append((sample_id, chrom, start, start + 99999, "2", "1"))
    input_df = pd.DataFrame(
        rows, columns=["sample_id", "chrom", "start", "end", "cn_a", "cn_b"])
    input_df["chrom"] = medicc.tools.format_chromosomes(input_df["chrom"])
    input_df = input_df.set_index(
        ["sample_id", "chrom", "start", "end"]).sort_index()

    symbol_table = medicc.io.read_fst().input_symbols()
    fsas = {
        "internal_1": fstlib.factory.from_string(
            "12X34X22X11", isymbols=symbol_table, osymbols=symbol_table),
        "internal_2": fstlib.factory.from_string(
            "11X33X21X12", isymbols=symbol_table, osymbols=symbol_table),
    }

    expected = _reference_create_df_from_fsa(input_df, fsas)
    observed = medicc.core.create_df_from_fsa(input_df, fsas)
    pd.testing.assert_frame_equal(observed, expected, check_exact=True)


def test_spilled_ancestor_reconstruction_is_exact(monkeypatch, tmp_path):
    input_path = (
        Path(__file__).parent.parent /
        "examples/simple_example/simple_example.tsv")
    input_df = medicc.io.read_and_parse_input_data(str(input_path))
    model_fst = medicc.io.read_fst()
    monkeypatch.setenv("MEDICC2_PAIRWISE_MODE", "inprocess")
    monkeypatch.setenv("MEDICC2_FST_CACHE_MODE", "default")

    monkeypatch.setenv("MEDICC2_ANCESTOR_SPILL_MODE", "never")
    in_memory = medicc.main(
        input_df.copy(), model_fst, reconstruct_events=True)

    monkeypatch.setenv("MEDICC2_ANCESTOR_SPILL_MODE", "always")
    monkeypatch.setenv("MEDICC2_ANCESTOR_SPILL_DIR", str(tmp_path))
    spilled = medicc.main(
        input_df.copy(), model_fst, reconstruct_events=True)

    pd.testing.assert_frame_equal(spilled[1], in_memory[1], check_exact=True)
    assert spilled[2].format("newick") == in_memory[2].format("newick")
    assert spilled[3].format("newick") == in_memory[3].format("newick")
    pd.testing.assert_frame_equal(spilled[4], in_memory[4], check_exact=True)
    pd.testing.assert_frame_equal(spilled[5], in_memory[5], check_exact=True)
    assert not list(tmp_path.iterdir())
