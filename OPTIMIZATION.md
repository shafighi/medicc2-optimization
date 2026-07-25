# Exact memory and runtime optimization

This branch preserves MEDICC2 distances, trees, ancestral profiles, and event
calls. It does not filter bins or segments and does not change copy-number
cutoffs.

## Default behavior

- Pairwise distances run in fresh external Python workers. Native OpenFST
  allocations are returned to the operating system after each bounded batch.
- A batch contains 64 sample pairs by default.
- SLURM jobs below 192 GB use one worker by default. Larger jobs use at most
  two workers unless `MEDICC2_PAIRWISE_WORKERS` is set explicitly.
- Pairwise matrices can be checkpointed and resumed.
- Intermediate ancestral FSAs spill to temporary disk automatically for trees
  with at least 64 input samples. Only the candidate needed by the current
  upward or downward reconstruction step is loaded.
- Event FSTs are loaded once per run, and dataframe pivots/copies that scale
  with samples times segments are avoided.

## Optional settings

```bash
export MEDICC2_PAIRWISE_MODE=external
export MEDICC2_PAIRWISE_BATCH_SIZE=64
export MEDICC2_PAIRWISE_WORKERS=1
export MEDICC2_PAIRWISE_CHECKPOINT=/path/to/pairwise-checkpoint.npz

export MEDICC2_ANCESTOR_SPILL_MODE=auto  # auto, always, or never
export MEDICC2_ANCESTOR_SPILL_DIR=/path/to/fast/scratch
```

OpenFST keeps its original 1 MB composition cache by default. For an emergency
lower-memory run, use:

```bash
export MEDICC2_FST_CACHE_MODE=streaming
```

The streaming cache is exact but can be slower. A custom byte limit is also
available:

```bash
export MEDICC2_FST_CACHE_BYTES=1048576
```

## Validation

Run the exactness suite:

```bash
pytest -q medicc/test_optimization_equivalence.py
```

Export profiles from a read-only 100 KB RDS table and benchmark them:

```bash
Rscript --vanilla tools/export_rds_profiles.R \
  /path/to/cn_binned_100.rds \
  benchmarks/sample_100kb_profiles.tsv \
  30

python tools/benchmark_pairwise_exact.py \
  --profiles benchmarks/sample_100kb_profiles.tsv \
  --output-prefix benchmarks/results/original \
  --implementation original

MEDICC2_PAIRWISE_MODE=external \
MEDICC2_PAIRWISE_WORKERS=2 \
MEDICC2_PAIRWISE_BATCH_SIZE=64 \
python tools/benchmark_pairwise_exact.py \
  --profiles benchmarks/sample_100kb_profiles.tsv \
  --output-prefix benchmarks/results/optimized \
  --implementation optimized
```

The optimized and original matrix digests must be identical before performance
results are accepted.
