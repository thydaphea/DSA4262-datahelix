"""Construct site-level and read-level model features from cleaned numeric arrays."""
import hashlib
import json
import numpy as np

KEY = ['transcript_id', 'transcript_position']
SCHEMA = 'nine-signals-summary-v1'
MIL_SCHEMA = 'nine-signals-attention-v1'


def sequence_features(motif):
    return np.array([float(base == nt) for base in motif for nt in 'ACGT'])


def pooled_features(motif, a):
    # Nine columns each contribute five summaries = 45 signal features.
    signal = np.concatenate([a.mean(0), a.std(0),
                             np.quantile(a, [.25, .5, .75], axis=0).ravel()])
    return np.concatenate([signal, [np.log1p(len(a))], sequence_features(motif)])


def bag_features(key, motif, reads, max_reads, seed):
    """Bounded, deterministic bag; canonical sorting makes sampling order-invariant.

    Coordinate hashing seeds sampling only. Coordinates are never model features.
    Padding is excluded from both standardization and attention using the mask.
    """
    if len(reads) > max_reads:
        order = np.lexsort(tuple(reads[:, j] for j in range(8, -1, -1)))
        digest = hashlib.sha256(json.dumps([seed, *key]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        reads = reads[order[rng.choice(len(reads), max_reads, replace=False)]]
    padded = np.zeros((max_reads, 9), dtype=np.float32)
    padded[:len(reads)] = reads
    mask = np.arange(max_reads) < len(reads)
    return padded, mask, sequence_features(motif).astype(np.float32)
