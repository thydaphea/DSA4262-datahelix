"""Build inference-time GRU site features using the packaged GRU cleaning rules."""
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOCATIONS = ("minus1", "center", "plus1")
MEASURES = ("dwell", "signal_sd", "signal_mean")
STAT_NAMES = ("mean", "std", "median", "q25", "q75")
NONNEGATIVE_COLUMNS = (0, 1, 3, 4, 6, 7)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_matrix(reads):
    if not isinstance(reads, list):
        raise ValueError("Expected a list of reads")
    valid = []
    for read in reads:
        if (not isinstance(read, list) or len(read) != 9
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in read)):
            continue
        values = np.asarray(read, dtype=float)
        if (not np.isfinite(values).all()
                or (values[list(NONNEGATIVE_COLUMNS)] < 0).any()):
            continue
        valid.append(values)
    return np.asarray(valid, dtype=float).reshape(-1, 9)


def clean_gru_site(transcript, position, sequence, reads):
    matrix = _read_matrix(reads)
    invalid_reads_removed = len(reads) - len(matrix)
    if not re.fullmatch(r"[ACGT]{7}", sequence):
        return None, "invalid_sequences", invalid_reads_removed
    if not re.fullmatch(r"[AGT][AG]AC[ACT]", sequence[1:6]):
        return None, "non_drach_sequences", invalid_reads_removed
    if not len(matrix):
        return None, "too_few_valid_reads", invalid_reads_removed

    summaries = (
        matrix.mean(axis=0),
        matrix.std(axis=0),
        np.median(matrix, axis=0),
        np.quantile(matrix, 0.25, axis=0),
        np.quantile(matrix, 0.75, axis=0),
    )
    if not np.isfinite(np.concatenate(summaries)).all():
        return None, "nonfinite_summary", invalid_reads_removed
    row = {
        "transcript_id": transcript,
        "transcript_position": int(position),
        "sequence": sequence,
        "n_reads_valid": len(matrix),
    }
    for stat_name, summary in zip(STAT_NAMES, summaries):
        for location_index, location in enumerate(LOCATIONS):
            for measure_index, measure in enumerate(MEASURES):
                column_index = location_index * 3 + measure_index
                row[f"{location}_{measure}_{stat_name}"] = float(
                    summary[column_index]
                )
    return row, None, invalid_reads_removed


def clean_gru_json(path):
    """Return filtered GRU features and an audit using the original notebook policy."""
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    rows = []
    seen = set()
    counts = {
        "lines_scanned": 0,
        "blank_lines": 0,
        "invalid_reads_removed": 0,
        "invalid_sequences": 0,
        "non_drach_sequences": 0,
        "too_few_valid_reads": 0,
        "nonfinite_summary": 0,
        "retained_sites": 0,
    }

    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            counts["lines_scanned"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(record, dict) or len(record) != 1:
                    raise ValueError("Expected exactly one transcript per line")
                transcript, positions = next(iter(record.items()))
                if (not isinstance(transcript, str) or not transcript.strip()
                        or transcript != transcript.strip()
                        or not isinstance(positions, dict) or len(positions) != 1):
                    raise ValueError("Expected one position and a nonempty transcript ID")
                position_text, sequences = next(iter(positions.items()))
                if not isinstance(position_text, str) or not re.fullmatch(r"\d+", position_text):
                    raise ValueError("Position must be a nonnegative integer string")
                if not isinstance(sequences, dict) or len(sequences) != 1:
                    raise ValueError("Expected exactly one sequence per site")
                sequence, reads = next(iter(sequences.items()))
                if not isinstance(sequence, str):
                    raise ValueError("Sequence must be text")
                key = (transcript, int(position_text))
                if key in seen:
                    raise ValueError(f"Duplicate site: {key}")
                seen.add(key)

                row, reason, invalid_reads_removed = clean_gru_site(
                    transcript, key[1], sequence, reads
                )
                counts["invalid_reads_removed"] += invalid_reads_removed
                if reason:
                    counts[reason] += 1
                    continue
                rows.append(row)
                counts["retained_sites"] += 1
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"Cannot parse line {line_number}: {error}") from error

    if not rows:
        raise ValueError(f"{source}: no sites passed GRU cleaning")
    return pd.DataFrame(rows), counts
