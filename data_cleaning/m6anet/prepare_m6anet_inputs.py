#!/usr/bin/env python3
"""Prepare course JSONs and dataset2 binary targets for an m6Anet benchmark.

The nine signal features remain unchanged. A tenth value is a synthetic
read-row ID unique within the site; it is NOT an original nanopore read ID
and cannot be used to link reads across sites. Dataset2 fractions are exported
to a separate binary-label CSV; by default fraction > 0 means label 1.
Targets are never supplied to inference. Original source files are preserved.
Use --labels-only to add labels to an already prepared benchmark.
Compatibility checked against m6anet 2.1.0 NanopolishDS.load_data/_load_data.
Reference: https://github.com/GoekeLab/m6anet (MIT-licensed upstream package).
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile


SOURCE_FILES = {
    "dataset0": "data0/dataset0_json/dataset0.json",
    "dataset1": "data1/dataset1_json/dataset1.json",
    "dataset2": "data2/dataset2_json/dataset2.json",
}


def prepare_dataset2_labels(source, destination, rule="nonzero", threshold=None):
    """Convert ground truth only, retaining fractions and recording the rule."""
    if destination.exists():
        raise FileExistsError("Label output already exists; choose a new "
                              "--dataset2-label-output: " + str(destination))
    if rule == "nonzero":
        if threshold is not None:
            raise ValueError("--dataset2-threshold requires --dataset2-rule threshold")
    elif rule == "threshold":
        if threshold is None or not 0 < threshold <= 1:
            raise ValueError("Threshold rule requires 0 < --dataset2-threshold <= 1")
    else:
        raise ValueError("Unknown dataset2 label rule: " + str(rule))
    rows, seen = [], set()
    with source.open(encoding="utf-8", newline="") as incoming:
        reader = csv.DictReader(incoming)
        required = {"transcript_id", "transcript_position", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Dataset2 labels require columns: " + ", ".join(sorted(required)))
        for line_number, row in enumerate(reader, 2):
            tx = row["transcript_id"]
            if not tx or not tx.strip():
                raise ValueError("Missing transcript ID at label line {}".format(line_number))
            pos = row["transcript_position"]
            if pos is None or str(int(pos)) != pos or int(pos) < 0:
                raise ValueError("Invalid position at label line {}".format(line_number))
            fraction = float(row["label"])
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Expected a nonmissing fraction in [0, 1] at label line {}".format(line_number))
            key = (tx, pos)
            if key in seen:
                raise ValueError("Duplicate label site: {}:{}".format(tx, pos))
            seen.add(key)
            binary = int(fraction > 0 if rule == "nonzero" else fraction >= threshold)
            rows.append([tx, pos, row["label"], binary, rule] +
                        ([threshold] if rule == "threshold" else []))
    positives = sum(row[3] for row in rows)
    if not rows or positives in (0, len(rows)):
        raise ValueError("Dataset2 conversion needs both binary classes for ROC/PR benchmarking")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as outgoing:
        writer = csv.writer(outgoing)
        writer.writerow(["transcript_id", "transcript_position", "modification_fraction",
                         "label", "binary_rule"] +
                        (["binary_threshold"] if rule == "threshold" else []))
        writer.writerows(rows)
    print("LABELS READY: {} | {} sites | {} negative | {} positive | rule {}{}".format(
        destination, len(rows), len(rows) - positives, positives, rule,
        " (fraction > 0)" if rule == "nonzero" else " (fraction >= {})".format(threshold)), flush=True)
    return {"sites": len(rows), "negative": len(rows) - positives,
            "positive": positives, "rule": rule, "threshold": threshold}


def read_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Repeated JSON key: {!r}".format(key))
        obj[key] = value
    return obj


def validate_site(tx, pos, motif, reads):
    if not isinstance(tx, str) or not tx:
        raise ValueError("Invalid transcript ID")
    if not isinstance(pos, str) or str(int(pos)) != pos or int(pos) < 0:
        raise ValueError("Position must be a canonical nonnegative integer: {!r}".format(pos))
    if (not isinstance(motif, str) or len(motif) != 7
            or any(base not in "ACGT" for base in motif)):
        raise ValueError("Expected a seven-base A/C/G/T sequence: {!r}".format(motif))
    central = motif[1:6]
    if not (central[0] in "AGT" and central[1] in "AG"
            and central[2:4] == "AC" and central[4] in "ACT"):
        raise ValueError("Non-DRACH central motif at {}:{}: {}".format(tx, pos, motif))
    if not isinstance(reads, list) or not reads:
        raise ValueError("Empty/invalid read list at {}:{}".format(tx, pos))
    for row in reads:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("Expected exactly nine signal values at {}:{}".format(tx, pos))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in row):
            raise ValueError("Nonfinite/nonnumeric signal at {}:{}".format(tx, pos))


def prepare_one(source, destination, limit_sites=None):
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if destination.exists():
        raise FileExistsError("Output already exists; choose a new --out-root: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    sites = total_reads = eligible_sites = 0

    # A failed conversion leaves no completed dataset directory.
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=str(destination.parent)) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        with source.open(encoding="utf-8") as incoming, \
                (staging / "data.json").open("wb") as outgoing, \
                (staging / "data.info").open("w", encoding="utf-8", newline="") as index:
            writer = csv.writer(index)
            writer.writerow(["transcript_id", "transcript_position", "n_reads", "start", "end"])
            for line_number, line in enumerate(incoming, 1):
                if not line.strip():
                    continue
                record = json.loads(line, object_pairs_hook=read_pairs)
                if not isinstance(record, dict) or not record:
                    raise ValueError("Invalid JSON record at line {}".format(line_number))
                for tx, positions in record.items():
                    if not isinstance(positions, dict) or not positions:
                        raise ValueError("Invalid positions at line {}".format(line_number))
                    for pos, motifs in positions.items():
                        if not isinstance(motifs, dict) or len(motifs) != 1:
                            raise ValueError("Expected one motif per site at line {}".format(line_number))
                        motif, reads = next(iter(motifs.items()))
                        validate_site(tx, pos, motif, reads)
                        key = (tx, pos)
                        if key in seen:
                            raise ValueError("Duplicate site: {}:{}".format(tx, pos))
                        seen.add(key)
                        prepared_reads = [row + [i] for i, row in enumerate(reads)]
                        prepared = {tx: {pos: {motif: prepared_reads}}}
                        payload = (json.dumps(prepared, ensure_ascii=True, allow_nan=False,
                                              separators=(",", ":")) + "\n").encode("ascii")
                        start = outgoing.tell()
                        outgoing.write(payload)
                        end = outgoing.tell()
                        writer.writerow([tx, pos, len(reads), start, end])
                        sites += 1
                        total_reads += len(reads)
                        eligible_sites += int(len(reads) >= 20)
                        if sites % 10000 == 0:
                            print("{}: {:,} sites prepared".format(destination.name, sites), flush=True)
                        if limit_sites is not None and sites >= limit_sites:
                            break
                    if limit_sites is not None and sites >= limit_sites:
                        break
                if limit_sites is not None and sites >= limit_sites:
                    break
        if not sites:
            raise ValueError("No sites found in " + str(source))
        if not eligible_sites:
            raise ValueError("No prepared sites meet m6Anet's 20-read minimum: " + str(source))
        summary = {
            "source_json": str(source.resolve()),
            "prepared_sites": sites,
            "prepared_reads": total_reads,
            "eligible_sites_at_least_20_reads": eligible_sites,
            "sites_below_20_reads": sites - eligible_sites,
            "partial_preview": limit_sites is not None,
            "limit_sites": limit_sites,
            "signal_values_per_read": 9,
            "synthetic_read_ids": True,
            "read_id_note": "IDs identify rows within a site, not original nanopore molecules.",
            "labels_used": False,
        }
        (staging / "preparation_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.rename(str(staging), str(destination))
    print("READY: {} | {:,} sites | {:,} eligible | {:,} below 20 reads".format(
        destination, sites, eligible_sites, sites - eligible_sites), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "project" / "datasets")
    parser.add_argument("--out-root", type=Path,
                        default=Path.home() / "project" / "m6anet_bench" / "input")
    parser.add_argument("--datasets", nargs="+", choices=sorted(SOURCE_FILES),
                        default=sorted(SOURCE_FILES))
    parser.add_argument("--limit-sites", type=int, default=None,
                        help="Optional partial preview; omit for the complete benchmark")
    parser.add_argument("--labels-only", action="store_true",
                        help="Create dataset2 binary labels without preparing JSON inputs again")
    parser.add_argument("--dataset2-label-source", type=Path, default=None,
                        help="Default: <data-root>/data2/data.info")
    parser.add_argument("--dataset2-label-output", type=Path, default=None,
                        help="Default: <out-root parent>/dataset2_binary_labels.csv")
    parser.add_argument("--dataset2-rule", choices=["nonzero", "threshold"], default="nonzero",
                        help="Binary target rule; default nonzero means fraction > 0")
    parser.add_argument("--dataset2-threshold", type=float, default=None,
                        help="Required only with --dataset2-rule threshold")
    args = parser.parse_args()
    if args.limit_sites is not None and args.limit_sites <= 0:
        parser.error("--limit-sites must be positive")
    if args.labels_only and "dataset2" not in args.datasets:
        parser.error("--labels-only requires dataset2 in --datasets")
    if args.dataset2_rule == "threshold":
        if args.dataset2_threshold is None or not 0 < args.dataset2_threshold <= 1:
            parser.error("--dataset2-rule threshold requires 0 < --dataset2-threshold <= 1")
    elif args.dataset2_threshold is not None:
        parser.error("--dataset2-threshold requires --dataset2-rule threshold")
    # Preflight every output before creating any files. Prepared inference input
    # directories are never overwritten; existing benchmarks use --labels-only.
    if not args.labels_only:
        for name in args.datasets:
            source = args.data_root / SOURCE_FILES[name]
            if not source.is_file():
                raise FileNotFoundError(str(source))
            if (args.out_root / name).exists():
                raise FileExistsError("Prepared input already exists: {}. Use --labels-only "
                                      "to create labels, or a new --out-root for new inputs.".format(args.out_root / name))
    if "dataset2" in args.datasets:
        label_source = args.dataset2_label_source or args.data_root / "data2/data.info"
        label_output = args.dataset2_label_output or args.out_root.parent / "dataset2_binary_labels.csv"
        prepare_dataset2_labels(label_source, label_output, args.dataset2_rule,
                                args.dataset2_threshold)
    if not args.labels_only:
        for name in args.datasets:
            prepare_one(args.data_root / SOURCE_FILES[name], args.out_root / name,
                        args.limit_sites)


if __name__ == "__main__":
    main()
