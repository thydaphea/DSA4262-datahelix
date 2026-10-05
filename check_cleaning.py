"""Temporary-only regression checks for reproducibility and cleaning decisions."""
import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from features import pooled_features, bag_features
from clean_data import raw_sites, cleaned_sites

ROOT = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, labels = root / 'input.json', root / 'labels.csv'
        reads = [[.01, 2., 100., .02, 3., 110., .03, 4., 120.]]
        source.write_text('\n' + json.dumps({'TX.1': {'0002': {'aagacca': reads}, '3': {'AAGACCA': reads * 2}}}) + '\n')
        labels.write_text('transcript_id,transcript_position,n_reads,label,start,end\nTX.1,0002,1,0.25,0,1\nTX.1,3,2,1,1,2\nMISSING,4,1,0,2,3\n')

        def run(output, extra=(), success=True, input_path=source):
            command = [sys.executable, str(ROOT / 'clean_data.py'), '--data', str(input_path),
                       '--labels', str(labels), '--out-dir', str(output), *extra]
            result = subprocess.run(command, capture_output=True, text=True)
            assert (result.returncode == 0) == success, result.stderr
            return result

        # Fractional labels are not silently rounded or thresholded.
        failed = run(root / 'binary', ['--label-mode', 'binary'], success=False)
        assert 'fractional label' in failed.stderr and not (root / 'binary').exists()
        for name in ('first', 'second'):
            run(root / name, ['--label-mode', 'fractional'])
        first, second = root / 'first', root / 'second'
        for item in first.iterdir():
            assert item.read_bytes() == (second / item.name).read_bytes(), item.name
        clean = list(cleaned_sites(first / 'data.json.gz'))
        original = list(raw_sites(source))
        assert clean[0][0] == ('TX.1', '0002') and clean[0][1] == 'AAGACCA'
        for (key, motif, a), (key2, motif2, b) in zip(original, clean):
            assert (key, motif) == (key2, motif2)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(pooled_features(motif,a), pooled_features(motif2,b))
            for x,y in zip(bag_features(key,motif,a,64,42), bag_features(key2,motif2,b,64,42)):
                np.testing.assert_array_equal(x,y)
        with (first / 'labels.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]['label'] == '0.25' and 'start' not in rows[0] and 'end' not in rows[0]
        audit = json.loads((first / 'audit.json').read_text())
        assert audit['counts']['labels_without_signal'] == 1
        # Presence/absence conversion is explicit and leaves source fractions intact.
        run(root / 'converted')
        with (root / 'converted/labels.csv').open() as handle:
            converted = list(csv.DictReader(handle))
        assert [float(row['modification_fraction']) for row in converted] == [0.25, 1.]
        assert [row['label'] for row in converted] == ['1', '1']
        conversion_audit = json.loads((root / 'converted/audit.json').read_text())
        assert conversion_audit['binary_label_counts'] == {'1': 2}
        # Default conversion maps zero to zero; explicit mode produces identical output.
        run(root / 'explicit_conversion', ['--label-mode', 'fractional-binary'])
        assert (root/'converted/labels.csv').read_bytes() == (root/'explicit_conversion/labels.csv').read_bytes()
        original_labels = labels.read_text()
        labels.write_text(original_labels.replace(',1,0.25,', ',1,0,'))
        run(root / 'zero_conversion')
        with (root/'zero_conversion/labels.csv').open() as handle:
            zero_rows = list(csv.DictReader(handle))
        assert [row['label'] for row in zero_rows] == ['0', '1']
        labels.write_text(original_labels)
        # Unlabeled inference cleaning continues to work with the new default.
        subprocess.run([sys.executable, str(ROOT/'clean_data.py'), '--data', str(source),
                        '--out-dir', str(root/'unlabeled')], check=True, capture_output=True)
        assert not (root/'unlabeled/labels.csv').exists()
        assert json.loads((root/'unlabeled/audit.json').read_text())['binary_label_rule'] is None
        # Original files and existing outputs are never overwritten.
        before = (first / 'data.json.gz').read_bytes()
        run(first, ['--label-mode','fractional'], success=False)
        assert (first / 'data.json.gz').read_bytes() == before
        # Optional coverage threshold is explicit and audited.
        run(root / 'filtered', ['--label-mode','fractional','--min-reads','2'])
        audit = json.loads((root/'filtered/audit.json').read_text())
        assert audit['counts']['below_min_reads'] == 1 and audit['counts']['output_sites'] == 1
        # Gzip and plain-text source inputs are both supported.
        compressed = root / 'input.json.gz'
        with gzip.open(compressed, 'wt') as handle:
            handle.write(source.read_text())
        run(root/'compressed', ['--label-mode','fractional'], input_path=compressed)
        assert (root/'compressed/data.json.gz').read_bytes() == before
        # Fail atomically on duplicate JSON keys, nonfinite values, or count mismatch.
        for index, contents in enumerate([
            '{"TX.1":{"3":{"AAGACCA":'+json.dumps(reads)+'},"3":{"AAGACCA":'+json.dumps(reads)+'}}}',
            json.dumps({'TX.1': {'3': {'AAGACCA': [[float('nan')] * 9]}}}),
            json.dumps({'TX.1': {'3': {'AAGACCA': reads}}}),
        ]):
            bad = root / f'bad{index}.json'
            bad.write_text(contents+'\n')
            output = root / f'bad_output{index}'
            run(output,['--label-mode','fractional'],False,bad)
            assert not output.exists()
        assert not list(root.glob('.bad_output*'))
    print('PASS: deterministic output, preserved coordinates/signals/features, fractional-label policy, gzip, audit, filtering, and atomic failures')


if __name__ == '__main__':
    main()
