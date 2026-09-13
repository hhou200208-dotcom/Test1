"""Merge independently evaluated constellation sizes and draw joint curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_scalability_evaluation import ALGORITHMS, SLOT_FIELDS, SUMMARY_FIELDS, plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, nargs="+", default=[128, 160, 192, 256, 320])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(args.input_root.resolve().glob("**/scalability_summary.csv"))
    slot_paths = sorted(args.input_root.resolve().glob("**/scalability_slot_metrics.csv"))
    if not summary_paths or not slot_paths:
        raise FileNotFoundError("no scalability CSV files found below input root")

    summaries = [row for path in summary_paths for row in read_csv(path)]
    observed_pairs = sorted((row["algorithm"], int(row["n_sats"])) for row in summaries)
    expected_n = sorted(args.expected_n)
    expected_pairs = sorted((algorithm, n) for algorithm in ALGORITHMS for n in expected_n)
    if observed_pairs != expected_pairs:
        raise RuntimeError(
            f"expected every algorithm at N={expected_n}; observed {observed_pairs}"
        )
    summaries.sort(key=lambda row: (row["algorithm"], int(row["n_sats"])))

    summary_csv = output / "scalability_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    slot_csv = output / "scalability_slot_metrics.csv"
    slot_rows = 0
    with slot_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=SLOT_FIELDS)
        writer.writeheader()
        for path in slot_paths:
            for row in read_csv(path):
                writer.writerow(row)
                slot_rows += 1
    expected_slot_rows = len(ALGORITHMS) * len(expected_n) * int(summaries[0]["eval_slots"])
    if slot_rows != expected_slot_rows:
        raise RuntimeError(f"expected {expected_slot_rows} slot rows, observed {slot_rows}")

    figures = plot_results(summary_csv, output / "figures")
    source_manifests = []
    for path in sorted(args.input_root.resolve().glob("**/run_manifest.json")):
        source_manifests.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "experiment": "joint five-policy zero-shot constellation scalability",
        "algorithms": list(ALGORITHMS),
        "n_satellites": expected_n,
        "summary_rows": len(summaries),
        "slot_rows": slot_rows,
        "figures": [path.name for path in figures],
        "single_seed_no_confidence_interval": True,
        "source_runs": source_manifests,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(summary_csv.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
