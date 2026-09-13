"""Merge per-alpha load-robustness artifacts and draw the joint curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_load_robustness import SLOT_FIELDS, SUMMARY_FIELDS, plot_results


def find_files(root: Path, filename: str) -> list[Path]:
    return sorted(path for path in root.rglob(filename) if path.is_file())


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-alpha", type=float, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = read_rows(find_files(args.input_root, "load_robustness_summary.csv"))
    slot_rows = read_rows(find_files(args.input_root, "load_robustness_slot_metrics.csv"))
    summary_rows.sort(key=lambda row: float(row["alpha"]))
    slot_rows.sort(key=lambda row: (float(row["alpha"]), int(row["eval_slot"])))
    actual = [float(row["alpha"]) for row in summary_rows]
    expected = sorted(args.expected_alpha)
    if actual != expected:
        raise RuntimeError(f"alpha coverage mismatch: expected {expected}, got {actual}")

    output_dir = args.output_dir.resolve()
    summary_csv = output_dir / "load_robustness_summary.csv"
    slot_csv = output_dir / "load_robustness_slot_metrics.csv"
    write_rows(summary_csv, SUMMARY_FIELDS, summary_rows)
    write_rows(slot_csv, SLOT_FIELDS, slot_rows)
    figures = plot_results(summary_csv, output_dir / "figures")
    manifest = {
        "experiment": "combined zero-shot BLA-MAPPO business-load robustness",
        "alphas": actual,
        "summary_rows": len(summary_rows),
        "slot_rows": len(slot_rows),
        "figures": [path.name for path in figures],
        "source_manifests": [
            json.loads(path.read_text(encoding="utf-8"))
            for path in find_files(args.input_root, "run_manifest.json")
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_csv)
    print(slot_csv)


if __name__ == "__main__":
    main()
