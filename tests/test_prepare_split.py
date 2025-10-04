import json
import sys
from pathlib import Path

import pandas as pd

from src import prepare_split


def run_prepare_split(config_path: Path) -> None:
    argv = sys.argv
    sys.argv = ["prepare_split", "-c", str(config_path)]
    try:
        prepare_split.main()
    finally:
        sys.argv = argv


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def test_multi_ec_accessions_stay_in_single_split(tmp_path):
    joined = tmp_path / "joined.tsv"
    splits_dir = tmp_path / "splits"
    config_path = tmp_path / "config.yaml"

    df = pd.DataFrame(
        [
            {"accession": "ACC1", "ec": "1.1.1.1; 2.2.2.2"},
            {"accession": "ACC2", "ec": "1.1.1.1"},
            {"accession": "ACC3", "ec": "3.3.3.3"},
            {"accession": "ACC4", "ec": "4.4.4.4; 2.2.2.2"},
        ]
    )
    df.to_csv(joined, sep="\t", index=False)

    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  joined_tsv: {joined}",
                f"  splits_dir: {splits_dir}",
                "min_sequences_per_class_for_train: 1",
                "random_seed: 123",
                "allow_multi_ec: true",
            ]
        ),
        encoding="utf-8",
    )

    run_prepare_split(config_path)

    seen_in_split: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        file_path = splits_dir / f"{split_name}.jsonl"
        assert file_path.exists()
        for item in read_jsonl(file_path):
            for acc in item["accessions"]:
                seen_in_split.setdefault(acc, set()).add(split_name)

    assert seen_in_split, "expected at least one accession in the output splits"
    for splits in seen_in_split.values():
        assert len(splits) == 1, f"accession appears in multiple splits: {splits}"
