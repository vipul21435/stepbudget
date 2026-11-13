"""Turn shards of the nebius/SWE-agent-trajectories dataset into one row per run.

The dataset (https://huggingface.co/datasets/nebius/SWE-agent-trajectories,
CC BY 4.0) stores one trajectory per row with the fields ``instance_id``,
``model_name``, ``target`` (whether the issue was solved), ``exit_status`` and
``trajectory`` (a list of messages with a ``role``). This script reads only
those fields, counts the messages with role ``ai`` as the steps of the run and
writes a CSV with the columns item, system, outcome, budget, exit_reason. No
message text is read or written.

Needs pyarrow, which stepbudget itself does not depend on:

    uv run --with pyarrow python scripts/public_runs.py --data-dir DIR --shards 0 1 2 --out runs.csv

Shards that are not yet in ``--data-dir`` are downloaded (about 90 MB each).
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories/resolve/main/data"
N_SHARDS = 12
COLUMNS = ["instance_id", "model_name", "target", "exit_status", "trajectory.list.element.role"]


def shard_name(index: int) -> str:
    return f"train-{index:05d}-of-{N_SHARDS:05d}.parquet"


def fetch(index: int, data_dir: Path) -> Path:
    path = data_dir / shard_name(index)
    if not path.exists():
        url = f"{BASE}/{shard_name(index)}"
        print(f"downloading {url}")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:  # noqa: S310
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
    return path


def rows(path: Path):
    """Yield (instance_id, model_name, solved, steps, exit_status), one row group at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for group in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(group, columns=COLUMNS).to_pydict()
        for item, system, solved, status, messages in zip(
            table["instance_id"],
            table["model_name"],
            table["target"],
            table["exit_status"],
            table["trajectory"],
            strict=True,
        ):
            steps = sum(1 for m in messages if m["role"] == "ai")
            yield item, system, bool(solved), steps, status if status is not None else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    seen: Counter[tuple[str, str]] = Counter()
    n = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "system", "outcome", "budget", "exit_reason"])
        for index in args.shards:
            for item, system, solved, steps, status in rows(fetch(index, args.data_dir)):
                writer.writerow([item, system, "success" if solved else "failure", steps, status])
                seen[(system, status)] += 1
                n += 1
    print(f"wrote {n} runs to {args.out}")
    for (system, status), count in sorted(seen.items()):
        print(f"{system}\t{status}\t{count}")


if __name__ == "__main__":
    main()
