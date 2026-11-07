#!/usr/bin/env python3
"""Download and normalize permissively licensed public code/shell instruction datasets."""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data/train.jsonl")
    parser.add_argument("--max-code", type=int, default=20000)
    parser.add_argument("--max-shell", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=800)
    args = parser.parse_args()

    rows = []

    # Apache-2.0 tagged CodeAlpaca mirror: instruction/input/output schema.
    code = load_dataset("flwrlabs/code-alpaca-20k", split="train")
    code_rows = []
    for row in code:
        instruction = str(row.get("instruction", "")).strip()
        answer = str(row.get("output", "")).strip()
        extra = str(row.get("input", "")).strip()
        if instruction and answer:
            code_rows.append({
                "instruction": instruction,
                "input": extra,
                "output": answer,
                "source": "flwrlabs/code-alpaca-20k",
                "source_license": "Apache-2.0; verify current dataset card",
            })
    random.Random(args.seed).shuffle(code_rows)
    rows.extend(code_rows[:args.max_code])

    # Apache-2.0 tagged EasyCommand shell pairs: request + response.value.
    shell = load_dataset("dirac-run/ec-training-data", split="train")
    shell_rows = []
    for row in shell:
        request = str(row.get("request", "")).strip()
        response = row.get("response") or {}
        if not isinstance(response, dict):
            continue
        command = str(response.get("value", "")).strip()
        kind = str(response.get("kind", "")).strip().upper()
        if request and command and kind == "COMMAND":
            shell_rows.append({
                "instruction": "Write a safe Linux Bash command for this task. Explain assumptions if needed: " + request,
                "output": command,
                "source": "dirac-run/ec-training-data",
                "source_license": "Apache-2.0; verify current dataset card",
            })
    random.Random(args.seed + 1).shuffle(shell_rows)
    rows.extend(shell_rows[:args.max_shell])

    random.Random(args.seed + 2).shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} normalized rows to {output}")
    print(f"  code instruction rows: {min(len(code_rows), args.max_code)}")
    print(f"  shell command rows: {min(len(shell_rows), args.max_shell)}")
    print("Add a separately licensed C/C++ and Chinese technical dataset before training a specialized checkpoint.")


if __name__ == "__main__":
    main()
