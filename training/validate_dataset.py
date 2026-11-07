#!/usr/bin/env python3
"""Validate Local AI 800 instruction data in JSONL format."""
import json
import sys
from pathlib import Path


def valid(row):
    if isinstance(row.get("messages"), list):
        messages = [m for m in row["messages"] if isinstance(m, dict)]
        has_assistant = any(m.get("role") == "assistant" and str(m.get("content", "")).strip()
                            for m in messages)
        has_user = any(m.get("role") == "user" and str(m.get("content", "")).strip()
                       for m in messages)
        return has_user and has_assistant
    prompt = row.get("instruction", row.get("prompt", ""))
    answer = row.get("output", row.get("completion", ""))
    return isinstance(prompt, str) and bool(prompt.strip()) and isinstance(answer, str) and bool(answer.strip())


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python validate_dataset.py data/train.jsonl")
    path = Path(sys.argv[1])
    good = bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not valid(row):
                    raise ValueError("missing user/instruction or assistant/output content")
                good += 1
            except Exception as exc:
                bad += 1
                print(f"Line {number}: {exc}")
    print(f"Valid rows: {good}; invalid rows: {bad}")
    if not good or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
