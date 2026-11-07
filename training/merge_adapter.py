#!/usr/bin/env python3
"""Merge a trained PEFT adapter into its base model for GGUF conversion."""
import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Folder created by train_lora.py")
    parser.add_argument("--output-dir", default="outputs/localai800-merged")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
