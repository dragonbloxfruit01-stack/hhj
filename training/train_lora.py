#!/usr/bin/env python3
"""Assistant-only LoRA/QLoRA fine-tuning for Local AI 800. Run on a CUDA GPU."""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)


def read_messages(row):
    if isinstance(row.get("messages"), list):
        messages = []
        for item in row["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        return messages
    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    extra = str(row.get("input", "")).strip()
    answer = str(row.get("output", row.get("completion", ""))).strip()
    if not instruction or not answer:
        return []
    user_text = instruction + (("\n\n" + extra) if extra else "")
    messages = []
    system = str(row.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ])
    return messages


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            messages = read_messages(row)
            if messages and any(m["role"] == "assistant" for m in messages) and any(m["role"] == "user" for m in messages):
                rows.append(messages)
    if len(rows) < 10:
        raise ValueError(f"Need at least 10 usable examples; found {len(rows)}")
    return rows


class AssistantOnlyCollator:
    """Pad batches while computing loss only on assistant response tokens."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            ids = list(item["input_ids"])
            mask = list(item["assistant_masks"])
            if len(ids) != len(mask):
                raise ValueError("Token and assistant mask lengths differ")
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append([token if is_assistant else -100 for token, is_assistant in zip(ids, mask)] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/localai800-lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Run this script in a CUDA-enabled cloud notebook.")
    if not 0.01 <= args.eval_ratio <= 0.2:
        raise SystemExit("--eval-ratio must be between 0.01 and 0.2")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto",
        torch_dtype=compute_dtype, quantization_config=quant_config,
    )
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    messages = load_jsonl(args.train_file)
    encoded = []
    skipped = 0
    if not tokenizer.is_fast:
        raise RuntimeError("Assistant-only masking requires a fast tokenizer with offset mappings.")

    for example in messages:
        # Qwen3's stock chat template may not define {% generation %}. Instead,
        # locate each assistant message's content in the rendered conversation,
        # then map its character span to token offsets. This avoids training on
        # system/user prompt tokens and fails closed if the template rewrites content.
        rendered = tokenizer.apply_chat_template(
            example, tokenize=False, add_generation_prompt=False
        )
        assistant_spans = []
        for message_index, message in enumerate(example):
            if message["role"] != "assistant":
                continue
            marker = f"LOCALAI800_ASSISTANT_SPAN_{message_index}_END"
            marked_messages = [dict(item) for item in example]
            marked_messages[message_index]["content"] = marker
            marked_rendered = tokenizer.apply_chat_template(
                marked_messages, tokenize=False, add_generation_prompt=False
            )
            if marked_rendered.count(marker) != 1:
                raise RuntimeError("Could not uniquely locate assistant content in chat template.")
            prefix, suffix = marked_rendered.split(marker, 1)
            if rendered != prefix + message["content"] + suffix:
                raise RuntimeError(
                    "Chat template transforms assistant content; safe offset masking "
                    "cannot be guaranteed. Refusing to train."
                )
            assistant_spans.append((len(prefix), len(prefix) + len(message["content"])))

        tokenized = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True,
            truncation=True, max_length=args.max_length,
        )
        ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        assistant_mask = [
            int(any(start < span_end and end > span_start
                    for span_start, span_end in assistant_spans) and end > start)
            for start, end in offsets
        ]
        if len(ids) != len(assistant_mask) or not any(assistant_mask):
            skipped += 1
            continue
        encoded.append({"input_ids": ids, "assistant_masks": assistant_mask})

    if len(encoded) < 10:
        raise SystemExit(f"Only {len(encoded)} examples have assistant-token labels; check chat template.")
    dataset = Dataset.from_list(encoded).train_test_split(
        test_size=args.eval_ratio, seed=800
    )
    print(f"Training examples: {len(dataset['train'])}; validation examples: {len(dataset['test'])}; skipped: {skipped}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
        save_total_limit=2, gradient_checkpointing=True,
        optim="paged_adamw_8bit" if quant_config is not None else "adamw_torch",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        report_to="none", remove_unused_columns=False,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["test"],
        data_collator=AssistantOnlyCollator(tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with (Path(args.output_dir) / "localai800_training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"LoRA adapter saved to: {args.output_dir}")
    print(f"Validation metrics: {metrics}")
    print("Do not merge/quantize until held-out task tests pass.")


if __name__ == "__main__":
    main()
