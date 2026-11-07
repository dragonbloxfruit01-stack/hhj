#!/usr/bin/env python3
"""LoRA/QLoRA fine-tuning for Local AI 800. Run on a CUDA cloud GPU."""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def read_messages(row):
    if isinstance(row.get("messages"), list):
        messages = []
        for item in row["messages"]:
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
            if messages and any(m["role"] == "assistant" for m in messages):
                rows.append({"messages": messages})
    if not rows:
        raise ValueError(f"No usable training rows found in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", default="outputs/localai800-lora")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this starter QLoRA recipe. Use a cloud GPU runtime.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if not args.no_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw_rows = load_jsonl(args.train_file)
    texts = []
    for row in raw_rows:
        try:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        except Exception as exc:
            raise RuntimeError(
                "Tokenizer has no compatible chat template. Choose an Instruct checkpoint."
            ) from exc
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=args.max_length, padding=False)

    tokenized = dataset.map(
        tokenize_batch, batched=True, remove_columns=dataset.column_names,
        desc="Tokenizing licensed instruction data",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if quant_config is not None else "adamw_torch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")
    print("Run held-out evaluation before merging or quantizing this adapter.")


if __name__ == "__main__":
    main()
