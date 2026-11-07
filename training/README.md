# Local AI 800 — training plan

This directory documents the proposed fine-tuning workflow. It does not contain trained weights.

## Base model

Evaluate a permissively licensed compact Chinese-capable open model (for example, a Qwen 0.6B-class instruct model). Verify the exact model card and license before use.

## Data

Use public datasets only when their licenses permit the intended use. Keep source, license, language, and filtering metadata for every dataset. Remove secrets and personal data. Split train/validation sets by source to reduce leakage.

## Focus areas

- Chinese instruction following and open conversation
- C and C++ programming
- POSIX / Android shell scripting
- Android NDK, JNI, Gradle and APK packaging
- ELF, ARM64 and dump-file analysis
- Technical reasoning and troubleshooting

## Fine-tuning

Use a cloud GPU for LoRA/QLoRA supervised fine-tuning. Validate on held-out Chinese coding and technical tasks. Merge or retain adapters as supported by the selected runtime, then quantize to a mobile-supported GGUF format.

An 800 MB target is a storage budget, not a quality guarantee. Actual size depends on architecture, quantization, tokenizer and metadata. Measure quality and memory use on the target phone.

## Android integration status

The Android app currently contains only a JNI status stub. A real offline chat app still needs a compatible inference backend (such as llama.cpp), model loading, token streaming, cancellation, context management and device-memory testing.
