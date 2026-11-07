# Local AI 800 — training plan

## Current status

No custom training has been run and no fine-tuned weights are included. The Android application currently runs an imported GGUF model using llama.cpp. Fine-tuning is a separate phase that requires a cloud GPU.

## Proposed base model

Start with Qwen3-0.6B-Instruct as a compact Chinese-capable baseline. Verify the model card and Apache-2.0 license for the exact checkpoint and any converted/quantized copy. The initial Android test can use a Q4_K_M GGUF; quantized weights are for inference, not the preferred fine-tuning input.

## Data policy and focus

Use public datasets only when their licenses permit the intended use. Record source, license, language, and filtering metadata. Remove secrets and personal data. Split train/validation sets by source to reduce leakage.

Prioritize:
- Chinese instruction following and open conversation
- C and C++ programming, compiler diagnostics and debugging
- POSIX / Android shell scripting
- Android NDK, JNI, Gradle and APK packaging
- ELF, ARM64 and dump-file analysis
- Technical reasoning and safe troubleshooting

## Fine-tuning plan

1. Download the base model and tokenizer on a cloud GPU.
2. Normalize and deduplicate licensed public instruction data.
3. Create supervised conversation examples with explicit role boundaries.
4. Run LoRA/QLoRA supervised fine-tuning; keep a held-out validation split.
5. Evaluate on unseen Chinese C/C++ and shell tasks, not just training loss.
6. Merge or retain adapters as supported by the model/runtime.
7. Convert to GGUF and quantize; compare output quality, file size, latency and peak RAM on the target phone.

An 800 MB target is a storage budget, not a quality guarantee. The 0.6B baseline is small and may make mistakes on complex reverse engineering or large code tasks. Measure before claiming capability.
