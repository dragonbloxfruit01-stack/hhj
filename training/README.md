# Local AI 800 — training pipeline

## Current status

No custom training has been run and no fine-tuned weights are included. The Android app runs an imported GGUF model using llama.cpp. Fine-tuning is a separate cloud-GPU task.

## Base model

Start with Qwen3-0.6B-Instruct / Qwen3-0.6B. Confirm the exact model card and license before use. The unquantized checkpoint is the training source; GGUF Q4_K_M is intended for phone inference, not for LoRA training.

## Dataset focus

- Chinese instruction following and open conversation
- C and C++ programming, compiler diagnostics and debugging
- POSIX / Android shell scripting
- Android NDK, JNI, Gradle and APK packaging
- ELF, ARM64 and dump-file analysis
- Technical reasoning and safe troubleshooting

Use public datasets only when their licenses permit the intended training and redistribution. Record source, license, attribution and filtering metadata. Remove secrets and personal data. Keep validation examples separate from training sources. See data/FORMAT.md for the accepted JSONL schema and data/SOURCES.md for the starter source register.

## Cloud GPU workflow

The scripts target a CUDA-enabled Linux notebook or cloud GPU. Install a CUDA-compatible PyTorch build, then:

~~~bash
pip install -r training/requirements.txt
python training/prepare_public_data.py --output training/data/train.jsonl
python training/validate_dataset.py training/data/train.jsonl

python training/train_lora.py \
  --model Qwen/Qwen3-0.6B \
  --train-file data/train.jsonl \
  --output-dir outputs/localai800-lora \
  --epochs 2 \
  --max-length 2048
~~~

The default uses 4-bit NF4 QLoRA, rank 16, gradient checkpointing and a small batch. Adjust batch size, gradient accumulation and sequence length to fit the GPU. The starter script reports training loss; it does not certify model quality. Evaluate on a held-out set of C/C++, Shell, Chinese and Android tasks before using the adapter.

## Merge and quantize

After evaluation:

~~~bash
python training/merge_adapter.py \
  --adapter outputs/localai800-lora \
  --output-dir outputs/localai800-merged

bash training/quantize_to_gguf.sh \
  outputs/localai800-merged \
  /path/to/llama.cpp \
  Q4_K_M
~~~

Copy the resulting GGUF to the phone and import it in Local AI 800.

## Size and performance

An 800 MB target is a storage budget, not a quality guarantee. Quantization changes file size and can affect accuracy. The 0.6B baseline is small and may make mistakes on complex reverse engineering or long code tasks. Measure quality, latency and peak memory on the target device.
