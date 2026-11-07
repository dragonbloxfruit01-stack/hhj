# Local AI 800 — training pipeline

## Honest status

The Android APK and llama.cpp inference integration are built, but **no custom model training has run and no Local AI 800 GGUF weights exist yet**. Do not rename a stock Qwen GGUF and call it the trained model. Training requires a CUDA-capable cloud GPU; the Android phone is for inference, not full fine-tuning.

## Model plan

- Training base: `Qwen/Qwen3-0.6B` (confirm its current model card/license before use).
- Fine-tuning: 4-bit NF4 QLoRA, LoRA rank 16.
- Export: merge adapter, convert to GGUF, quantize to Q4_K_M.
- Phone output filename: `LocalAI800-Q4_K_M.gguf`.

The Q4_K_M GGUF is for phone inference. Do not use the quantized GGUF as the LoRA training source.

## Training focus

1. Chinese technical explanation and conversation.
2. C/C++ programming, pointers, memory safety, compiler/linker errors.
3. POSIX/Linux/Android Shell scripting and safe command construction.
4. Android NDK, JNI, Gradle and APK packaging.
5. ELF, ARM64, shared-library dependencies and dump analysis.
6. Careful technical reasoning and uncertainty handling.

The data-preparation script combines public code and shell instruction data with a small set of original Chinese technical examples. This is only a starter dataset, **not enough by itself to produce a strong specialist model**. Add more reviewed Chinese C/C++/Android/ELF instruction-answer examples and keep a separate held-out evaluation set.

## Data and licensing

Use public datasets only when their licenses permit the intended training and redistribution. Re-check the current upstream dataset cards and source licenses. Keep source/license metadata, remove secrets and personal data, and manually review command safety. See [data/FORMAT.md](data/FORMAT.md) and [data/SOURCES.md](data/SOURCES.md).

## Cloud GPU workflow

Run in a CUDA-enabled Linux notebook (for example, a Colab GPU runtime). Install a CUDA-compatible PyTorch build first, then from the repository root:

~~~bash
pip install -r training/requirements.txt
python training/prepare_public_data.py --output training/data/train.jsonl
python training/validate_dataset.py training/data/train.jsonl

python training/train_lora.py \
  --model Qwen/Qwen3-0.6B \
  --train-file training/data/train.jsonl \
  --output-dir outputs/localai800-lora \
  --epochs 2 \
  --max-length 2048
~~~

The trainer uses assistant-only loss: it masks user/system tokens and refuses to train if the tokenizer does not expose assistant token masks. It makes a deterministic train/validation split and records validation loss. If the selected chat template does not support assistant masks, use a compatible Instruct checkpoint/template rather than silently training on the prompt.

## Merge and quantize

Only continue after reviewing validation loss and testing a held-out set of Chinese C/C++, Shell, Android NDK, ELF and ARM64 questions.

~~~bash
python training/merge_adapter.py \
  --adapter outputs/localai800-lora \
  --output-dir outputs/localai800-merged

git clone --depth 1 --branch b12182 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --target llama-quantize -j2

bash training/quantize_to_gguf.sh \
  outputs/localai800-merged \
  llama.cpp \
  Q4_K_M
~~~

The expected output is `outputs/localai800-merged/localai800-Q4_K_M.gguf`. Copy that file to the phone and import it in Local AI 800. The app currently uses CPU inference, 2K context and a 384-token response limit; test memory use and generation speed on the actual device.

## Limitations

- A successful APK build does not prove that the app or every GGUF template works on the target phone.
- The current Android app has not been validated with a newly fine-tuned Local AI 800 checkpoint.
- The 800 MB target is a storage budget, not a quality guarantee. A 0.6B model remains limited on complex reverse engineering and long code tasks.
