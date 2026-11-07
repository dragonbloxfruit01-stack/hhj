# Local AI 800

Android 9.0+ (API 28) local GGUF chat application. Target build: arm64-v8a.

## Current app features

- Import a GGUF model using Android's document picker (copied to app-private storage)
- Load the model through the llama.cpp C API
- Apply the model's built-in chat template
- Multi-turn Chinese chat with system instructions
- CPU inference, configurable native thread count, 2K context
- Stop generation and clear conversation
- No network permission or cloud inference in the app

## Build

GitHub Actions builds a debug APK on pushes to `main`. Open the repository's **Actions** tab, select **Android Debug APK**, and download the `LocalAI800-debug-apk` artifact after a successful run.

The workflow fetches llama.cpp tag `b12182` during the CMake build. The first native build takes longer because it compiles the inference engine.

## Model

The APK does not bundle model weights. Import a compatible **chat/instruct GGUF** file after installing the app. A practical starting point is Qwen3-0.6B GGUF Q4_K_M (roughly 0.5 GB). The upstream Qwen model is Apache-2.0; check the exact quantized repository and model card before redistribution.

- Official model family: https://huggingface.co/Qwen/Qwen3-0.6B
- Official GGUF repository: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF

A 0.6B model is a small local assistant, not a frontier coding model. The 800 MB target is a storage budget, not a guarantee of reasoning or coding quality. Larger models need more storage and RAM.

## Important limitations

- This build uses llama.cpp CPU inference. Vulkan/GPU acceleration has not been enabled or device-validated.
- The app uses a 2048-token context and a 384-token response limit.
- Model quality depends on the selected GGUF. No custom fine-tuning has been completed yet.
- Build success does not equal device validation. Test on the target OPPO device before relying on it.

## Training

See [training/README.md](training/README.md). Training is a separate cloud-GPU task; it is not performed by the Android app.
