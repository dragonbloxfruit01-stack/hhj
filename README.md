# Local AI 800

Android local-AI starter project for OPPO Reno14 Pro / Android 16.

## Current implementation status

This repository is the Android application scaffold and training setup. It is **not yet a completed offline AI app**: the current native bridge is a placeholder until a GGUF inference backend (such as llama.cpp) is integrated.

## Planned capabilities

- Chinese conversation and technical assistance
- C/C++, Shell, Android NDK and APK build guidance
- Local GGUF model selection
- Offline inference after the inference backend and model are added

## Build

The Android debug APK is built by the GitHub Actions workflow after the project files are committed.

## Model and training

See [training/README.md](training/README.md) for the proposed open-model fine-tuning workflow. Training requires a suitable cloud GPU; the Android phone is intended for inference, not full model training.
