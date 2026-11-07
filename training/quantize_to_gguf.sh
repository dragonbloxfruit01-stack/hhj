#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash training/quantize_to_gguf.sh /path/to/merged-hf-model /path/to/llama.cpp Q4_K_M
MODEL_DIR="${1:?Merged Hugging Face model directory required}"
LLAMA_DIR="${2:?llama.cpp source directory required}"
QUANT="${3:-Q4_K_M}"
F16_GGUF="${MODEL_DIR%/}/localai800-f16.gguf"
OUT_GGUF="${MODEL_DIR%/}/localai800-${QUANT}.gguf"

python "${LLAMA_DIR}/convert_hf_to_gguf.py" "${MODEL_DIR}" --outfile "${F16_GGUF}" --outtype f16
"${LLAMA_DIR}/build/bin/llama-quantize" "${F16_GGUF}" "${OUT_GGUF}" "${QUANT}"
echo "Quantized GGUF: ${OUT_GGUF}"
