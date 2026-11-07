# Public dataset source register

These are starter sources for a reproducible training set. Re-check the current dataset card, upstream source license, attribution and terms before each use; a public download link alone is not a license grant.

| Dataset | Use | Reported license | Notes |
|---|---|---|---|
| https://huggingface.co/datasets/flwrlabs/code-alpaca-20k | General code instruction / answer pairs | Apache-2.0 | 20k English examples; fields: instruction, input, output |
| https://huggingface.co/datasets/dirac-run/ec-training-data | Linux shell request / command pairs | Apache-2.0 | EasyCommand data; converter filters COMMAND responses |
| https://huggingface.co/datasets/ajibawa-2023/C-Code-Large | C source-code corpus | MIT | Large raw code corpus, not instruction/answer SFT data; requires a separate carefully licensed transformation/evaluation pipeline |

## Normalize starter datasets

On a CUDA notebook after installing requirements:

~~~bash
pip install datasets
python training/prepare_public_data.py --output training/data/train.jsonl
python training/validate_dataset.py training/data/train.jsonl
~~~

The included converter combines CodeAlpaca and EasyCommand into the training JSONL schema. It does not silently treat raw source-code files as instruction examples.

## Missing specialization data

The starter sources are not enough to make a strong Chinese C/ELF/Android reverse-engineering assistant. Add licensed and reviewed C/C++, Android NDK, ELF/ABI, and Chinese technical instruction data. Keep validation/test prompts separate from training rows. Do not scrape private dumps, proprietary code, credentials, or files whose license does not permit training.
