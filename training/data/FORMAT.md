# Dataset format

Use UTF-8 JSON Lines (.jsonl), one training example per line. Two formats are accepted.

## Chat messages

~~~json
{"messages":[{"role":"system","content":"You are a careful C programming tutor."},{"role":"user","content":"Explain a pointer."},{"role":"assistant","content":"A pointer stores the address of an object..."}]}
~~~

## Instruction / output

~~~json
{"instruction":"Write a C function that reverses a string.","input":"Use an in-place algorithm.","output":"...complete answer..."}
~~~

Keep source, license, attribution and preprocessing notes for every imported dataset. Do not include passwords, private dumps, personal records, or data without redistribution/training permission. Keep validation examples separate from training sources.
