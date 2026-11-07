#!/usr/bin/env python3
"""Build a starter Local AI 800 dataset from public sources plus authored Chinese examples."""
import argparse
import json
import random
import re
from pathlib import Path

# Original starter examples authored for this project; expand and review before production training.
CURATED = [
    ("用 C 语言解释指针和取地址运算符。", "指针变量保存对象的内存地址。若 int x=10; int *p=&x;，&x 取得 x 的地址，p 保存该地址，*p 访问该地址处的 int 对象。解引用前必须确保指针有效且类型匹配。"),
    ("写一个安全的 C 函数，计算字符串长度，参数可能为空。", "#include <stddef.h>\nsize_t safe_strlen(const char *s) {\n    size_t n = 0;\n    if (s == NULL) return 0;\n    while (s[n] != '\\0') ++n;\n    return n;\n}\n说明：返回 0 同时代表空字符串或 NULL；若需要区分，应改为返回错误码并通过输出参数返回长度。"),
    ("C 编译时报 undefined reference，应该怎样排查？", "这是链接阶段错误，不一定是语法错误。依次检查：函数是否有定义；目标文件是否参与链接；库是否通过 -l 指定；库搜索路径是否通过 -L 指定；静态库顺序是否把依赖库放在使用方之后；C/C++ 混编时检查 extern \"C\"。用 nm -C 和链接器 -v 输出辅助定位。"),
    ("解释 ELF 文件中的 .text、.data、.bss。", ".text 通常保存机器指令；.data 保存已初始化的可写全局/静态数据；.bss 保存未显式初始化或初始化为零的全局/静态数据，通常在文件中不占对应的实际零字节内容。可用 readelf -S 查看节表。"),
    ("如何查看 Android APK 是否包含 arm64-v8a 原生库？", "先执行 unzip -l app.apk，检查是否存在 lib/arm64-v8a/*.so。也可用 aapt dump badging app.apk 查看 APK 元数据。存在该目录只说明打包了该 ABI 的库，不保证库依赖、签名或运行时兼容性正确。"),
    ("给出一个安全的 Shell 文件存在性检查示例。", "#!/system/bin/sh\nFILE=\"/data/local/tmp/example.bin\"\nif [ -f \"$FILE\" ]; then\n  echo \"文件存在\"\nelse\n  echo \"文件不存在\"\nfi\n注意：变量加引号，避免空格或通配符导致参数拆分。"),
    ("Android NDK 中 JNI 是什么？", "JNI 是 Java/Kotlin 与 native C/C++ 代码交互的接口。Java/Kotlin 声明 native 方法，C/C++ 实现对应 JNI 函数；需注意 JNI 签名、线程附加/分离、局部引用生命周期以及异常检查。"),
    ("ARM64 汇编中 x0 到 x7 通常有什么用途？", "在 AArch64 常见调用约定中，x0-x7 用于传递前 8 个整数或指针参数，x0 通常也用于返回值。具体仍取决于 ABI、参数类型和调用约定；分析二进制时应结合函数原型和反汇编上下文。"),
    ("Shell 脚本执行命令前如何降低误删风险？", "先确认目标路径非空且位于预期目录；对变量加双引号；拒绝对空变量执行递归删除；先用 printf 或 find 预览目标；需要破坏性操作时要求人工确认，并避免把网络下载内容直接管道交给 shell 执行。"),
    ("如何用 readelf 查看 ELF 的动态依赖？", "执行 readelf -d 文件.so，查看 NEEDED 条目；也可用 llvm-readobj --needed-libs。不要仅凭依赖列表断定设备一定能加载，还要核对 ABI、符号版本、API level 和依赖库是否存在。"),
    ("解释 C 指针越界的风险和修复方法。", "越界读写会造成未定义行为，可能导致崩溃、数据破坏或安全漏洞。为缓冲区显式传递长度，循环条件严格小于容量，检查整数溢出，使用有长度约束的接口，并在测试中启用 AddressSanitizer。"),
    ("如何检查 Android .so 的 ABI 和最低 API 兼容性？", "使用 file 和 llvm-readelf -h 检查 ELF class、机器架构；用 llvm-readelf -d 查看依赖；用 NDK 的 llvm-readelf/llvm-objdump 检查符号。最低 API 兼容性还需检查导入的 Android 符号版本及实际设备测试。"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data/train.jsonl")
    parser.add_argument("--max-code", type=int, default=20000)
    parser.add_argument("--max-shell", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=800)
    args = parser.parse_args()

    rows = []
    risky_shell = re.compile(
        r"(^|[;&|\s])(rm\s+-rf|mkfs(?:\.|\s)|dd\s+if=|shutdown|reboot|"
        r"chmod\s+-R\s+777|curl[^\n]*\|\s*(sh|bash)|wget[^\n]*\|\s*(sh|bash))",
        re.IGNORECASE,
    )

    code = load_dataset("flwrlabs/code-alpaca-20k", split="train")
    code_rows = []
    for row in code:
        instruction = str(row.get("instruction", "")).strip()
        answer = str(row.get("output", "")).strip()
        extra = str(row.get("input", "")).strip()
        if instruction and answer:
            code_rows.append({
                "instruction": instruction, "input": extra, "output": answer,
                "source": "flwrlabs/code-alpaca-20k",
                "source_license": "Apache-2.0; verify current dataset card",
            })
    random.Random(args.seed).shuffle(code_rows)
    rows.extend(code_rows[:args.max_code])

    shell = load_dataset("dirac-run/ec-training-data", split="train")
    shell_rows = []
    for row in shell:
        request = str(row.get("request", "")).strip()
        response = row.get("response") or {}
        if not isinstance(response, dict):
            continue
        command = str(response.get("value", "")).strip()
        kind = str(response.get("kind", "")).strip().upper()
        if request and command and kind == "COMMAND" and not risky_shell.search(command):
            shell_rows.append({
                "instruction": "Write a safe Linux Bash command for this task. Explain assumptions if needed: " + request,
                "output": command, "source": "dirac-run/ec-training-data",
                "source_license": "Apache-2.0; verify current dataset card",
            })
    random.Random(args.seed + 1).shuffle(shell_rows)
    rows.extend(shell_rows[:args.max_shell])

    for prompt, answer in CURATED:
        rows.append({
            "instruction": prompt, "output": answer,
            "source": "Local AI 800 original curated examples",
            "source_license": "Project-authored",
        })

    random.Random(args.seed + 2).shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"  CodeAlpaca: {min(len(code_rows), args.max_code)}")
    print(f"  Shell commands: {min(len(shell_rows), args.max_shell)}")
    print(f"  Original Chinese technical examples: {len(CURATED)}")
    print("Verify upstream dataset licenses and manually review data before training.")


if __name__ == "__main__":
    main()
