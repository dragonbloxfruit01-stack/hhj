#!/usr/bin/env python3
"""Build a starter Local AI 800 dataset from public sources plus authored Chinese examples."""
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

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
    ("C 语言中 malloc 返回 NULL 时应该怎么办？", "检查分配结果后再访问内存；失败时释放此前已成功分配的资源并返回明确错误。不要解引用 NULL。示例：void *p = malloc(n); if (p == NULL) return -1;"),
    ("解释 sizeof 指针和 sizeof 数组的区别。", "在数组仍处于定义作用域时，sizeof(array) 返回整个数组占用字节数；sizeof(pointer) 返回指针本身大小。数组作为函数参数时会退化为指针，因此不能在被调函数内用 sizeof 得到原数组长度。"),
    ("C++ 中 RAII 是什么？", "RAII 将资源生命周期绑定到对象生命周期：构造时获取资源，析构时释放资源。适用于内存、文件、锁等资源，可用 std::vector、std::unique_ptr、std::lock_guard 等标准库类型减少泄漏。"),
    ("解释 const char* 和 char* const。", "const char* 表示不能通过该指针修改所指字符，但指针本身可改指向；char* const 表示指针本身不可改指向，但可通过它修改字符（前提是底层对象可写）。"),
    ("如何用 clang 编译 Android arm64 的 C++ 程序？", "使用 Android NDK 提供的 aarch64-linux-android<API>-clang++ 交叉编译器，指定目标 API、源文件和链接库。确保所有依赖库也是 arm64 Android ABI，不能把桌面 Linux 的 .so 直接链接进 Android APK。"),
    ("Android JNI 为什么不能随意跨线程使用 JNIEnv？", "JNIEnv 是线程相关的接口指针，不能把一个线程的 JNIEnv 直接传给另一个线程。新建的 native 线程应通过 JavaVM::AttachCurrentThread 获取自己的 JNIEnv，并在不再需要时 DetachCurrentThread。"),
    ("APK 安装失败时如何区分签名和 ABI 问题？", "先记录完整安装错误。INSTALL_PARSE_FAILED 或签名不一致通常与包解析/签名有关；INSTALL_FAILED_NO_MATCHING_ABIS 通常表示 APK 没有设备支持的 ABI。用 apksigner verify --verbose 和 unzip -l 检查签名与 lib/<abi> 内容。"),
    ("解释 ELF 的 PT_LOAD 段。", "PT_LOAD 描述需要映射到进程地址空间的加载段，包含文件偏移、虚拟地址、文件大小、内存大小和权限标志。内存大小大于文件大小的部分通常以零填充。用 readelf -l 查看程序头。"),
    ("ELF 的 PLT 和 GOT 有什么作用？", "PLT 常用于外部函数调用的过程链接，GOT 保存运行时重定位相关地址。动态链接器可在加载或首次调用时解析符号并更新 GOT。具体布局受架构、链接选项和工具链影响。"),
    ("如何用 nm 查看共享库导出的符号？", "运行 nm -D --defined-only libexample.so；若符号被剥离或使用动态符号表，可结合 readelf --dyn-syms 查看。C++ 符号可用 c++filt 解码名称。"),
    ("ARM64 中 ADRP 和 ADD 常见组合是什么意思？", "ADRP 通常计算目标所在 4KB 页的 PC-relative 地址，随后 ADD 加上页内偏移得到完整地址。反汇编分析时应结合寄存器流和重定位信息，不能仅凭单条指令猜测目标对象。"),
    ("ARM64 的 BL 和 BR 指令有什么区别？", "BL 执行带链接的直接分支，并把返回地址写入 x30；BR 通过寄存器间接跳转，不自动写入新的链接寄存器。BLR 则是间接调用并更新 x30。"),
    ("Shell 中 set -euo pipefail 有什么作用？", "在 Bash 中，-e 尝试在未处理的失败命令处退出，-u 将未定义变量视为错误，pipefail 使管道返回失败命令的状态。它们有例外语义，不能替代显式检查和错误处理。"),
    ("Shell 变量为什么通常要加双引号？", "双引号可防止变量展开后的空格分词和通配符扩展。例如使用 rm -- \"$file\" 比未加引号安全，但破坏性命令仍应校验路径并先预览目标。"),
    ("如何安全地处理 Shell 命令参数中的空格？", "使用数组保存参数并通过双引号展开，例如 Bash 的 args=(\"$a\" \"$b\") 和 command \"${args[@]}\"。POSIX sh 没有 Bash 数组，可逐个正确引用参数或使用 set --。"),
    ("Android logcat 如何按进程过滤？", "先用 adb shell pidof 包名获取 PID，再执行 adb logcat --pid=<PID>。也可按 tag 和级别过滤。不同 Android 版本的 logcat 参数支持可能略有差异。"),
    ("CMake 如何为 Android NDK 指定 ABI？", "使用 Android Gradle Plugin 的 abiFilters，或独立 CMake 配置时指定 -DANDROID_ABI=arm64-v8a、-DANDROID_PLATFORM=android-<API> 和 NDK toolchain 文件。"),
    ("如何排查 dlopen 加载 .so 失败？", "记录 dlerror() 的完整信息；检查 ELF ABI、DT_NEEDED 依赖、符号缺失、命名空间限制、文件路径和权限。Android 上不能假设任意系统私有库都允许应用加载。"),
    ("什么是 ASLR？", "地址空间布局随机化会随机化进程关键内存区域的基址，增加攻击者预测地址的难度。分析崩溃或调试时应区分运行时地址与 ELF 文件偏移，并结合映射信息换算。"),
    ("如何从运行时地址换算 ELF 文件偏移？", "先根据 /proc/<pid>/maps 找到映射起始地址和文件偏移，再结合对应 PT_LOAD 段的 p_vaddr 与 p_offset 计算 load bias。必须确认地址属于哪个映射和哪个 ELF 段，不能直接把运行时地址当文件偏移。"),
    ("遇到技术问题时 AI 应如何避免编造偏移值？", "先要求明确二进制版本、架构、文件哈希和 dump 来源；只从实际文件或可复现输出提取数值；给出偏移所属基址和计算方法；没有证据时明确说无法确定，不猜测数值。"),
    ("如何检查一个二进制文件是否为 AArch64 ELF？", "执行 file 文件名，并用 readelf -h 文件名查看 Class、Data、Type 和 Machine。AArch64 通常显示 ELF64 与 AArch64；仍需确认 Android/Linux ABI 和动态依赖。"),
    ("C 语言整数溢出为什么危险？", "有符号整数溢出在 C 中属于未定义行为；无符号运算按模数回绕。分配数组前应检查 count > SIZE_MAX / sizeof(*ptr)，防止乘法溢出导致分配空间过小。"),
    ("如何写一个检查整数乘法溢出的 C 示例？", "#include <stdint.h>\\n#include <stddef.h>\\nint size_for_array(size_t n, size_t elem, size_t *out) {\\n    if (out == NULL) return -1;\\n    if (elem != 0 && n > SIZE_MAX / elem) return -1;\\n    *out = n * elem;\\n    return 0;\\n}"),
    ("git rebase 和 git merge 有什么区别？", "merge 通常创建合并提交并保留分支历史；rebase 将当前分支提交重放到新基线上，形成更线性的历史。共享分支上改写历史前应协调团队，避免覆盖他人提交。"),
    ("Gradle 构建报依赖解析失败怎么排查？", "检查仓库配置、网络代理、依赖坐标和版本；查看 --stacktrace 与 --info 输出；确认 settings.gradle 的 dependencyResolutionManagement 和 pluginManagement 仓库配置。不要盲目删除整个 Gradle 缓存。"),
    ("如何判断 APK 是否是 debug 构建？", "检查构建变体和 applicationIdSuffix，使用 apksigner 验证签名证书，并检查 manifest 的 debuggable 属性。文件名包含 debug 不能单独证明 APK 的真实构建配置。"),
    ("C++ 中 unique_ptr 和 shared_ptr 如何选择？", "默认优先 unique_ptr 表达唯一所有权；只有确实需要共享生命周期时才使用 shared_ptr。避免不必要的共享所有权和循环引用，循环结构可考虑 weak_ptr。"),
    ("如何分析 native crash 的 tombstone？", "先查看 signal、fault address、线程栈和 Build ID；使用与崩溃设备完全匹配的未剥离符号文件，通过 ndk-stack 或 llvm-symbolizer 符号化。不同版本二进制的符号不能混用。"),
    ("什么是 Build ID，为什么逆向分析要记录它？", "Build ID 是 ELF 中用于标识构建产物的标记之一。记录 Build ID、文件哈希、版本和 ABI 有助于确认符号文件与目标二进制是否匹配，但 Build ID 不应替代完整哈希验证。"),
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
