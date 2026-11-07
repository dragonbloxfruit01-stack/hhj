#include <jni.h>
#include <llama.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {
std::mutex g_mutex;
std::atomic<bool> g_cancel{false};
llama_model * g_model = nullptr;
llama_context * g_context = nullptr;
llama_sampler * g_sampler = nullptr;
const llama_vocab * g_vocab = nullptr;
bool g_backend_initialized = false;

void release_model_locked() {
    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
}

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (!value) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring to_java(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(32);
    int32_t n = llama_token_to_piece(vocab, token, buffer.data(),
                                     static_cast<int32_t>(buffer.size()), 0, false);
    if (n < 0) {
        buffer.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, buffer.data(),
                                 static_cast<int32_t>(buffer.size()), 0, false);
    }
    if (n <= 0) return {};
    return std::string(buffer.data(), static_cast<size_t>(n));
}
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_localai800_NativeBridge_nativeLoadModel(
        JNIEnv * env, jobject, jstring path, jint contextSize, jint threads) {
    std::lock_guard<std::mutex> guard(g_mutex);
    release_model_locked();

    const std::string model_path = jstring_to_string(env, path);
    if (model_path.empty()) return to_java(env, "ERROR: 模型路径为空");

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // portable CPU path; no vendor-specific GPU assumptions
    g_model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!g_model) return to_java(env, "ERROR: 无法加载 GGUF。请确认文件完整且格式受支持。");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(std::clamp(contextSize, 512, 8192));
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = std::clamp(threads, 1, 8);
    ctx_params.n_threads_batch = std::clamp(threads, 1, 8);
    g_context = llama_init_from_model(g_model, ctx_params);
    if (!g_context) {
        release_model_locked();
        return to_java(env, "ERROR: 初始化推理上下文失败，可能是内存不足。请降低上下文长度。");
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        release_model_locked();
        return to_java(env, "ERROR: 读取模型词表失败");
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sampler_params);
    if (!g_sampler) {
        release_model_locked();
        return to_java(env, "ERROR: 初始化采样器失败");
    }
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.70f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return to_java(env, "模型加载成功。推理使用 CPU；首次生成可能需要一些时间。");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_localai800_NativeBridge_nativeGenerate(
        JNIEnv * env, jobject, jobjectArray roles, jobjectArray contents,
        jint maxTokens, jfloat temperature) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_model || !g_context || !g_sampler || !g_vocab) {
        return to_java(env, "ERROR: 请先加载 GGUF 模型");
    }
    if (!roles || !contents) return to_java(env, "ERROR: 对话内容为空");

    const jsize count = env->GetArrayLength(roles);
    if (count <= 0 || env->GetArrayLength(contents) != count) {
        return to_java(env, "ERROR: 对话记录格式不正确");
    }

    std::vector<std::string> role_storage;
    std::vector<std::string> content_storage;
    std::vector<llama_chat_message> messages;
    role_storage.reserve(static_cast<size_t>(count));
    content_storage.reserve(static_cast<size_t>(count));
    messages.reserve(static_cast<size_t>(count));

    for (jsize i = 0; i < count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
        role_storage.push_back(jstring_to_string(env, role));
        content_storage.push_back(jstring_to_string(env, content));
        if (role) env->DeleteLocalRef(role);
        if (content) env->DeleteLocalRef(content);
    }
    for (jsize i = 0; i < count; ++i) {
        messages.push_back({
            const_cast<char *>(role_storage[static_cast<size_t>(i)].c_str()),
            const_cast<char *>(content_storage[static_cast<size_t>(i)].c_str())
        });
    }

    const char * chat_template = llama_model_chat_template(g_model, nullptr);
    if (!chat_template) return to_java(env, "ERROR: 模型没有可用的聊天模板");

    size_t capacity = 1024;
    for (const auto & s : content_storage) capacity += s.size() * 2 + 64;
    for (const auto & s : role_storage) capacity += s.size() + 16;
    std::vector<char> formatted(capacity);
    int32_t formatted_size = llama_chat_apply_template(
        chat_template, messages.data(), messages.size(), true,
        formatted.data(), static_cast<int32_t>(formatted.size()));
    if (formatted_size < 0) {
        return to_java(env, "ERROR: 无法应用该模型的聊天模板。请尝试使用 Instruct GGUF。");
    }
    if (static_cast<size_t>(formatted_size) >= formatted.size()) {
        formatted.resize(static_cast<size_t>(formatted_size) + 1);
        formatted_size = llama_chat_apply_template(
            chat_template, messages.data(), messages.size(), true,
            formatted.data(), static_cast<int32_t>(formatted.size()));
    }
    if (formatted_size < 0) return to_java(env, "ERROR: 聊天模板格式化失败");

    const std::string prompt(formatted.data(), static_cast<size_t>(formatted_size));
    std::vector<llama_token> tokens(prompt.size() + 16);
    int32_t token_count = llama_tokenize(
        g_vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (token_count < 0) {
        tokens.resize(static_cast<size_t>(-token_count));
        token_count = llama_tokenize(
            g_vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
            tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    }
    if (token_count <= 0) return to_java(env, "ERROR: 提示词分词失败");
    tokens.resize(static_cast<size_t>(token_count));

    const uint32_t n_ctx = llama_n_ctx(g_context);
    if (tokens.size() + 8 >= n_ctx) {
        return to_java(env, "ERROR: 对话超过上下文长度，请清空聊天记录后重试");
    }

    llama_memory_clear(llama_get_memory(g_context), true);
    llama_sampler_reset(g_sampler);
    g_cancel.store(false);

    llama_batch batch = llama_batch_init(token_count + 1, 0, 1);
    if (!batch.token || !batch.pos || !batch.logits) {
        llama_batch_free(batch);
        return to_java(env, "ERROR: 分配推理批次失败");
    }

    batch.n_tokens = token_count;
    for (int32_t i = 0; i < token_count; ++i) {
        batch.token[i] = tokens[static_cast<size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == token_count - 1) ? 1 : 0;
    }

    if (llama_decode(g_context, batch) != 0) {
        llama_batch_free(batch);
        return to_java(env, "ERROR: 模型计算失败，可能是内存不足或模型不兼容");
    }

    std::string output;
    const int limit = std::clamp(maxTokens, 1, 1024);
    const float temp = std::clamp(temperature, 0.05f, 2.0f);
    // Rebuild sampler temperature for this request while retaining deterministic-safe defaults.
    llama_sampler_free(g_sampler);
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    for (int i = 0; i < limit && !g_cancel.load(); ++i) {
        const llama_token token = llama_sampler_sample(g_sampler, g_context, batch.n_tokens - 1);
        if (llama_vocab_is_eog(g_vocab, token)) break;

        output += token_piece(g_vocab, token);

        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = token_count + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(g_context, batch) != 0) {
            output += "\n[推理中断：模型计算失败]";
            break;
        }
    }
    llama_batch_free(batch);
    if (g_cancel.load()) output += "\n[已停止生成]";
    return to_java(env, output);
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_localai800_NativeBridge_nativeStop(JNIEnv *, jobject) {
    g_cancel.store(true);
}

extern "C"
JNIEXPORT void JNICALL
Java_ai_localai800_NativeBridge_nativeUnload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> guard(g_mutex);
    release_model_locked();
}

extern "C"
JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM *, void *) {
    return JNI_VERSION_1_6;
}
