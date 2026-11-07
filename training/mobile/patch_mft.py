#!/usr/bin/env python3
"""Apply Local AI 800 mobile-training extensions to the checked-out MobileFineTuner source."""
from pathlib import Path
R = Path("MobileFineTuner")

def replace(path, old, new):
    p = R / path
    s = p.read_text()
    if new in s: return
    if old not in s: raise RuntimeError("Patch anchor missing: " + path + " :: " + old[:70])
    p.write_text(s.replace(old, new, 1))

replace("operator/finetune_ops/optim/adam.h",
"    void load_state(const std::string& path) override;\n",
"    void load_state(const std::string& path) override;\n    void save_state(const std::string& path, const std::vector<TensorPtr>& params) const;\n    void load_state(const std::string& path, const std::vector<TensorPtr>& params);\n")

replace("operator/finetune_ops/optim/adam.cpp",
"void Adam::update_param(TensorPtr /* param */, const TensorPtr& /* grad */, OptimizerState& /* state */) {",
r'''void Adam::save_state(const std::string& path, const std::vector<TensorPtr>& params) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write Adam checkpoint");
    const char magic[8] = {'M','F','T','A','D','A','M','2'};
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&adam_config_.learning_rate), sizeof(float));
    f.write(reinterpret_cast<const char*>(&adam_config_.beta1), sizeof(float));
    f.write(reinterpret_cast<const char*>(&adam_config_.beta2), sizeof(float));
    f.write(reinterpret_cast<const char*>(&adam_config_.epsilon), sizeof(float));
    f.write(reinterpret_cast<const char*>(&adam_config_.weight_decay), sizeof(float));
    f.write(reinterpret_cast<const char*>(&adam_config_.amsgrad), sizeof(bool));
    uint64_t count = params.size(); f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& p : params) {
        if (!p) throw std::runtime_error("Null Adam checkpoint parameter");
        auto it = states_.find(p);
        const size_t step = it == states_.end() ? 0 : it->second.step;
        uint32_t rank = static_cast<uint32_t>(p->shape().size());
        f.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (auto d : p->shape()) f.write(reinterpret_cast<const char*>(&d), sizeof(d));
        f.write(reinterpret_cast<const char*>(&step), sizeof(step));
        uint64_t n = static_cast<uint64_t>(p->numel());
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        if (it == states_.end()) {
            std::vector<float> zeros(static_cast<size_t>(n), 0.0f);
            f.write(reinterpret_cast<const char*>(zeros.data()), n * sizeof(float));
            f.write(reinterpret_cast<const char*>(zeros.data()), n * sizeof(float));
            if (adam_config_.amsgrad) f.write(reinterpret_cast<const char*>(zeros.data()), n * sizeof(float));
        } else {
            const auto& st = it->second;
            f.write(reinterpret_cast<const char*>(st.m.at(0)->data<float>()), n * sizeof(float));
            f.write(reinterpret_cast<const char*>(st.v.at(0)->data<float>()), n * sizeof(float));
            if (adam_config_.amsgrad) f.write(reinterpret_cast<const char*>(st.v_hat.at(0)->data<float>()), n * sizeof(float));
        }
    }
    if (!f.good()) throw std::runtime_error("Adam checkpoint write failed");
}

void Adam::load_state(const std::string& path, const std::vector<TensorPtr>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read Adam checkpoint");
    char magic[8]{}; const char expected[8] = {'M','F','T','A','D','A','M','2'};
    f.read(magic, 8);
    if (!f || std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("Invalid Adam checkpoint");
    f.read(reinterpret_cast<char*>(&adam_config_.learning_rate), sizeof(float));
    f.read(reinterpret_cast<char*>(&adam_config_.beta1), sizeof(float));
    f.read(reinterpret_cast<char*>(&adam_config_.beta2), sizeof(float));
    f.read(reinterpret_cast<char*>(&adam_config_.epsilon), sizeof(float));
    f.read(reinterpret_cast<char*>(&adam_config_.weight_decay), sizeof(float));
    f.read(reinterpret_cast<char*>(&adam_config_.amsgrad), sizeof(bool));
    uint64_t count = 0; f.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!f || count != params.size()) throw std::runtime_error("Adam parameter count mismatch");
    states_.clear();
    for (const auto& p : params) {
        uint32_t rank = 0; f.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        if (!f || rank != p->shape().size()) throw std::runtime_error("Adam tensor rank mismatch");
        for (uint32_t i=0; i<rank; ++i) {
            int64_t dim=0; f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            if (!f || dim != p->shape()[i]) throw std::runtime_error("Adam tensor shape mismatch");
        }
        size_t step=0; uint64_t n=0;
        f.read(reinterpret_cast<char*>(&step), sizeof(step)); f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!f || n != static_cast<uint64_t>(p->numel())) throw std::runtime_error("Adam tensor size mismatch");
        init_state(p); auto& st = states_.at(p); st.step=step;
        f.read(reinterpret_cast<char*>(st.m.at(0)->data<float>()), n*sizeof(float));
        f.read(reinterpret_cast<char*>(st.v.at(0)->data<float>()), n*sizeof(float));
        if (adam_config_.amsgrad) f.read(reinterpret_cast<char*>(st.v_hat.at(0)->data<float>()), n*sizeof(float));
        if (!f) throw std::runtime_error("Truncated Adam checkpoint");
    }
}

void Adam::update_param(TensorPtr /* param */, const TensorPtr& /* grad */, OptimizerState& /* state */) {''')

replace("operator/finetune_ops/optim/auto_trainer.h",
"    AutoTrainStepResult train_step(const TensorPtr& input_ids,\n                                   const TensorPtr& attention_mask,\n                                   const TensorPtr& labels);\n",
"    AutoTrainStepResult train_step(const TensorPtr& input_ids,\n                                   const TensorPtr& attention_mask,\n                                   const TensorPtr& labels);\n    void save_checkpoint(const std::string& path);\n    void load_checkpoint(const std::string& path);\n")

replace("operator/finetune_ops/optim/auto_trainer.cpp",
"#include <vector>\n",
"#include <vector>\n#include <fstream>\n#include <cstdint>\n#include <cstring>\n")

replace("operator/finetune_ops/optim/auto_trainer.cpp",
"    return result;\n}\n\n}  // namespace ops\n",
r'''    return result;
}

void AutoTrainer::save_checkpoint(const std::string& path) {
    auto params = model_.trainable_parameters();
    std::ofstream f(path + ".weights", std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write LoRA checkpoint");
    const char magic[8] = {'M','F','T','L','O','R','A','2'};
    f.write(magic, 8);
    uint64_t count=params.size(); f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& p : params) {
        if (!p || p->dtype()!=kFloat32) throw std::runtime_error("LoRA checkpoint requires FP32 parameters");
        uint32_t rank=static_cast<uint32_t>(p->shape().size()); f.write(reinterpret_cast<const char*>(&rank),sizeof(rank));
        for(auto d:p->shape()) f.write(reinterpret_cast<const char*>(&d),sizeof(d));
        uint64_t n=static_cast<uint64_t>(p->numel()); f.write(reinterpret_cast<const char*>(&n),sizeof(n));
        f.write(reinterpret_cast<const char*>(p->data<float>()),n*sizeof(float));
    }
    if(!f.good()) throw std::runtime_error("LoRA checkpoint write failed");
    f.close();
    optimizer_->save_state(path + ".adam", params);
}

void AutoTrainer::load_checkpoint(const std::string& path) {
    auto params = model_.trainable_parameters();
    std::ifstream f(path + ".weights", std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read LoRA checkpoint");
    char magic[8]{}; const char expected[8]={'M','F','T','L','O','R','A','2'};
    f.read(magic,8); if(!f || std::memcmp(magic,expected,8)!=0) throw std::runtime_error("Invalid LoRA checkpoint");
    uint64_t count=0; f.read(reinterpret_cast<char*>(&count),sizeof(count));
    if(!f || count!=params.size()) throw std::runtime_error("LoRA parameter count mismatch");
    for(const auto& p:params) {
        uint32_t rank=0; f.read(reinterpret_cast<char*>(&rank),sizeof(rank));
        if(!f || rank!=p->shape().size()) throw std::runtime_error("LoRA tensor rank mismatch");
        for(uint32_t i=0;i<rank;++i) { int64_t d=0; f.read(reinterpret_cast<char*>(&d),sizeof(d)); if(!f || d!=p->shape()[i]) throw std::runtime_error("LoRA tensor shape mismatch"); }
        uint64_t n=0; f.read(reinterpret_cast<char*>(&n),sizeof(n));
        if(!f || n!=static_cast<uint64_t>(p->numel())) throw std::runtime_error("LoRA tensor size mismatch");
        f.read(reinterpret_cast<char*>(p->data<float>()),n*sizeof(float));
        if(!f) throw std::runtime_error("Truncated LoRA checkpoint");
    }
    optimizer_->load_state(path + ".adam", params);
}

}  // namespace ops
''')

java = R / "android-visualizer/mft-sdk/src/main/java/com/mobilefinetuner/sdk/MobileFineTuner.java"
s = java.read_text()
if "saveCheckpoint(String path)" not in s:
    s=s.replace("    public int trainableTensorCount() {\n        return nativeTrainableTensorCount(requireOpen());\n    }\n",
"""    public int trainableTensorCount() { return nativeTrainableTensorCount(requireOpen()); }
    public void saveCheckpoint(String path) { nativeSaveCheckpoint(requireOpen(), path); }
    public void loadCheckpoint(String path) { nativeLoadCheckpoint(requireOpen(), path); }
    public static int[] encodeText(String modelDir, String text, int maxLength) {
        return nativeEncodeText(modelDir, text, maxLength);
    }
""")
    s=s.replace("    private static native int nativeTrainableTensorCount(long handle);\n    private static native void nativeClose(long handle);\n",
"""    private static native int nativeTrainableTensorCount(long handle);
    private static native void nativeSaveCheckpoint(long handle, String path);
    private static native void nativeLoadCheckpoint(long handle, String path);
    private static native int[] nativeEncodeText(String modelDir, String text, int maxLength);
    private static native void nativeClose(long handle);
""")
    java.write_text(s)

cpp = R / "android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp"
s=cpp.read_text()
if "nativeSaveCheckpoint" not in s:
    anchor='extern "C" JNIEXPORT void JNICALL\nJava_com_mobilefinetuner_sdk_MobileFineTuner_nativeClose(JNIEnv* env, jclass, jlong handle) {'
    addition=r'''extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeSaveCheckpoint(JNIEnv* env, jclass, jlong handle, jstring path) {
    guarded_void(env, [&]() { auto* s=require_session(env,handle); if(!s || !s->trainer) throw std::runtime_error("Trainer not initialized"); s->trainer->save_checkpoint(to_string(env,path)); });
}
extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeLoadCheckpoint(JNIEnv* env, jclass, jlong handle, jstring path) {
    guarded_void(env, [&]() { auto* s=require_session(env,handle); if(!s || !s->trainer) throw std::runtime_error("Trainer not initialized"); s->trainer->load_checkpoint(to_string(env,path)); });
}
extern "C" JNIEXPORT jintArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeEncodeText(JNIEnv* env, jclass, jstring modelDir, jstring text, jint maxLength) {
    return guarded(env, [&]() -> jintArray {
        auto tokenizer=ops::TokenizerFactory::from_pretrained(to_string(env,modelDir));
        auto ids=tokenizer->encode(to_string(env,text));
        if(maxLength>0 && ids.size()>static_cast<size_t>(maxLength)) ids.resize(static_cast<size_t>(maxLength));
        jintArray out=env->NewIntArray(static_cast<jsize>(ids.size()));
        if(out && !ids.empty()) { std::vector<jint> values(ids.begin(),ids.end()); env->SetIntArrayRegion(out,0,static_cast<jsize>(values.size()),values.data()); }
        return out;
    }, static_cast<jintArray>(nullptr));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeClose(JNIEnv* env, jclass, jlong handle) {'''
    if anchor not in s: raise RuntimeError("JNI close anchor missing")
    cpp.write_text(s.replace(anchor,addition,1))

replace("android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp",
"#include <vector>\n",
"#include <vector>\n#include <unordered_map>\n")
replace("android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp",
"namespace {\n\nstruct NativeSession",
"namespace {\nstd::mutex g_tokenizer_mutex;\nstd::unordered_map<std::string, std::unique_ptr<ops::Tokenizer>> g_tokenizers;\n\nstruct NativeSession")
replace("android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp",
'''        auto tokenizer=ops::TokenizerFactory::from_pretrained(to_string(env,modelDir));
        auto ids=tokenizer->encode(to_string(env,text));''',
'''        const std::string dir=to_string(env,modelDir);
        std::lock_guard<std::mutex> lock(g_tokenizer_mutex);
        auto& tokenizer=g_tokenizers[dir];
        if(!tokenizer) tokenizer=ops::TokenizerFactory::from_pretrained(dir);
        auto ids=tokenizer->encode(to_string(env,text));''')

print("MobileFineTuner mobile training extensions applied.")
