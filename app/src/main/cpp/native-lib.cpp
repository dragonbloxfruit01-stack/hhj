#include <jni.h>
#include <string>

extern "C"
JNIEXPORT jstring JNICALL
Java_ai_localai800_NativeBridge_backendStatus(JNIEnv *env, jobject /* thiz */) {
    const std::string status =
        "C++ 原生桥接已加载。GGUF 推理后端尚未集成。";
    return env->NewStringUTF(status.c_str());
}
