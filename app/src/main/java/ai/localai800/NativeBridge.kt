package ai.localai800

object NativeBridge {
    init {
        System.loadLibrary("localai800")
    }

    external fun nativeLoadModel(path: String, contextSize: Int, threads: Int): String
    external fun nativeGenerate(prompt: String, maxTokens: Int, temperature: Float): String
    external fun nativeStop()
    external fun nativeUnload()
}
