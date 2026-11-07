package ai.localai800

fun interface TokenCallback {
    fun onToken(text: String)
}

object NativeBridge {
    init {
        System.loadLibrary("localai800")
    }

    external fun nativeLoadModel(path: String, contextSize: Int, threads: Int): String
    external fun nativeGenerate(
        roles: Array<String>,
        contents: Array<String>,
        maxTokens: Int,
        temperature: Float,
        callback: TokenCallback
    ): String
    external fun nativeStop()
    external fun nativeUnload()
}
