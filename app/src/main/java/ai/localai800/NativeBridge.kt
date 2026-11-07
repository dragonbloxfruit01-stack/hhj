package ai.localai800

object NativeBridge {
    init {
        System.loadLibrary("localai800")
    }

    external fun backendStatus(): String
}
