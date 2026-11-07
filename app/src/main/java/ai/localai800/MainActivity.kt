package ai.localai800

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private data class Message(val role: String, val content: String)

    private val worker = Executors.newSingleThreadExecutor()
    private val messages = mutableListOf<Message>()
    private lateinit var status: TextView
    private lateinit var chat: TextView
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var stopButton: Button
    private var modelLoaded = false
    private var busy = false
    private var modelFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        restoreSavedModel()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 24, 20, 16)
        }

        val title = TextView(this).apply {
            text = "Local AI 800"
            textSize = 27f
            setTextColor(0xFF172033.toInt())
        }
        val subtitle = TextView(this).apply {
            text = "完全本地运行 · GGUF · llama.cpp"
            textSize = 14f
            setTextColor(0xFF667085.toInt())
            setPadding(0, 4, 0, 14)
        }
        status = TextView(this).apply {
            text = "尚未加载模型。选择一个 GGUF Instruct 模型开始使用。"
            textSize = 14f
            setTextColor(0xFF344054.toInt())
            setPadding(12, 12, 12, 12)
            setBackgroundColor(0xFFF2F4F7.toInt())
        }

        val choose = Button(this).apply {
            text = "选择 / 导入 GGUF 模型"
            setOnClickListener { chooseModel() }
        }

        chat = TextView(this).apply {
            text = "欢迎使用 Local AI 800。\n\n先导入 GGUF 模型，然后就可以离线聊天。"
            textSize = 15f
            setTextColor(0xFF101828.toInt())
            setPadding(12, 12, 12, 12)
            textIsSelectable = true
        }
        val chatScroll = ScrollView(this).apply {
            addView(chat)
            isFillViewport = true
        }

        input = EditText(this).apply {
            hint = "输入问题，例如：用 C 写一个 ARM64 ELF 解析器"
            minLines = 2
            maxLines = 5
            textSize = 15f
            gravity = Gravity.TOP
        }

        val actions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        sendButton = Button(this).apply {
            text = "发送"
            isEnabled = false
            setOnClickListener { sendMessage() }
        }
        stopButton = Button(this).apply {
            text = "停止"
            isEnabled = false
            setOnClickListener {
                NativeBridge.nativeStop()
                status.text = "正在请求停止生成…"
                stopButton.isEnabled = false
            }
        }
        val clear = Button(this).apply {
            text = "清空对话"
            setOnClickListener {
                if (!busy) {
                    resetConversation()
                    chat.text = "对话已清空。"
                }
            }
        }
        actions.addView(sendButton, LinearLayout.LayoutParams(0, -2, 1f))
        actions.addView(stopButton, LinearLayout.LayoutParams(0, -2, 1f))
        actions.addView(clear, LinearLayout.LayoutParams(0, -2, 1f))

        root.addView(title)
        root.addView(subtitle)
        root.addView(status)
        root.addView(choose)
        root.addView(chatScroll, LinearLayout.LayoutParams(-1, 0, 1f).apply {
            topMargin = 12
            bottomMargin = 8
        })
        root.addView(input)
        root.addView(actions)
        setContentView(root)
    }


    private fun restoreSavedModel() {
        val path = getSharedPreferences(PREFS, MODE_PRIVATE).getString(KEY_MODEL_PATH, null) ?: return
        val file = File(path)
        if (!file.isFile || file.length() == 0L) {
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().remove(KEY_MODEL_PATH).apply()
            return
        }
        status.text = "发现已保存的 GGUF 模型，正在重新加载…"
        worker.execute {
            val cpuThreads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 6)
            val result = NativeBridge.nativeLoadModel(file.absolutePath, 2048, cpuThreads)
            runOnUiThread {
                if (result.startsWith("ERROR:")) {
                    modelLoaded = false
                    status.text = result
                    sendButton.isEnabled = false
                } else {
                    modelFile = file
                    modelLoaded = true
                    resetConversation()
                    status.text = result + "\\n已自动恢复本地模型 · 上下文：2048 tokens"
                    sendButton.isEnabled = true
                    chat.text = "本地模型已恢复。可以继续开始新对话。"
                }
            }
        }
    }

    private fun chooseModel() {
        if (busy) return
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }
        startActivityForResult(intent, REQUEST_MODEL)
    }

    @Deprecated("Use Activity Result API in a future UI revision")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_MODEL || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        status.text = "正在导入模型文件，请保持应用在前台…"
        sendButton.isEnabled = false
        worker.execute {
            try {
                val modelsDir = File(filesDir, "models").apply { mkdirs() }
                val destination = File(modelsDir, "model-${System.currentTimeMillis()}.gguf")
                contentResolver.openInputStream(uri).use { source ->
                    requireNotNull(source) { "无法打开所选文件" }
                    destination.outputStream().use { target -> source.copyTo(target, 1024 * 1024) }
                }
                runOnUiThread {
                    modelFile = destination
                    status.text = "文件已导入：${destination.name}（${destination.length() / (1024 * 1024)} MB），正在加载…"
                }
                val cpuThreads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 6)
                val previousModel = modelFile
                val result = NativeBridge.nativeLoadModel(destination.absolutePath, 2048, cpuThreads)
                if (previousModel != null && previousModel != destination) previousModel.delete()
                runOnUiThread {
                    if (result.startsWith("ERROR:")) {
                        modelLoaded = false
                        status.text = result
                        sendButton.isEnabled = false
                    } else {
                        modelLoaded = true
                        resetConversation()
                        status.text = result + "\n上下文：2048 tokens · CPU 推理 · 模型文件保存在应用私有目录"
                        sendButton.isEnabled = true
                        chat.text = "模型已就绪。可以开始中文聊天、C/C++、Shell、Android NDK 和技术分析。"
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "导入失败：${e.message}"
                    sendButton.isEnabled = false
                }
            }
        }
    }

    private fun sendMessage() {
        if (!modelLoaded || busy) return
        val question = input.text.toString().trim()
        if (question.isEmpty()) return
        input.setText("")
        messages.add(Message("user", question))
        appendChat("\n\n你：\n$question")
        busy = true
        sendButton.isEnabled = false
        stopButton.isEnabled = true
        status.text = "模型正在思考…（全部计算在本机进行）"

        val roles = messages.map { it.role }.toTypedArray()
        val contents = messages.map { it.content }.toTypedArray()
        appendChat("\n\nAI：\n")
        worker.execute {
            val callback = TokenCallback { piece ->
                runOnUiThread { appendChat(piece) }
            }
            val answer = try {
                NativeBridge.nativeGenerate(roles, contents, 384, 0.7f, callback)
            } catch (e: Exception) {
                "ERROR: ${e.message}"
            }
            runOnUiThread {
                val failed = answer.startsWith("ERROR:")
                if (!failed && answer.isNotBlank()) {
                    messages.add(Message("assistant", answer))
                } else {
                    appendChat("\n\n系统：\n$answer")
                }
                busy = false
                sendButton.isEnabled = modelLoaded
                stopButton.isEnabled = false
                status.text = if (failed) answer else "就绪 · 离线推理 · ${modelFile?.name ?: "模型已加载"}"
            }
        }
    }

    private fun appendChat(text: String) {
        chat.append(text)
        (chat.parent as? ScrollView)?.post { (chat.parent as? ScrollView)?.fullScroll(View.FOCUS_DOWN) }
    }

    private fun resetConversation() {
        messages.clear()
        messages.add(Message("system", "你是 Local AI 800，一个中文优先的本地技术助手。擅长 C/C++、Shell、Android NDK、APK 构建、Linux、ARM64、ELF、调试和逻辑分析。回答准确、分步骤；不确定时明确说明，不编造测试结果。"))
    }

    override fun onDestroy() {
        NativeBridge.nativeStop()
        worker.execute { NativeBridge.nativeUnload() }
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_MODEL = 8001
        private const val PREFS = "local_ai_800"
        private const val KEY_MODEL_PATH = "model_path"
    }
}
