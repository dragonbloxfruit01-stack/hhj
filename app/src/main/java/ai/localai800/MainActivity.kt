package ai.localai800

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 36, 28, 24)
            gravity = Gravity.TOP
        }

        val title = TextView(this).apply {
            text = "Local AI 800"
            textSize = 28f
        }
        val subtitle = TextView(this).apply {
            text = "手机本地 AI · Android 16"
            textSize = 16f
            setPadding(0, 8, 0, 24)
        }
        val status = TextView(this).apply {
            textSize = 15f
            text = try {
                NativeBridge.backendStatus()
            } catch (e: UnsatisfiedLinkError) {
                "原生模块加载失败：${e.message}"
            }
            setPadding(0, 8, 0, 20)
        }
        val note = TextView(this).apply {
            text = "当前为可构建的工程骨架。尚未集成 GGUF 推理引擎，因此暂时不能加载模型或进行离线对话。"
            textSize = 15f
        }
        val info = Button(this).apply {
            text = "项目状态"
            setOnClickListener {
                status.text = try {
                    NativeBridge.backendStatus()
                } catch (e: Exception) {
                    "状态读取失败：${e.message}"
                }
            }
        }

        root.addView(title)
        root.addView(subtitle)
        root.addView(status)
        root.addView(note)
        root.addView(info)
        setContentView(root)
    }
}
