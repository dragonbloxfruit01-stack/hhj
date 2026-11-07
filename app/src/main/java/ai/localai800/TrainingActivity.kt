package ai.localai800

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.BatteryManager
import android.provider.DocumentsContract
import android.database.Cursor
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.io.FileOutputStream

class TrainingActivity : Activity() {
    private lateinit var status: TextView
    private var modelPath: String? = null
    private var dataPath: String? = null
    private val prefs by lazy { getSharedPreferences("local_ai_800_training", MODE_PRIVATE) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelPath = prefs.getString("model_dir", null)
        dataPath = prefs.getString("dataset_file", null)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(22, 28, 22, 20)
        }
        root.addView(TextView(this).apply {
            text = "Local AI 800 · 手机本地训练"
            textSize = 23f
        })
        root.addView(TextView(this).apply {
            text = "训练只在手机本机运行。首版支持 Qwen2.5 系列 BPE + SafeTensors 模型目录。GGUF 只能聊天，不能直接用于训练。"
            textSize = 14f
            setPadding(0, 12, 0, 14)
        })
        status = TextView(this).apply {
            textSize = 14f
            setPadding(12, 12, 12, 12)
            setBackgroundColor(0xFFF2F4F7.toInt())
        }
        root.addView(status)
        root.addView(Button(this).apply {
            text = "1. 选择模型文件夹（config/tokenizer/safetensors）"
            setOnClickListener {
                val i = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
                startActivityForResult(i, PICK_MODEL)
            }
        })
        root.addView(Button(this).apply {
            text = "2. 选择训练数据 JSONL"
            setOnClickListener {
                val i = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }
                startActivityForResult(i, PICK_DATA)
            }
        })
        root.addView(Button(this).apply {
            text = "开始训练"
            setOnClickListener { startTraining(false) }
        })
        root.addView(Button(this).apply {
            text = "继续 / 从断点恢复"
            setOnClickListener { startTraining(true) }
        })
        root.addView(Button(this).apply {
            text = "暂停训练"
            setOnClickListener { serviceAction(LocalTrainingService.ACTION_PAUSE) }
        })
        root.addView(Button(this).apply {
            text = "恢复运行"
            setOnClickListener { serviceAction(LocalTrainingService.ACTION_RESUME) }
        })
        root.addView(Button(this).apply {
            text = "停止并保存检查点"
            setOnClickListener { serviceAction(LocalTrainingService.ACTION_STOP) }
        })
        root.addView(Button(this).apply {
            text = "返回聊天"
            setOnClickListener { finish() }
        })
        setContentView(root)
        refreshStatus()
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 8033)
        }
    }

    private fun refreshStatus() {
        val model = modelPath?.let { File(it) }
        val data = dataPath?.let { File(it) }
        val statusText = prefs.getString("training_status", "尚未开始训练") ?: "尚未开始训练"
        val step = prefs.getInt("training_step", 0)
        status.text = "模型：${model?.name ?: "未选择"}\n数据：${data?.name ?: "未选择"}\n状态：$statusText\n已完成步数：$step"
    }

    private fun startTraining(resume: Boolean) {
        val model = modelPath?.let(::File)
        val data = dataPath?.let(::File)
        if (model == null || !model.isDirectory || data == null || !data.isFile) {
            Toast.makeText(this, "请先选择模型文件夹和 JSONL 数据", Toast.LENGTH_LONG).show()
            return
        }
        val checkpoint = File(filesDir, "training/checkpoint/latest")
        val intent = Intent(this, LocalTrainingService::class.java).apply {
            action = LocalTrainingService.ACTION_START
            putExtra(LocalTrainingService.EXTRA_MODEL_DIR, model.absolutePath)
            putExtra(LocalTrainingService.EXTRA_DATASET, data.absolutePath)
            putExtra(LocalTrainingService.EXTRA_CHECKPOINT, checkpoint.absolutePath)
            putExtra(LocalTrainingService.EXTRA_MAX_SEQUENCE, 128)
            putExtra(LocalTrainingService.EXTRA_EPOCHS, 1)
            putExtra(LocalTrainingService.EXTRA_RESUME, resume)
        }
        try {
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(intent) else startService(intent)
            Toast.makeText(this, if (resume) "正在尝试恢复训练" else "已启动本地训练", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "启动失败：${e.message}", Toast.LENGTH_LONG).show()
        }
        refreshStatus()
    }

    private fun serviceAction(action: String) {
        try {
            startService(Intent(this, LocalTrainingService::class.java).setAction(action))
            Toast.makeText(this, "已发送训练控制指令", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "操作失败：${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @Deprecated("SAF result handler")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK || data?.data == null) return
        val uri = data.data!!
        when (requestCode) {
            PICK_MODEL -> {
                try {
                    contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    status.text = "正在导入模型目录…请勿关闭应用"
                    Thread {
                        try {
                            val target = File(filesDir, "training/model")
                            target.deleteRecursively()
                            target.mkdirs()
                            copyDocumentTree(uri, target)
                            runOnUiThread {
                                modelPath = target.absolutePath
                                prefs.edit().putString("model_dir", modelPath).apply()
                                refreshStatus()
                                Toast.makeText(this, "模型目录导入完成", Toast.LENGTH_LONG).show()
                            }
                        } catch (e: Exception) {
                            runOnUiThread { Toast.makeText(this, "模型导入失败：${e.message}", Toast.LENGTH_LONG).show() }
                        }
                    }.start()
                } catch (e: Exception) {
                    Toast.makeText(this, "无法读取模型目录：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
            PICK_DATA -> {
                try {
                    contentResolver.openInputStream(uri).use { input ->
                        requireNotNull(input) { "无法打开数据文件" }
                        val dest = File(filesDir, "training/dataset.jsonl")
                        dest.parentFile?.mkdirs()
                        FileOutputStream(dest).use { output -> input.copyTo(output, 1024 * 1024) }
                        dataPath = dest.absolutePath
                        prefs.edit().putString("dataset_file", dataPath).apply()
                    }
                    refreshStatus()
                    Toast.makeText(this, "训练数据导入完成", Toast.LENGTH_SHORT).show()
                } catch (e: Exception) {
                    Toast.makeText(this, "数据导入失败：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun copyDocumentTree(tree: Uri, destination: File) {
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        copyChildren(tree, rootId, destination)
    }

    private fun copyChildren(tree: Uri, parentId: String, destination: File) {
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId)
        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )
        contentResolver.query(children, projection, null, null, null)?.use { cursor ->
            val idCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeCol = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
            while (cursor.moveToNext()) {
                val id = cursor.getString(idCol)
                val name = cursor.getString(nameCol).orEmpty().replace("/", "_").replace("\\\\", "_")
                if (name.isBlank() || name == "." || name == "..") continue
                val mime = cursor.getString(mimeCol)
                val target = File(destination, name)
                if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                    target.mkdirs()
                    copyChildren(tree, id, target)
                } else {
                    val doc = DocumentsContract.buildDocumentUriUsingTree(tree, id)
                    contentResolver.openInputStream(doc).use { input ->
                        if (input != null) FileOutputStream(target).use { output -> input.copyTo(output, 1024 * 1024) }
                    }
                }
            }
        } ?: throw IllegalStateException("无法读取所选目录")
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    companion object {
        private const val PICK_MODEL = 8101
        private const val PICK_DATA = 8102
    }
}
