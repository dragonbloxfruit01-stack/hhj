package ai.localai800

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.os.BatteryManager
import android.content.Context
import org.json.JSONObject
import com.mobilefinetuner.sdk.MobileFineTuner
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.FileWriter
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * User-started on-device LoRA training service.
 * Model weights, tokenizer and JSONL data are read from app-private local storage.
 */
class LocalTrainingService : Service() {
    private val stopRequested = AtomicBoolean(false)
    private val pauseRequested = AtomicBoolean(false)
    @Volatile private var session: MobileFineTuner? = null
    @Volatile private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var modelDir: File
    private lateinit var datasetFile: File
    private lateinit var checkpointPrefix: File
    private var maxSequence = 128
    private var epochs = 1
    private var resume = false
    private var currentStep = 0
    private var currentEpoch = 0
    private var currentLine = 0
    private var lastLoss = Float.NaN

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_PAUSE -> {
                pauseRequested.set(true)
                updateNotification("正在保存检查点并暂停…")
                return START_NOT_STICKY
            }
            ACTION_RESUME -> {
                pauseRequested.set(false)
                if (worker?.isAlive != true) {
                    val p = getSharedPreferences("local_ai_800_training", MODE_PRIVATE)
                    val model = p.getString("model_dir", null)
                    val data = p.getString("dataset_file", null)
                    if (model != null && data != null) {
                        modelDir = File(model)
                        datasetFile = File(data)
                        checkpointPrefix = File(filesDir, "training/checkpoint/latest")
                        maxSequence = 128
                        epochs = 1
                        resume = true
                        stopRequested.set(false)
                        startForeground(NOTIFICATION_ID, buildNotification("正在从本地检查点恢复…"))
                        p.edit().putBoolean("service_running", true).apply()
                        acquireWakeLock()
                        worker = thread(name = "LocalAI800-Training") { runTraining() }
                    }
                } else {
                    updateNotification("训练继续运行")
                }
                return START_NOT_STICKY
            }
            ACTION_STOP -> {
                stopRequested.set(true)
                pauseRequested.set(false)
                updateNotification("正在保存并停止…")
                return START_NOT_STICKY
            }
            ACTION_START -> {
                if (worker?.isAlive == true) return START_NOT_STICKY
                modelDir = File(intent.getStringExtra(EXTRA_MODEL_DIR) ?: return START_NOT_STICKY)
                datasetFile = File(intent.getStringExtra(EXTRA_DATASET) ?: return START_NOT_STICKY)
                checkpointPrefix = File(intent.getStringExtra(EXTRA_CHECKPOINT) ?: return START_NOT_STICKY)
                maxSequence = intent.getIntExtra(EXTRA_MAX_SEQUENCE, 128).coerceIn(32, 512)
                epochs = intent.getIntExtra(EXTRA_EPOCHS, 1).coerceIn(1, 10)
                resume = intent.getBooleanExtra(EXTRA_RESUME, false)
                stopRequested.set(false)
                pauseRequested.set(false)
                currentStep = 0
                currentEpoch = 0
                currentLine = 0
                startForeground(NOTIFICATION_ID, buildNotification("准备本地训练…"))
                getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("service_running", true).apply()
                acquireWakeLock()
                worker = thread(name = "LocalAI800-Training") { runTraining() }
            }
        }
        return START_NOT_STICKY
    }

    private fun runTraining() {
        var trainer: MobileFineTuner? = null
        try {
            require(modelDir.isDirectory) { "模型目录不存在" }
            require(File(modelDir, "config.json").isFile) { "缺少 config.json" }
            require(datasetFile.isFile && datasetFile.length() > 0) { "训练 JSONL 文件不存在或为空" }
            require(File(modelDir, "tokenizer.json").isFile) { "缺少 tokenizer.json" }
            require(File(modelDir, "vocab.json").isFile && File(modelDir, "merges.txt").isFile) {
                "当前版本需要 Qwen BPE tokenizer 文件（vocab.json、merges.txt）"
            }
            require(File(modelDir, "model.safetensors").isFile ||
                File(modelDir, "model.safetensors.index.json").isFile) {
                "缺少 model.safetensors 或分片索引文件"
            }

            updateNotification("加载本地模型权重…")
            trainer = MobileFineTuner.open(modelDir.absolutePath, true)
            session = trainer
            trainer.initLora(
                MobileFineTuner.LoraConfig(
                    8, 16f, 0.05f, 42L,
                    arrayOf("q_proj", "k_proj", "v_proj", "o_proj")
                )
            )
            trainer.createTrainer(MobileFineTuner.TrainerConfig.defaults())

            val metadata = File(checkpointPrefix.absolutePath + ".json")
            if (resume && metadata.isFile &&
                File(checkpointPrefix.absolutePath + ".weights").isFile &&
                File(checkpointPrefix.absolutePath + ".adam").isFile) {
                val saved = JSONObject(metadata.readText())
                currentStep = saved.optInt("step", 0)
                currentEpoch = saved.optInt("epoch", 0)
                currentLine = saved.optInt("line", 0)
                trainer.loadCheckpoint(checkpointPrefix.absolutePath)
                updateNotification("已恢复断点：第 ${currentStep} 步")
            }

            val power = getSystemService(Context.POWER_SERVICE) as PowerManager
            val rows = datasetFile.readLines(Charsets.UTF_8)
            require(rows.isNotEmpty()) { "训练数据没有记录" }
            val total = rows.size * epochs
            updateNotification("本地训练开始：共 ${rows.size} 条样本")

            for (epoch in currentEpoch until epochs) {
                val start = if (epoch == currentEpoch) currentLine else 0
                for (lineIndex in start until rows.size) {
                    if (stopRequested.get()) break
                    if (pauseRequested.get() && !stopRequested.get()) {
                        saveCheckpoint(trainer, epoch, lineIndex)
                        updateNotification("已暂停 · 第 ${currentStep} 步")
                        while (pauseRequested.get() && !stopRequested.get()) {
                            SystemClock.sleep(1000)
                        }
                        if (!stopRequested.get()) updateNotification("训练继续运行")
                    }
                    if (stopRequested.get()) break

                    // Avoid prolonged high-temperature training and protect a low battery.
                    if (Build.VERSION.SDK_INT >= 29 &&
                        power.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) {
                        saveCheckpoint(trainer, epoch, lineIndex)
                        updateNotification("温度较高，冷却 30 秒后继续…")
                        SystemClock.sleep(30_000)
                    }
                    val battery = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
                    val level = battery.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                    val batteryIntent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                    val batteryStatus = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
                    val charging = batteryStatus == BatteryManager.BATTERY_STATUS_CHARGING ||
                        batteryStatus == BatteryManager.BATTERY_STATUS_FULL
                    if (level in 0..14 && !charging) {
                        pauseRequested.set(true)
                        saveCheckpoint(trainer, epoch, lineIndex)
                        updateNotification("电量低于 15%，已自动暂停")
                        continue
                    }

                    val example = parseExample(rows[lineIndex]) ?: continue
                    val prefixIds = MobileFineTuner.encodeText(modelDir.absolutePath, example.first, maxSequence)
                    val fullIds = MobileFineTuner.encodeText(modelDir.absolutePath, example.first + example.second, maxSequence)
                    if (fullIds.size < 2 || prefixIds.isEmpty() || prefixIds.size >= fullIds.size) continue
                    if (fullIds.size > maxSequence || prefixIds.size >= maxSequence) continue

                    val ids = IntArray(maxSequence)
                    val mask = FloatArray(maxSequence)
                    val labels = IntArray(maxSequence) { -100 }
                    for (i in fullIds.indices) {
                        ids[i] = fullIds[i]
                        mask[i] = 1f
                        if (i >= prefixIds.size) labels[i] = fullIds[i]
                    }
                    val result = trainer.trainStep(ids, mask, labels, 1, maxSequence)
                    currentStep++
                    lastLoss = result.loss
                    currentEpoch = epoch
                    currentLine = lineIndex + 1
                    if (currentStep % 10 == 0) {
                        updateNotification("训练中 · ${currentStep} 步 · loss ${"%.4f".format(lastLoss)} · ${100 * (epoch * rows.size + lineIndex + 1) / total}%")
                    }
                    if (currentStep % 25 == 0) saveCheckpoint(trainer, epoch, lineIndex + 1)
                }
                currentEpoch = epoch + 1
                currentLine = 0
                saveCheckpoint(trainer, currentEpoch, 0)
                if (stopRequested.get()) break
            }

            if (stopRequested.get()) {
                saveCheckpoint(trainer, currentEpoch, currentLine)
                updateNotification("已停止 · 检查点已保存（${currentStep} 步）")
            } else {
                saveCheckpoint(trainer, epochs, 0)
                updateNotification("训练完成 · ${currentStep} 步 · 最终 loss ${"%.4f".format(lastLoss)}")
            }
        } catch (t: Throwable) {
            updateNotification("训练错误：${t.message ?: t.javaClass.simpleName}")
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString("training_error", t.stackTraceToString()).apply()
        } finally {
            try { trainer?.close() } catch (_: Throwable) {}
            session = null
            releaseWakeLock()
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean("service_running", false).apply()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun parseExample(line: String): Pair<String, String>? {
        return try {
            val obj = JSONObject(line)
            if (obj.has("instruction") && obj.has("output")) {
                val instruction = obj.optString("instruction")
                val input = obj.optString("input")
                val prefix = "<|im_start|>user\n" + instruction +
                    (if (input.isNotBlank()) "\n" + input else "") +
                    "<|im_end|>\n<|im_start|>assistant\n"
                prefix to (obj.optString("output") + "<|im_end|>")
            } else {
                val messages = obj.optJSONArray("messages") ?: return null
                if (messages.length() < 2) return null
                val last = messages.optJSONObject(messages.length() - 1) ?: return null
                if (last.optString("role") != "assistant") return null
                val prefix = StringBuilder()
                for (i in 0 until messages.length() - 1) {
                    val m = messages.optJSONObject(i) ?: continue
                    val role = m.optString("role")
                    if (role !in setOf("system", "user", "assistant")) continue
                    prefix.append("<|im_start|>").append(role).append('\n')
                        .append(m.optString("content")).append("<|im_end|>\n")
                }
                prefix.append("<|im_start|>assistant\n")
                prefix.toString() to (last.optString("content") + "<|im_end|>")
            }
        } catch (_: Throwable) { null }
    }

    private fun saveCheckpoint(t: MobileFineTuner, epoch: Int, line: Int) {
        val parent = checkpointPrefix.parentFile ?: filesDir
        parent.mkdirs()
        val tempPrefix = File(parent, checkpointPrefix.name + ".tmp").absolutePath
        t.saveCheckpoint(tempPrefix)
        File(tempPrefix + ".weights").renameTo(File(checkpointPrefix.absolutePath + ".weights"))
        File(tempPrefix + ".adam").renameTo(File(checkpointPrefix.absolutePath + ".adam"))
        val json = JSONObject()
            .put("step", currentStep)
            .put("epoch", epoch)
            .put("line", line)
            .put("loss", if (lastLoss.isFinite()) lastLoss.toDouble() else JSONObject.NULL)
            .put("model", modelDir.name)
        val tmp = File(parent, checkpointPrefix.name + ".json.tmp")
        tmp.writeText(json.toString())
        tmp.renameTo(File(checkpointPrefix.absolutePath + ".json"))
    }

    private fun acquireWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LocalAI800:Training").apply {
            setReferenceCounted(false)
            acquire(12 * 60 * 60 * 1000L)
        }
    }

    private fun releaseWakeLock() {
        try { if (wakeLock?.isHeld == true) wakeLock?.release() } catch (_: Throwable) {}
        wakeLock = null
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val channel = NotificationChannel(CHANNEL_ID, "Local AI 800 本地训练", NotificationManager.IMPORTANCE_LOW)
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(channel)
        }
    }

    private fun buildNotification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = Notification.Builder(this, CHANNEL_ID)
        return builder.setContentTitle("Local AI 800")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val manager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, buildNotification(text))
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString("training_status", text)
            .putInt("training_step", currentStep).apply()
    }

    override fun onDestroy() {
        stopRequested.set(true)
        pauseRequested.set(false)
        try { session?.close() } catch (_: Throwable) {}
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        const val ACTION_START = "ai.localai800.training.START"
        const val ACTION_PAUSE = "ai.localai800.training.PAUSE"
        const val ACTION_RESUME = "ai.localai800.training.RESUME"
        const val ACTION_STOP = "ai.localai800.training.STOP"
        const val EXTRA_MODEL_DIR = "model_dir"
        const val EXTRA_DATASET = "dataset"
        const val EXTRA_CHECKPOINT = "checkpoint"
        const val EXTRA_MAX_SEQUENCE = "max_sequence"
        const val EXTRA_EPOCHS = "epochs"
        const val EXTRA_RESUME = "resume"
        private const val CHANNEL_ID = "local_ai_800_training"
        private const val NOTIFICATION_ID = 8002
        private const val PREFS = "local_ai_800_training"
    }
}
