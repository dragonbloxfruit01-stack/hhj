package com.jiahao.delta.runner

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class GameScreenObserverService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val root = rootInActiveWindow ?: return
        val lines = mutableListOf<String>()
        collectText(root, lines, 0)
        val snapshot = lines.distinct().take(100).joinToString("\n")
        val prefs = getSharedPreferences("screen_observer", MODE_PRIVATE)
        prefs.edit()
            .putString("snapshot", snapshot)
            .putString("state", classify(snapshot))
            .putLong("updated_at", System.currentTimeMillis())
            .apply()
        root.recycle()
    }

    private fun collectText(node: AccessibilityNodeInfo, out: MutableList<String>, depth: Int) {
        if (depth > 18 || out.size >= 150) return
        node.text?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let(out::add)
        node.contentDescription?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let(out::add)
        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            collectText(child, out, depth + 1)
            child.recycle()
        }
    }

    private fun classify(text: String): String {
        val s = text.lowercase()
        return when {
            listOf("撤离成功", "行动成功", "结算").any(s::contains) -> "疑似结算/撤离结果"
            listOf("撤离点", "撤离倒计时", "前往撤离").any(s::contains) -> "疑似撤离阶段"
            listOf("背包", "物品价值", "出售").any(s::contains) -> "疑似背包/物资界面"
            listOf("复活", "部署", "行动开始", "准备").any(s::contains) -> "疑似开局/部署界面"
            listOf("搜索", "开启容器", "拾取").any(s::contains) -> "疑似搜索/拾取提示"
            else -> "正在观察屏幕文字（游戏画面可能不暴露无障碍文本）"
        }
    }

    override fun onInterrupt() {
        getSharedPreferences("screen_observer", MODE_PRIVATE).edit()
            .putString("state", "屏幕观察已暂停").apply()
    }
}