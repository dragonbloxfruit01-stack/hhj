package com.jiahao.delta.runner

import android.app.Activity
import android.os.Bundle
import android.content.Intent
import android.provider.Settings
import android.graphics.Typeface
import android.view.Gravity
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import org.json.JSONObject
import java.util.PriorityQueue

data class Zone(val id: String, val name: String, val risk: Int, val loot: String)
data class Edge(val a: String, val b: String, val distance: Double, val danger: Int)
data class Route(val names: List<String>, val distance: Double, val risk: Int)

class MainActivity : Activity() {
    private val zones = listOf(
        Zone("visitor", "访客中心", 2, "补给、医疗"),
        Zone("small_substation", "小变电站", 2, "电子、工具"),
        Zone("river", "河道外围", 2, "野外补给"),
        Zone("cement", "水泥厂", 3, "工具、武器物资"),
        Zone("barracks", "军营", 3, "补给、电子"),
        Zone("administration", "行政区", 4, "电子、保险柜"),
        Zone("main_substation", "主变电站", 4, "电子、工具")
    )
    // Provisional planning graph only; replace with verified Garena map paths after calibration.
    private val edges = listOf(
        Edge("visitor", "small_substation", 1.0, 1),
        Edge("visitor", "river", 1.2, 1),
        Edge("small_substation", "cement", 1.5, 2),
        Edge("small_substation", "main_substation", 1.7, 3),
        Edge("river", "barracks", 1.8, 2),
        Edge("cement", "barracks", 1.4, 2),
        Edge("cement", "administration", 2.0, 3),
        Edge("barracks", "administration", 1.8, 3),
        Edge("main_substation", "administration", 1.2, 3)
    )
    private lateinit var start: Spinner
    private lateinit var destination: Spinner
    private lateinit var output: TextView
    private lateinit var observerStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (resources.displayMetrics.density * 18).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(0xFFF6F8FB.toInt())
        }
        val title = TextView(this).apply {
            text = "DELTA AUTO RUNNER"
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(0xFF172033.toInt())
        }
        val subtitle = TextView(this).apply {
            text = "零号大坝 · 普通行动 · 单人低风险路线规划"
            textSize = 14f
            setPadding(0, 8, 0, 18)
            setTextColor(0xFF526174.toInt())
        }
        root.addView(title)
        root.addView(subtitle)
        observerStatus = TextView(this).apply {
            text = observerSummary()
            textSize = 13f
            setTextColor(0xFF175CD3.toInt())
            setPadding(0, 4, 0, 8)
        }
        root.addView(observerStatus)
        root.addView(Button(this).apply {
            text = "开启屏幕文字识别服务"
            setOnClickListener { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }
        })
        root.addView(label("起点区域"))
        start = spinner()
        root.addView(start)
        root.addView(label("目标区域"))
        destination = spinner()
        root.addView(destination)
        val names = zones.map { it.name }
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, names).also {
            it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        start.adapter = adapter
        destination.adapter = adapter
        start.setSelection(0)
        destination.setSelection(3)
        val calculate = Button(this).apply {
            text = "计算低风险路线"
            setOnClickListener { showRoute() }
        }
        root.addView(calculate)
        output = TextView(this).apply {
            textSize = 15f
            setTextColor(0xFF172033.toInt())
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(0xFFFFFFFF.toInt())
            text = "选择起点和目标区域，计算规划路线。"
        }
        root.addView(output, LinearLayout.LayoutParams(-1, 0, 1f).apply { topMargin = pad })
        val note = TextView(this).apply {
            text = "数据说明：区域风险为规划用估值；路线连接、撤离条件及坐标尚未按当前 Garena 版本实测校准。本应用仅提供离线地图规划，不读取或控制游戏进程。"
            textSize = 12f
            setTextColor(0xFF667085.toInt())
            setPadding(0, pad, 0, 0)
        }
        root.addView(note)
        val scroll = ScrollView(this).apply { addView(root) }
        setContentView(scroll)
    }

    override fun onResume() {
        super.onResume()
        if (::observerStatus.isInitialized) observerStatus.text = observerSummary()
    }

    private fun observerSummary(): String {
        val p = getSharedPreferences("screen_observer", MODE_PRIVATE)
        val state = p.getString("state", "尚未启用")
        val time = p.getLong("updated_at", 0L)
        val freshness = if (time == 0L) "尚无识别记录" else "最近更新：" + ((System.currentTimeMillis() - time) / 1000) + " 秒前"
        return "屏幕观察：" + state + "\n" + freshness
    }

    private fun label(value: String) = TextView(this).apply {
        text = value
        textSize = 14f
        setTypeface(null, Typeface.BOLD)
        setTextColor(0xFF344054.toInt())
        setPadding(0, 10, 0, 4)
    }

    private fun spinner() = Spinner(this)

    private fun showRoute() {
        val source = zones[start.selectedItemPosition]
        val target = zones[destination.selectedItemPosition]
        if (source.id == target.id) {
            output.text = "起点与目标相同，请选择不同区域。"
            return
        }
        val route = findRoute(source.id, target.id)
        output.text = if (route == null) "当前规划图中没有连接路径。" else buildString {
            append("推荐规划路线\n\n")
            append(route.names.joinToString("  →  "))
            append("\n\n规划距离权重：")
            append(String.format("%.1f", route.distance))
            append("\n累计风险分：")
            append(route.risk)
            append("\n\n沿途区域：\n")
            route.names.forEachIndexed { index, id ->
                val z = zones.first { it.id == id }
                append("${index + 1}. ${z.name} · 风险 ${z.risk}/5\n")
                append("   物资类别：${z.loot}\n")
            }
            append("\n注意：这是基于示例连接图的离线规划结果，不是实测游戏导航。出发前请以当前局内地图和撤离条件为准。")
        }
    }

    private fun findRoute(source: String, target: String): Route? {
        data class State(val id: String, val cost: Double)
        val best = zones.associate { it.id to Double.POSITIVE_INFINITY }.toMutableMap()
        val previous = mutableMapOf<String, String>()
        val queue = PriorityQueue<State>(compareBy { it.cost })
        best[source] = 0.0
        queue.add(State(source, 0.0))
        while (queue.isNotEmpty()) {
            val current = queue.poll()
            if (current.cost != best[current.id]) continue
            if (current.id == target) break
            edges.filter { it.a == current.id || it.b == current.id }.forEach { edge ->
                val next = if (edge.a == current.id) edge.b else edge.a
                val zone = zones.first { it.id == next }
                val cost = edge.distance + edge.danger * 3.0 + zone.risk * 2.0
                val candidate = current.cost + cost
                if (candidate < best[next]!!) {
                    best[next] = candidate
                    previous[next] = current.id
                    queue.add(State(next, candidate))
                }
            }
        }
        if (!best[target]!!.isFinite()) return null
        val path = mutableListOf<String>()
        var cursor = target
        path.add(cursor)
        while (cursor != source) {
            cursor = previous[cursor] ?: return null
            path.add(cursor)
        }
        path.reverse()
        val totalRisk = path.sumOf { id -> zones.first { it.id == id }.risk }
        return Route(path, best[target]!!, totalRisk)
    }
}
