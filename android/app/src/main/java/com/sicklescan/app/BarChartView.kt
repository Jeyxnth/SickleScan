package com.sicklescan.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.android.material.color.MaterialColors
import com.sicklescan.app.data.DailyCount
import kotlin.math.max

/**
 * A minimal bar chart drawn directly with Canvas -- positive screenings per day.
 * Deliberately not a charting library: this is exactly the "basic
 * Canvas/Compose chart" the brief allowed, so nothing heavier was added.
 */
class BarChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var data: List<DailyCount> = emptyList()

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#B71C1C") }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = MaterialColors.getColor(context, com.google.android.material.R.attr.colorOnSurfaceVariant, Color.GRAY)
        textAlign = Paint.Align.CENTER
        textSize = 28f
    }
    private val countPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = MaterialColors.getColor(context, com.google.android.material.R.attr.colorOnSurface, Color.BLACK)
        textAlign = Paint.Align.CENTER
        textSize = 30f
    }

    fun setData(newData: List<DailyCount>) {
        data = newData
        requestLayout()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (data.isEmpty()) return

        val labelHeight = labelPaint.textSize + 12f
        val countHeight = countPaint.textSize + 8f
        val chartTop = countHeight
        val chartBottom = height - labelHeight
        val chartHeight = (chartBottom - chartTop).coerceAtLeast(1f)

        val maxCount = max(1, data.maxOf { it.count })
        val slotWidth = width.toFloat() / data.size
        val barWidth = slotWidth * 0.5f

        data.forEachIndexed { index, day ->
            val centerX = slotWidth * index + slotWidth / 2f
            val barHeight = chartHeight * day.count / maxCount
            val top = chartBottom - barHeight
            canvas.drawRect(centerX - barWidth / 2f, top, centerX + barWidth / 2f, chartBottom, barPaint)

            if (day.count > 0) {
                canvas.drawText(day.count.toString(), centerX, top - 8f, countPaint)
            }
            canvas.drawText(day.label, centerX, height.toFloat() - 8f, labelPaint)
        }
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = resources.displayMetrics.density.times(160).toInt()
        setMeasuredDimension(width, height)
    }
}
