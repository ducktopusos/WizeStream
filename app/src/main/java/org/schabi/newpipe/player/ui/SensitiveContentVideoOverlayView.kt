package org.schabi.newpipe.player.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import org.schabi.newpipe.util.image.NormalizedRect

/** Draws blurred regions over the playing video. */
class SensitiveContentVideoOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var regions: List<NormalizedRect> = emptyList()
    private var blurredFrame: Bitmap? = null
    private var fullCover: Boolean = false
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val fallbackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(235, 18, 18, 18)
    }

    fun setRegions(newRegions: List<NormalizedRect>, frame: Bitmap?, coverAll: Boolean = false) {
        if (blurredFrame !== frame) {
            blurredFrame?.recycle()
        }
        blurredFrame = frame
        regions = newRegions
        fullCover = coverAll
        visibility = if (coverAll || newRegions.isNotEmpty()) VISIBLE else GONE
        invalidate()
    }

    fun clearRegions() {
        blurredFrame?.recycle()
        blurredFrame = null
        regions = emptyList()
        fullCover = false
        visibility = GONE
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (fullCover) {
            val frame = blurredFrame
            if (frame != null) {
                canvas.drawBitmap(frame, null, Rect(0, 0, width, height), paint)
            } else {
                canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), fallbackPaint)
            }
            return
        }
        val frame = blurredFrame
        for (region in regions) {
            val dst = RectF(
                region.left * width,
                region.top * height,
                region.right * width,
                region.bottom * height
            )
            if (dst.width() <= 1f || dst.height() <= 1f) continue
            if (frame != null) {
                val src = Rect(
                    (region.left * frame.width).toInt().coerceIn(0, frame.width - 1),
                    (region.top * frame.height).toInt().coerceIn(0, frame.height - 1),
                    (region.right * frame.width).toInt().coerceIn(1, frame.width),
                    (region.bottom * frame.height).toInt().coerceIn(1, frame.height)
                )
                canvas.drawBitmap(frame, src, dst, paint)
            } else {
                canvas.drawRoundRect(dst, 16f, 16f, fallbackPaint)
            }
        }
    }

    override fun onDetachedFromWindow() {
        clearRegions()
        super.onDetachedFromWindow()
    }
}
