package org.schabi.newpipe.player.ui

import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.PixelCopy
import android.view.SurfaceView
import androidx.core.view.isVisible
import java.util.concurrent.atomic.AtomicBoolean
import org.schabi.newpipe.databinding.PlayerBinding
import org.schabi.newpipe.player.Player
import org.schabi.newpipe.util.image.NormalizedRect
import org.schabi.newpipe.util.image.OnDeviceContentClassifier
import org.schabi.newpipe.util.image.SensitiveContent
import org.schabi.newpipe.util.image.blurBitmap
import org.schabi.newpipe.util.image.blurRadiusFor

/**
 * HaramBlur-style protection for the playing video.
 *
 * Samples the video surface with [PixelCopy], runs on-device detection, and
 * blurs only the sensitive regions (female faces) or the full frame for
 * explicit/blur-all modes. A tap reveals the current stream until the next
 * stream/metadata change.
 */
class VideoSensitiveContentController(
    private val binding: PlayerBinding,
    private val player: Player
) {
    private val handler = Handler(Looper.getMainLooper())
    private val sampling = AtomicBoolean(false)
    private var currentStreamUrl: String? = null
    private var revealedForStream = false
    private var destroyed = false

    private val sampleRunnable = object : Runnable {
        override fun run() {
            if (destroyed) return
            maybeSample()
        }
    }

    fun onMetadataChanged(streamUrl: String?) {
        currentStreamUrl = streamUrl
        revealedForStream = false
        hideOverlay()
        scheduleSample(300)
    }

    fun onRenderedFirstFrame() {
        scheduleSample(100)
    }

    fun onPlaying() {
        scheduleSample(250)
    }

    fun onPaused() {
        handler.removeCallbacks(sampleRunnable)
    }

    fun onTransition() {
        revealedForStream = false
        hideOverlay()
    }

    fun destroy() {
        destroyed = true
        handler.removeCallbacksAndMessages(null)
        hideOverlay()
        binding.sensitiveContentVideoOverlay.setOnClickListener(null)
    }

    private fun scheduleSample(delayMs: Long) {
        handler.removeCallbacks(sampleRunnable)
        if (!destroyed) {
            handler.postDelayed(sampleRunnable, delayMs)
        }
    }

    private fun maybeSample() {
        if (destroyed) return
        val context = binding.root.context
        val mode = SensitiveContent.mode(context)
        if (mode == SensitiveContent.OFF) {
            hideOverlay()
            return
        }

        val url = currentStreamUrl ?: player.playQueue?.item?.url
        if (url != currentStreamUrl) {
            currentStreamUrl = url
            revealedForStream = false
            hideOverlay()
        }

        if (revealedForStream) {
            hideOverlay()
            if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
            return
        }

        if (!player.isPlaying) return

        if (mode == SensitiveContent.BLUR_ALL) {
            captureAndApply(mode)
            return
        }

        if (SensitiveContent.requiresAiModels(mode) &&
            !SensitiveContentModelManager.isInstalled(context)
        ) {
            SensitiveContentModelManager.ensureDownloaded(context)
            showFullOverlay(null)
            if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
            return
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            // PixelCopy unavailable on API 23; be conservative.
            showFullOverlay(null)
            return
        }

        captureAndApply(mode)
    }

    private fun captureAndApply(mode: String) {
        val surface = activeSurfaceView() ?: run {
            scheduleSample(SAMPLE_INTERVAL_MS)
            return
        }
        if (!sampling.compareAndSet(false, true)) return

        capture(surface) { bitmap ->
            if (bitmap == null || destroyed) {
                sampling.set(false)
                if (!destroyed && player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
                return@capture
            }
            Thread {
                val result = try {
                    detect(bitmap, mode)
                } catch (_: Throwable) {
                    DetectionResult(emptyList(), false)
                }
                handler.post {
                    sampling.set(false)
                    if (destroyed) {
                        bitmap.recycle()
                        return@post
                    }
                    applyResult(bitmap, result)
                }
            }.start()
        }
    }

    private data class DetectionResult(
        val regions: List<NormalizedRect>,
        val fullFrame: Boolean
    )

    private fun detect(bitmap: Bitmap, mode: String): DetectionResult {
        val context = binding.root.context
        val explicit = (OnDeviceContentClassifier.nsfwScore(context, bitmap) ?: 0f) >=
            SensitiveContent.NSFW_THRESHOLD
        return when (mode) {
            SensitiveContent.BLUR_ALL -> DetectionResult(emptyList(), true)

            SensitiveContent.NSFW_ONLY -> DetectionResult(emptyList(), explicit)

            SensitiveContent.NSFW_AND_WOMEN -> {
                if (explicit) {
                    DetectionResult(emptyList(), true)
                } else {
                    val regions = OnDeviceContentClassifier.femaleFaceRegions(context, bitmap)
                        .map { expandRegion(it) }
                    DetectionResult(regions, false)
                }
            }

            else -> DetectionResult(emptyList(), false)
        }
    }

    private fun applyResult(source: Bitmap, result: DetectionResult) {
        if (revealedForStream || destroyed) {
            source.recycle()
            return
        }
        if (result.fullFrame) {
            val blurred = blurBitmap(source, blurRadiusFor(source.width, source.height, SensitiveContent.intensity(binding.root.context)))
            if (blurred != source) source.recycle()
            showFullOverlay(blurred)
        } else if (result.regions.isNotEmpty()) {
            val blurred = blurBitmap(source, blurRadiusFor(source.width, source.height, SensitiveContent.intensity(binding.root.context)))
            if (blurred != source) source.recycle()
            showRegions(result.regions, blurred)
        } else {
            source.recycle()
            hideOverlay()
        }
        if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
    }

    private fun showFullOverlay(blurred: Bitmap?) {
        val overlay = binding.sensitiveContentVideoOverlay
        val mapped = listOf(NormalizedRect(0f, 0f, 1f, 1f))
        overlay.setRegions(mapToOverlay(mapped), blurred, coverAll = true)
        overlay.isVisible = true
        overlay.isClickable = true
        overlay.setOnClickListener {
            revealedForStream = true
            hideOverlay()
            if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
        }
    }

    private fun showRegions(regions: List<NormalizedRect>, blurredFrame: Bitmap) {
        val overlay = binding.sensitiveContentVideoOverlay
        val mapped = mapToOverlay(regions)
        if (mapped.isEmpty()) {
            blurredFrame.recycle()
            hideOverlay()
            return
        }
        overlay.setRegions(mapped, blurredFrame, coverAll = false)
        overlay.isVisible = true
        overlay.isClickable = true
        overlay.setOnClickListener {
            revealedForStream = true
            hideOverlay()
            if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_MS)
        }
    }

    private fun hideOverlay() {
        val overlay = binding.sensitiveContentVideoOverlay
        overlay.setOnClickListener(null)
        overlay.isClickable = false
        overlay.clearRegions()
    }

    private fun mapToOverlay(regions: List<NormalizedRect>): List<NormalizedRect> {
        val overlay = binding.sensitiveContentVideoOverlay
        val surface = binding.surfaceView
        if (overlay.width <= 0 || overlay.height <= 0 || surface.width <= 0 || surface.height <= 0) {
            return regions
        }
        val overlayLoc = IntArray(2)
        val surfaceLoc = IntArray(2)
        overlay.getLocationOnScreen(overlayLoc)
        surface.getLocationOnScreen(surfaceLoc)
        val offsetLeft = (surfaceLoc[0] - overlayLoc[0]).toFloat() / overlay.width
        val offsetTop = (surfaceLoc[1] - overlayLoc[1]).toFloat() / overlay.height
        val scaleX = surface.width.toFloat() / overlay.width
        val scaleY = surface.height.toFloat() / overlay.height
        return regions.mapNotNull { r ->
            val left = (offsetLeft + r.left * scaleX).coerceIn(0f, 1f)
            val top = (offsetTop + r.top * scaleY).coerceIn(0f, 1f)
            val right = (offsetLeft + r.right * scaleX).coerceIn(0f, 1f)
            val bottom = (offsetTop + r.bottom * scaleY).coerceIn(0f, 1f)
            if (right - left < 0.01f || bottom - top < 0.01f) {
                null
            } else {
                NormalizedRect(left, top, right, bottom)
            }
        }
    }

    private fun expandRegion(region: NormalizedRect): NormalizedRect {
        // Expand face box to cover more of the head/hair area for privacy.
        val w = region.right - region.left
        val h = region.bottom - region.top
        val cx = (region.left + region.right) / 2f
        val cy = (region.top + region.bottom) / 2f
        val nw = w * 1.8f
        val nh = h * 2.2f
        return NormalizedRect(
            left = (cx - nw / 2f).coerceIn(0f, 1f),
            top = (cy - nh / 2f).coerceIn(0f, 1f),
            right = (cx + nw / 2f).coerceIn(0f, 1f),
            bottom = (cy + nh / 2f).coerceIn(0f, 1f)
        )
    }

    private fun activeSurfaceView(): SurfaceView? {
        val surface = binding.surfaceView
        return if (surface.isAttachedToWindow && surface.width > 0 && surface.height > 0) {
            surface
        } else {
            null
        }
    }

    private fun capture(surface: SurfaceView, callback: (Bitmap?) -> Unit) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            callback(null)
            return
        }
        val width = (surface.width / SAMPLE_SCALE).coerceAtLeast(64)
        val height = (surface.height / SAMPLE_SCALE).coerceAtLeast(64)
        val bitmap = try {
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        } catch (_: Throwable) {
            callback(null)
            return
        }
        try {
            PixelCopy.request(surface, bitmap, { result ->
                if (result == PixelCopy.SUCCESS) {
                    callback(bitmap)
                } else {
                    bitmap.recycle()
                    callback(null)
                }
            }, handler)
        } catch (_: Throwable) {
            bitmap.recycle()
            callback(null)
        }
    }

    companion object {
        private const val SAMPLE_INTERVAL_MS = 900L
        private const val SAMPLE_SCALE = 4
    }
}
