package org.schabi.newpipe.player.ui

import android.graphics.Bitmap
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.PixelCopy
import android.view.SurfaceView
import androidx.core.view.isVisible
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import org.schabi.newpipe.databinding.PlayerBinding
import org.schabi.newpipe.player.Player
import org.schabi.newpipe.util.image.NormalizedRect
import org.schabi.newpipe.util.image.OnDeviceContentClassifier
import org.schabi.newpipe.util.image.SensitiveContent
import org.schabi.newpipe.util.image.SensitiveContentModelManager
import org.schabi.newpipe.util.image.blurBitmap
import org.schabi.newpipe.util.image.blurRadiusFor

/**
 * HaramBlur-style protection for the playing video.
 *
 * Samples the video surface with [PixelCopy], runs on-device detection, and
 * blurs only the sensitive regions (female faces) or the full frame for
 * explicit/blur-all modes. Detection is throttled and skipped when the scene
 * is visually unchanged to keep playback smooth.
 */
class VideoSensitiveContentController(
    private val binding: PlayerBinding,
    private val player: Player
) {
    private val handler = Handler(Looper.getMainLooper())
    private val sampling = AtomicBoolean(false)
    private var currentStreamUrl: String? = null
    private var destroyed = false
    private var lastSignature: LongArray? = null
    private var lastResult: DetectionResult? = null
    private var stableSamples = 0

    private val sampleRunnable = object : Runnable {
        override fun run() {
            if (destroyed) return
            maybeSample()
        }
    }

    fun onMetadataChanged(streamUrl: String?) {
        currentStreamUrl = streamUrl
        lastSignature = null
        lastResult = null
        stableSamples = 0
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
        lastSignature = null
        lastResult = null
        stableSamples = 0
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

    private fun nextInterval(): Long = when {
        lastResult?.hasSensitive == true -> SAMPLE_INTERVAL_ACTIVE_MS
        stableSamples >= 2 -> SAMPLE_INTERVAL_STABLE_MS
        else -> SAMPLE_INTERVAL_DEFAULT_MS
    }

    private fun maybeSample() {
        if (destroyed) return
        val context = binding.root.context
        val mode = SensitiveContent.mode(context)
        if (mode == SensitiveContent.OFF) {
            hideOverlay()
            lastSignature = null
            lastResult = null
            stableSamples = 0
            return
        }

        val url = currentStreamUrl ?: player.playQueue?.item?.url
        if (url != currentStreamUrl) {
            currentStreamUrl = url
            lastSignature = null
            lastResult = null
            stableSamples = 0
            hideOverlay()
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
            if (player.isPlaying) scheduleSample(SAMPLE_INTERVAL_ACTIVE_MS)
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
            scheduleSample(nextInterval())
            return
        }
        if (!sampling.compareAndSet(false, true)) return

        capture(surface) { bitmap ->
            if (bitmap == null || destroyed) {
                sampling.set(false)
                if (!destroyed && player.isPlaying) scheduleSample(nextInterval())
                return@capture
            }

            val signature = frameSignature(bitmap)
            val unchanged = isSimilar(signature, lastSignature)
            if (unchanged && lastResult != null) {
                stableSamples = (stableSamples + 1).coerceAtMost(4)
                val result = lastResult!!
                handler.post {
                    sampling.set(false)
                    if (destroyed) {
                        bitmap.recycle()
                        return@post
                    }
                    bitmap.recycle()
                    if (!result.hasSensitive) {
                        hideOverlay()
                    }
                    if (player.isPlaying) scheduleSample(nextInterval())
                }
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
                    stableSamples = 0
                    lastSignature = signature
                    lastResult = result
                    applyResult(bitmap, result)
                }
            }.start()
        }
    }

    private data class DetectionResult(
        val regions: List<NormalizedRect>,
        val fullFrame: Boolean
    ) {
        val hasSensitive: Boolean get() = fullFrame || regions.isNotEmpty()
    }

    private fun detect(bitmap: Bitmap, mode: String): DetectionResult {
        val context = binding.root.context
        return when (mode) {
            SensitiveContent.BLUR_ALL -> DetectionResult(emptyList(), true)

            SensitiveContent.NSFW_ONLY -> {
                val explicit = (OnDeviceContentClassifier.nsfwScore(context, bitmap) ?: 0f) >=
                    SensitiveContent.NSFW_THRESHOLD
                DetectionResult(emptyList(), explicit)
            }

            SensitiveContent.NSFW_AND_WOMEN -> {
                val explicit = (OnDeviceContentClassifier.nsfwScore(context, bitmap) ?: 0f) >=
                    SensitiveContent.NSFW_THRESHOLD
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
        if (destroyed) {
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
        if (player.isPlaying) scheduleSample(nextInterval())
    }

    private fun showFullOverlay(blurred: Bitmap?) {
        val overlay = binding.sensitiveContentVideoOverlay
        val mapped = listOf(NormalizedRect(0f, 0f, 1f, 1f))
        overlay.setRegions(mapToOverlay(mapped), blurred, coverAll = true)
        overlay.isVisible = true
        overlay.isClickable = false
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
        overlay.isClickable = false
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
        // Expand detected female face to approximate the full body.
        val w = region.right - region.left
        val h = region.bottom - region.top
        val cx = (region.left + region.right) / 2f
        val top = (region.top - h * 0.6f).coerceIn(0f, 1f)
        val bottom = (region.bottom + h * 6.5f).coerceIn(0f, 1f)
        val halfWidth = (w * 2.2f).coerceAtLeast(h * 1.4f)
        return NormalizedRect(
            left = (cx - halfWidth).coerceIn(0f, 1f),
            top = top,
            right = (cx + halfWidth).coerceIn(0f, 1f),
            bottom = bottom
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
        val width = (surface.width / SAMPLE_SCALE).coerceIn(64, MAX_CAPTURE_SIZE)
        val height = (surface.height / SAMPLE_SCALE).coerceIn(64, MAX_CAPTURE_SIZE)
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

    private fun frameSignature(bitmap: Bitmap): LongArray {
        val w = SIGNATURE_SIZE
        val h = SIGNATURE_SIZE
        val pixels = IntArray(w * h)
        val scaled = Bitmap.createScaledBitmap(bitmap, w, h, true)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        if (scaled !== bitmap) scaled.recycle()
        val sig = LongArray(w * h)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            sig[i] = (r * 30 + g * 59 + b * 11).toLong()
        }
        return sig
    }

    private fun isSimilar(a: LongArray, b: LongArray?): Boolean {
        if (b == null || a.size != b.size) return false
        var diff = 0L
        for (i in a.indices) {
            diff += abs(a[i] - b[i])
            if (diff > SIGNATURE_DIFF_THRESHOLD) return false
        }
        return true
    }

    companion object {
        private const val SAMPLE_INTERVAL_ACTIVE_MS = 1200L
        private const val SAMPLE_INTERVAL_DEFAULT_MS = 1800L
        private const val SAMPLE_INTERVAL_STABLE_MS = 2600L
        private const val SAMPLE_SCALE = 6
        private const val MAX_CAPTURE_SIZE = 192
        private const val SIGNATURE_SIZE = 12
        private const val SIGNATURE_DIFF_THRESHOLD = 1400L
    }
}
