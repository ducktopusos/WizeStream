package org.schabi.newpipe.player.ui

import android.content.Context
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

/** Samples and blurs sensitive video regions without changing playback state. */
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
    private var lastMode: String? = null
    private var stableSamples = 0
    private var awaitingClassification = false
    private var cleanStreak = 0

    @Volatile
    private var femaleStreak = 0

    @Volatile
    private var lastExplicit = false

    @Volatile
    private var lastExplicitCheckMs = 0L

    private val sampleRunnable = object : Runnable {
        override fun run() {
            if (destroyed) return
            maybeSample()
        }
    }

    fun onMetadataChanged(streamUrl: String?) {
        currentStreamUrl = streamUrl
        resetDetection()
        armCoverageIfNeeded()
        scheduleSample(300)
    }

    fun onRenderedFirstFrame() {
        armCoverageIfNeeded()
        scheduleSample(100)
    }

    fun onPlaying() {
        armCoverageIfNeeded()
        scheduleSample(250)
    }

    fun onPaused() {
        handler.removeCallbacks(sampleRunnable)
    }

    fun onTransition() {
        resetDetection()
        hideOverlay()
    }

    fun destroy() {
        destroyed = true
        handler.removeCallbacksAndMessages(null)
        hideOverlay()
        binding.sensitiveContentVideoOverlay.setOnClickListener(null)
    }

    private fun resetDetection() {
        lastSignature = null
        lastResult = null
        stableSamples = 0
        cleanStreak = 0
        awaitingClassification = false
        femaleStreak = 0
        lastExplicit = false
        lastExplicitCheckMs = 0L
    }

    private fun scheduleSample(delayMs: Long) {
        handler.removeCallbacks(sampleRunnable)
        if (!destroyed) handler.postDelayed(sampleRunnable, delayMs)
    }

    private fun nextInterval(): Long = when {
        awaitingClassification -> 250L
        lastResult?.hasSensitive == true -> SAMPLE_INTERVAL_ACTIVE_MS
        stableSamples >= 2 -> SAMPLE_INTERVAL_STABLE_MS
        else -> SAMPLE_INTERVAL_DEFAULT_MS
    }

    private fun maybeSample() {
        if (destroyed) return
        val context = binding.root.context
        val mode = SensitiveContent.mode(context)
        if (mode != lastMode) {
            lastMode = mode
            resetDetection()
        }
        if (mode == SensitiveContent.OFF) {
            hideOverlay()
            resetDetection()
            return
        }

        armCoverageIfNeeded()

        val url = currentStreamUrl ?: player.playQueue?.item?.url
        if (url != currentStreamUrl) {
            currentStreamUrl = url
            resetDetection()
            awaitingClassification = true
            showFullOverlay(null)
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
            // PixelCopy is unavailable on API 23; stay conservative.
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
                    bitmap.recycle()
                    if (destroyed) return@post
                    awaitingClassification = false
                    if (!result.hasSensitive) hideOverlay()
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

            SensitiveContent.NSFW_ONLY -> DetectionResult(emptyList(), isExplicit(context, bitmap))

            SensitiveContent.NSFW_AND_WOMEN -> {
                val explicit = isExplicit(context, bitmap)
                if (explicit) {
                    DetectionResult(emptyList(), true)
                } else {
                    val regions = OnDeviceContentClassifier.femaleFaceRegions(context, bitmap)
                    femaleStreak = if (regions.isEmpty()) {
                        0
                    } else {
                        (femaleStreak + 1).coerceAtMost(2)
                    }
                    // Require two consistent samples before blurring a face region.
                    DetectionResult(if (femaleStreak >= 2) regions else emptyList(), false)
                }
            }

            else -> DetectionResult(emptyList(), false)
        }
    }

    /** NSFW inference is expensive, so reuse the score for a short window. */
    private fun isExplicit(context: Context, bitmap: Bitmap): Boolean {
        val now = System.currentTimeMillis()
        if (lastExplicitCheckMs != 0L && now - lastExplicitCheckMs < NSFW_RECHECK_INTERVAL_MS) {
            return lastExplicit
        }
        lastExplicit = (OnDeviceContentClassifier.nsfwScore(context, bitmap) ?: 0f) >=
            SensitiveContent.NSFW_THRESHOLD
        lastExplicitCheckMs = now
        return lastExplicit
    }

    private fun applyResult(source: Bitmap, result: DetectionResult) {
        if (destroyed) {
            source.recycle()
            return
        }
        awaitingClassification = false

        if (!result.hasSensitive) {
            source.recycle()
            cleanStreak++
            if (cleanStreak >= 2 || lastResult == null) hideOverlay()
            if (player.isPlaying) scheduleSample(nextInterval())
            return
        }

        val frame = blurBitmap(
            source,
            blurRadiusFor(
                source.width,
                source.height,
                SensitiveContent.intensity(binding.root.context)
            )
        )
        if (frame !== source) source.recycle()
        cleanStreak = 0

        if (result.fullFrame) {
            showFullOverlay(frame)
        } else {
            showRegions(result.regions, frame)
        }
        if (player.isPlaying) scheduleSample(nextInterval())
    }

    private fun showFullOverlay(frame: Bitmap?) = setOverlay(emptyList(), frame, coverAll = true)

    private fun showRegions(regions: List<NormalizedRect>, frame: Bitmap) {
        val mapped = mapToOverlay(regions)
        if (mapped.isEmpty()) {
            hideOverlay()
            frame.recycle()
            return
        }
        setOverlay(mapped, frame, coverAll = false)
    }

    private fun setOverlay(regions: List<NormalizedRect>, frame: Bitmap?, coverAll: Boolean) {
        val overlay = binding.sensitiveContentVideoOverlay
        overlay.setRegions(regions, frame, coverAll)
        overlay.isVisible = true
        overlay.isClickable = false
    }

    private fun armCoverageIfNeeded() {
        val mode = SensitiveContent.mode(binding.root.context)
        if (mode == SensitiveContent.OFF) {
            awaitingClassification = false
            return
        }
        if (lastResult == null && !binding.sensitiveContentVideoOverlay.isVisible) {
            awaitingClassification = true
            showFullOverlay(null)
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
        return regions.mapNotNull { region ->
            val left = (offsetLeft + region.left * scaleX).coerceIn(0f, 1f)
            val top = (offsetTop + region.top * scaleY).coerceIn(0f, 1f)
            val right = (offsetLeft + region.right * scaleX).coerceIn(0f, 1f)
            val bottom = (offsetTop + region.bottom * scaleY).coerceIn(0f, 1f)
            if (right - left < 0.01f || bottom - top < 0.01f) {
                null
            } else {
                NormalizedRect(left, top, right, bottom)
            }
        }
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
        val size = SIGNATURE_SIZE
        val pixels = IntArray(size * size)
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)
        if (scaled !== bitmap) scaled.recycle()

        val signature = LongArray(size * size)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF
            signature[i] = (red * 30 + green * 59 + blue * 11).toLong()
        }
        return signature
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
        private const val SAMPLE_INTERVAL_DEFAULT_MS = 1600L
        private const val SAMPLE_INTERVAL_STABLE_MS = 3000L
        private const val SAMPLE_SCALE = 8
        private const val MAX_CAPTURE_SIZE = 128
        private const val SIGNATURE_SIZE = 12
        private const val SIGNATURE_DIFF_THRESHOLD = 3000L
        private const val NSFW_RECHECK_INTERVAL_MS = 3500L
    }
}
