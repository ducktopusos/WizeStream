package org.schabi.newpipe.util.image

import android.content.Context
import androidx.preference.PreferenceManager
import java.util.Collections

/**
 * HaramBlur-style protection of thumbnails.
 *
 * The selected mode is stored in the default shared preferences under
 * [PREFERENCE_KEY]. Thumbnails loaded through [CoilHelper.loadThumbnail],
 * [CoilHelper.loadDetailsThumbnail] and [CoilHelper.loadPlaylistThumbnail]
 * are classified on-device by [OnDeviceContentClassifier] and blurred before
 * they are ever drawn, so unblurred pixels cannot flash on screen.
 *
 * A blurred thumbnail can be revealed for the current session by tapping it;
 * revealed URLs are remembered in [revealedUrls] and bypass the classifier.
 */
object SensitiveContent {
    const val PREFERENCE_KEY = "sensitive_content_mode"
    const val INTENSITY_KEY = "sensitive_content_intensity"

    const val OFF = "off"
    const val BLUR_ALL = "blur_all"
    const val NSFW_ONLY = "nsfw_only"
    const val NSFW_AND_WOMEN = "nsfw_and_women"

    const val INTENSITY_LOW = "low"
    const val INTENSITY_MEDIUM = "medium"
    const val INTENSITY_HIGH = "high"

    /** Probability above which the NSFW classifier marks a thumbnail as explicit. */
    const val NSFW_THRESHOLD = 0.5f

    /** Probability above which the gender classifier marks a face as female. */
    const val FEMALE_THRESHOLD = 0.45f

    private val revealedUrls: MutableSet<String> = Collections.synchronizedSet(mutableSetOf())
    private val blurredUrls: MutableMap<String, String> = Collections.synchronizedMap(mutableMapOf())

    @JvmStatic
    fun mode(context: Context): String = PreferenceManager
        .getDefaultSharedPreferences(context)
        .getString(PREFERENCE_KEY, OFF)
        ?: OFF

    @JvmStatic
    fun intensity(context: Context): String = PreferenceManager
        .getDefaultSharedPreferences(context)
        .getString(INTENSITY_KEY, INTENSITY_MEDIUM)
        ?: INTENSITY_MEDIUM

    @JvmStatic
    fun isEnabled(context: Context): Boolean = mode(context) != OFF

    @JvmStatic
    fun requiresAiModels(mode: String): Boolean = mode == NSFW_ONLY || mode == NSFW_AND_WOMEN

    @JvmStatic
    fun isRevealed(url: String?): Boolean = url != null && revealedUrls.contains(url)

    @JvmStatic
    fun reveal(url: String?) {
        if (url != null) {
            revealedUrls.add(url)
        }
    }

    /** Remembers that [url] was blurred in [mode], so a tap can offer to reveal it. */
    @JvmStatic
    fun markBlurred(url: String?, mode: String) {
        if (url != null) {
            blurredUrls[url] = mode
        }
    }

    /** True when the currently displayed thumbnail is blurred and can be revealed. */
    @JvmStatic
    fun shouldOfferReveal(url: String?, mode: String): Boolean = url != null && !isRevealed(url) && blurredUrls[url] == mode

    /** Memory cache keys must keep blurred, revealed and per-mode bitmaps apart. */
    @JvmStatic
    fun memoryCacheKey(baseKey: String?, mode: String, revealed: Boolean): String? = when {
        baseKey == null || mode == OFF -> baseKey
        revealed -> "wizestream-revealed:$mode:$baseKey"
        else -> "wizestream-sensitive:$mode:$baseKey"
    }
}

/**
 * Box blur applied to a whole thumbnail. Two separable passes approximate a
 * Gaussian blur while staying linear in the number of pixels, which keeps the
 * cost independent of the radius.
 */
internal fun blurBitmap(input: android.graphics.Bitmap, radius: Int): android.graphics.Bitmap {
    val width = input.width
    val height = input.height
    if (radius <= 0 || width <= 1 || height <= 1) {
        return input
    }

    val pixels = IntArray(width * height)
    input.getPixels(pixels, 0, width, 0, 0, width, height)

    val temp = IntArray(width * height)
    val result = IntArray(width * height)
    var current = pixels
    repeat(BLUR_PASSES) {
        boxBlurPass(current, temp, width, height, radius, horizontal = true)
        boxBlurPass(temp, result, width, height, radius, horizontal = false)
        current = result
    }

    return android.graphics.Bitmap.createBitmap(width, height, android.graphics.Bitmap.Config.ARGB_8888).apply {
        setPixels(result, 0, width, 0, 0, width, height)
    }
}

/** Radius proportional to the thumbnail size and the user-selected intensity. */
internal fun blurRadiusFor(width: Int, height: Int, intensity: String = SensitiveContent.INTENSITY_MEDIUM): Int {
    val minSide = minOf(width, height)
    val (divisor, minRadius, maxRadius) = when (intensity) {
        SensitiveContent.INTENSITY_LOW -> Triple(30, 4, 16)
        SensitiveContent.INTENSITY_HIGH -> Triple(12, 10, 48)
        else -> Triple(20, 6, 32)
    }
    return (minSide / divisor).coerceIn(minRadius, maxRadius)
}

private const val BLUR_PASSES = 2

/**
 * One separable box-blur pass with clamped edges, using a sliding window sum
 * so that the cost does not depend on [radius].
 */
private fun boxBlurPass(
    input: IntArray,
    output: IntArray,
    width: Int,
    height: Int,
    radius: Int,
    horizontal: Boolean
) {
    val lineLength = if (horizontal) width else height
    val lineCount = if (horizontal) height else width
    if (lineLength <= 1) {
        input.copyInto(output)
        return
    }

    val window = radius * 2 + 1
    val maxPosition = lineLength - 1

    fun indexOf(line: Int, position: Int): Int = if (horizontal) {
        line * width + position
    } else {
        position * width + line
    }

    for (line in 0 until lineCount) {
        var sumAlpha = 0
        var sumRed = 0
        var sumGreen = 0
        var sumBlue = 0

        for (offset in -radius..radius) {
            val pixel = input[indexOf(line, offset.coerceIn(0, maxPosition))]
            sumAlpha += pixel ushr 24 and 0xFF
            sumRed += pixel shr 16 and 0xFF
            sumGreen += pixel shr 8 and 0xFF
            sumBlue += pixel and 0xFF
        }

        for (position in 0 until lineLength) {
            output[indexOf(line, position)] =
                ((sumAlpha / window) shl 24) or
                ((sumRed / window) shl 16) or
                ((sumGreen / window) shl 8) or
                (sumBlue / window)

            val added = input[indexOf(line, (position + radius + 1).coerceAtMost(maxPosition))]
            val removed = input[indexOf(line, (position - radius).coerceAtLeast(0))]
            sumAlpha += (added ushr 24 and 0xFF) - (removed ushr 24 and 0xFF)
            sumRed += (added shr 16 and 0xFF) - (removed shr 16 and 0xFF)
            sumGreen += (added shr 8 and 0xFF) - (removed shr 8 and 0xFF)
            sumBlue += (added and 0xFF) - (removed and 0xFF)
        }
    }
}
