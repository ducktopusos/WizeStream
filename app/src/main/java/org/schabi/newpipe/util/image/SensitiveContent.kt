package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import androidx.preference.PreferenceManager

/**
 * Sensitive-content settings and helpers.
 *
 * Thumbnails are blurred inside Coil before they are drawn. AI models are
 * downloaded only when an AI mode is selected.
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

    /** Probability above which the NSFW classifier marks content as explicit. */
    const val NSFW_THRESHOLD = 0.5f

    /** Probability above which the gender classifier marks a face as female. */
    const val FEMALE_THRESHOLD = 0.45f

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

    /** Memory cache keys must keep blurred results for different modes apart. */
    @JvmStatic
    fun memoryCacheKey(baseKey: String?, mode: String): String? = when {
        baseKey == null || mode == OFF -> baseKey
        else -> "wizestream-sensitive:$mode:$baseKey"
    }
}

/** Applies a two-pass box blur, keeping the cost independent of [radius]. */
internal fun blurBitmap(input: Bitmap, radius: Int): Bitmap {
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

    return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
        setPixels(result, 0, width, 0, 0, width, height)
    }
}

/** Radius proportional to the image size and the selected intensity. */
internal fun blurRadiusFor(
    width: Int,
    height: Int,
    intensity: String = SensitiveContent.INTENSITY_MEDIUM
): Int {
    val minSide = minOf(width, height)
    val (divisor, minRadius, maxRadius) = when (intensity) {
        SensitiveContent.INTENSITY_LOW -> Triple(30, 4, 16)
        SensitiveContent.INTENSITY_HIGH -> Triple(12, 10, 48)
        else -> Triple(20, 6, 32)
    }
    return (minSide / divisor).coerceIn(minRadius, maxRadius)
}

private const val BLUR_PASSES = 2

/** One separable box-blur pass with clamped edges. */
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
