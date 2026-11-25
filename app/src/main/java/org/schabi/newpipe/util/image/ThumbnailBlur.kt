package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import androidx.preference.PreferenceManager
import coil3.size.Size
import coil3.transform.Transformation

/**
 * Opt-in blur applied to thumbnails.
 *
 * The selected strength is stored in the default shared preferences under
 * [PREFERENCE_KEY]. Every thumbnail loaded through [CoilHelper.loadThumbnail],
 * [CoilHelper.loadDetailsThumbnail] and [CoilHelper.loadPlaylistThumbnail] is
 * blurred with [ThumbnailBlurTransformation] when a strength other than [OFF]
 * is selected. Avatars, channel banners and notification thumbnails are never
 * blurred.
 */
object ThumbnailBlur {
    const val PREFERENCE_KEY = "thumbnail_blur_strength"

    const val OFF = "off"
    const val LIGHT = "light"
    const val MEDIUM = "medium"
    const val STRONG = "strong"

    private const val CACHE_KEY_PREFIX = "wizestream-thumbnail-blur:"

    /** The strength currently selected by the user, [OFF] when unset. */
    @JvmStatic
    fun strength(context: Context): String = PreferenceManager
        .getDefaultSharedPreferences(context)
        .getString(PREFERENCE_KEY, OFF)
        ?: OFF

    @JvmStatic
    fun isEnabled(context: Context): Boolean = radius(context) > 0

    @JvmStatic
    fun radius(context: Context): Int = radius(strength(context))

    /** Blur radius in bitmap pixels; 0 disables blurring. */
    @JvmStatic
    fun radius(strength: String): Int = when (strength) {
        LIGHT -> 8
        MEDIUM -> 16
        STRONG -> 28
        else -> 0
    }

    /**
     * A blurred bitmap must not be served from the memory cache for an
     * unblurred request (and vice versa), so the strength is part of the key.
     */
    @JvmStatic
    fun memoryCacheKey(baseKey: String?, radius: Int): String? = when {
        baseKey == null -> null
        radius <= 0 -> baseKey
        else -> "$CACHE_KEY_PREFIX$radius:$baseKey"
    }
}

/**
 * Coil transformation applying a two-pass box blur, which approximates a
 * Gaussian blur closely enough for thumbnails while staying O(pixels).
 */
internal class ThumbnailBlurTransformation(
    private val radius: Int
) : Transformation() {
    override val cacheKey: String = "wizestream-thumbnail-blur-transform:$radius"

    override suspend fun transform(input: Bitmap, size: Size): Bitmap {
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
        repeat(PASSES) {
            boxBlur(current, temp, width, height, radius, horizontal = true)
            boxBlur(temp, result, width, height, radius, horizontal = false)
            current = result
        }

        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            setPixels(result, 0, width, 0, 0, width, height)
        }
    }

    companion object {
        private const val PASSES = 2
    }
}

/**
 * One separable box-blur pass with clamped edges, using a sliding window sum
 * so that the cost does not depend on [radius].
 */
private fun boxBlur(
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
