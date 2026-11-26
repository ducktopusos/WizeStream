package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import coil3.size.Size
import coil3.transform.Transformation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Classifies and blurs a thumbnail inside Coil's background pipeline, before
 * the bitmap reaches the UI, so explicit pixels never render even for a frame.
 *
 * The classification result is memoized per URL by [SensitiveContent.markBlurred]
 * so that a later cache hit can still offer the tap-to-reveal affordance.
 */
class SensitiveContentTransformation(
    context: Context,
    private val mode: String,
    private val url: String? = null,
    private val classifier: ContentClassifier = OnDeviceContentClassifier
) : Transformation() {
    private val context = context.applicationContext

    override val cacheKey: String = "wizestream-sensitive-content:$mode"

    override suspend fun transform(input: Bitmap, size: Size): Bitmap {
        if (mode == SensitiveContent.OFF) {
            return input
        }

        return withContext(Dispatchers.IO) {
            if (!isSensitive(input)) {
                return@withContext input
            }
            SensitiveContent.markBlurred(url, mode)
            blurBitmap(input, blurRadiusFor(input.width, input.height))
        }
    }

    private fun isSensitive(bitmap: Bitmap): Boolean = when (mode) {
        SensitiveContent.BLUR_ALL -> true
        SensitiveContent.NSFW_ONLY -> isExplicit(bitmap)
        SensitiveContent.NSFW_AND_WOMEN -> isExplicit(bitmap) || classifier.hasFemaleFace(context, bitmap)
        else -> false
    }

    private fun isExplicit(bitmap: Bitmap): Boolean =
        (classifier.nsfwScore(context, bitmap) ?: 0f) >= SensitiveContent.NSFW_THRESHOLD
}
