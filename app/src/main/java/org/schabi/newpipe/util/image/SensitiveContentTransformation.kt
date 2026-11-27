package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import coil3.size.Size
import coil3.transform.Transformation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Blurs thumbnails inside Coil's pipeline, before any pixel reaches the UI. */
class SensitiveContentTransformation(
    context: Context,
    private val mode: String,
    private val classifier: ContentClassifier = OnDeviceContentClassifier
) : Transformation() {
    private val context = context.applicationContext

    override val cacheKey: String = "wizestream-sensitive-content:$mode"

    override suspend fun transform(input: Bitmap, size: Size): Bitmap = withContext(Dispatchers.IO) {
        if (mode == SensitiveContent.OFF) {
            return@withContext input
        }
        if (!SensitiveContent.requiresAiModels(mode) || isInstalled()) {
            if (isSensitive(input)) {
                blur(input)
            } else {
                input
            }
        } else {
            // Models are still downloading; blur conservatively.
            blur(input)
        }
    }

    private fun isInstalled(): Boolean {
        if (SensitiveContentModelManager.isInstalled(context)) return true
        SensitiveContentModelManager.ensureDownloaded(context)
        return false
    }

    private fun blur(input: Bitmap): Bitmap = blurBitmap(
        input,
        blurRadiusFor(input.width, input.height, SensitiveContent.intensity(context))
    )

    private fun isSensitive(bitmap: Bitmap): Boolean = when (mode) {
        SensitiveContent.BLUR_ALL -> true

        SensitiveContent.NSFW_ONLY -> isExplicit(bitmap)

        SensitiveContent.NSFW_AND_WOMEN ->
            isExplicit(bitmap) || classifier.femaleBodyRegions(context, bitmap).isNotEmpty()

        else -> false
    }

    private fun isExplicit(bitmap: Bitmap): Boolean = (classifier.nsfwScore(context, bitmap) ?: 0f) >= SensitiveContent.NSFW_THRESHOLD
}
