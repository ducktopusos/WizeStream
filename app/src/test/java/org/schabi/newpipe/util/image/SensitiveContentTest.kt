package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import androidx.preference.PreferenceManager
import coil3.size.Size
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class SensitiveContentTest {
    private val context get() = RuntimeEnvironment.getApplication()

    private fun setMode(mode: String) {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putString(SensitiveContent.PREFERENCE_KEY, mode)
            .commit()
    }

    private fun setVerifiedInstall(value: Boolean?) {
        val field = SensitiveContentModelManager::class.java.getDeclaredField("verifiedInstall")
        field.isAccessible = true
        field.set(SensitiveContentModelManager, value)
    }

    private fun installDummyModels() {
        val dir = SensitiveContentModelManager.modelsDir(context)
        if (!dir.exists()) dir.mkdirs()
        File(dir, "nsfw.tflite").writeBytes(byteArrayOf(0))
        File(dir, "person.tflite").writeBytes(byteArrayOf(0))
        File(dir, "gender.tflite").writeBytes(byteArrayOf(0))
        setVerifiedInstall(true)
    }

    @After
    fun reset() {
        setMode(SensitiveContent.OFF)
        SensitiveContentModelManager.modelsDir(context).deleteRecursively()
        setVerifiedInstall(null)
    }

    private class FakeClassifier(
        private val nsfw: Float? = 0f,
        private val female: Boolean = false
    ) : ContentClassifier {
        override fun nsfwScore(context: Context, bitmap: Bitmap): Float? = nsfw

        override fun femaleBodyRegions(
            context: Context,
            bitmap: Bitmap
        ): List<NormalizedRect> {
            return if (female) {
                listOf(NormalizedRect(0.25f, 0.25f, 0.75f, 0.75f))
            } else {
                emptyList()
            }
        }
    }

    private fun testBitmap(size: Int = 64): Bitmap {
        val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        for (y in 0 until size) {
            for (x in 0 until size) {
                bitmap.setPixel(x, y, if (x < size / 2) Color.BLACK else Color.WHITE)
            }
        }
        return bitmap
    }

    @Test
    fun protectionIsOffByDefault() {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .remove(SensitiveContent.PREFERENCE_KEY)
            .commit()

        assertEquals(SensitiveContent.OFF, SensitiveContent.mode(context))
        assertFalse(SensitiveContent.isEnabled(context))
    }

    @Test
    fun selectingAModeEnablesProtection() {
        setMode(SensitiveContent.NSFW_AND_WOMEN)

        assertTrue(SensitiveContent.isEnabled(context))
        assertEquals(SensitiveContent.NSFW_AND_WOMEN, SensitiveContent.mode(context))
    }

    @Test
    fun memoryCacheKeySeparatesModes() {
        assertEquals("url", SensitiveContent.memoryCacheKey("url", SensitiveContent.OFF))
        assertEquals(
            "wizestream-sensitive:nsfw_only:url",
            SensitiveContent.memoryCacheKey("url", SensitiveContent.NSFW_ONLY)
        )
        assertNull(SensitiveContent.memoryCacheKey(null, SensitiveContent.BLUR_ALL))
    }

    @Test
    fun blurAllBlursEveryThumbnail() = runBlocking {
        val input = testBitmap()
        val output = SensitiveContentTransformation(
            context,
            SensitiveContent.BLUR_ALL,
            FakeClassifier()
        ).transform(input, Size.ORIGINAL)

        assertEquals(input.width, output.width)
        assertNotEquals(
            Color.red(input.getPixel(input.width / 2 - 1, input.height / 2)),
            Color.red(output.getPixel(input.width / 2 - 1, input.height / 2))
        )
    }

    @Test
    fun nsfwOnlyBlursOnlyExplicitThumbnails() = runBlocking {
        installDummyModels()
        val input = testBitmap()
        val blurred = SensitiveContentTransformation(
            context,
            SensitiveContent.NSFW_ONLY,
            FakeClassifier(nsfw = SensitiveContent.NSFW_THRESHOLD + 0.1f)
        ).transform(input, Size.ORIGINAL)
        assertNotEquals(input, blurred)

        val safe = SensitiveContentTransformation(
            context,
            SensitiveContent.NSFW_ONLY,
            FakeClassifier(nsfw = 0.05f)
        ).transform(input, Size.ORIGINAL)
        assertEquals(input, safe)
    }

    @Test
    fun womenModeBlursThumbnailsWithAFemaleBody() = runBlocking {
        installDummyModels()
        val input = testBitmap()
        val withWoman = SensitiveContentTransformation(
            context,
            SensitiveContent.NSFW_AND_WOMEN,
            FakeClassifier(nsfw = 0f, female = true)
        ).transform(input, Size.ORIGINAL)
        assertNotEquals(input, withWoman)

        val withoutWoman = SensitiveContentTransformation(
            context,
            SensitiveContent.NSFW_AND_WOMEN,
            FakeClassifier(nsfw = 0f, female = false)
        ).transform(input, Size.ORIGINAL)
        assertEquals(input, withoutWoman)
    }

    @Test
    fun blurBitmapKeepsSizeAndSoftenEdges() {
        val input = testBitmap()
        val output = blurBitmap(input, 4)

        assertEquals(input.width, output.width)
        assertEquals(input.height, output.height)
        assertTrue(Color.red(output.getPixel(input.width / 2 - 1, input.height / 2)) > 0)
        assertEquals(0, Color.red(output.getPixel(0, input.height / 2)))
    }

    @Test
    fun blurRadiusGrowsWithThumbnailSizeAndIsClamped() {
        assertEquals(6, blurRadiusFor(32, 32))
        assertTrue(blurRadiusFor(640, 360) > blurRadiusFor(160, 90))
        assertEquals(32, blurRadiusFor(4000, 4000))
    }

    @Test
    fun blurRadiusRespectsIntensity() {
        val low = blurRadiusFor(640, 360, SensitiveContent.INTENSITY_LOW)
        val medium = blurRadiusFor(640, 360, SensitiveContent.INTENSITY_MEDIUM)
        val high = blurRadiusFor(640, 360, SensitiveContent.INTENSITY_HIGH)
        assertTrue(low < medium)
        assertTrue(medium < high)
    }

    @Test
    fun nsfwInputUsesRgbFloatsInZeroToOne() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(10, 20, 30))

        val buffer = nsfwInputBuffer(bitmap)

        assertEquals(10f / 255f, buffer.getFloat(0), 0.001f)
        assertEquals(20f / 255f, buffer.getFloat(4), 0.001f)
        assertEquals(30f / 255f, buffer.getFloat(8), 0.001f)
    }

    @Test
    fun genderInputUsesRgbFloatsInZeroToOne() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(10, 20, 30))

        val buffer = genderInputBuffer(bitmap)

        assertEquals(10f / 255f, buffer.getFloat(0), 0.001f)
        assertEquals(20f / 255f, buffer.getFloat(4), 0.001f)
        assertEquals(30f / 255f, buffer.getFloat(8), 0.001f)
    }

    @Test
    fun personInputUsesRgbBytes() {
        val bitmap = Bitmap.createBitmap(8, 8, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.rgb(10, 20, 30))

        val buffer = personInputBuffer(bitmap)

        assertEquals(10, buffer.get(0).toInt() and 0xFF)
        assertEquals(20, buffer.get(1).toInt() and 0xFF)
        assertEquals(30, buffer.get(2).toInt() and 0xFF)
    }

    @Test
    fun normalizedPersonBoxReordersCoordinates() {
        val rect = normalizedPersonBox(floatArrayOf(0.1f, 0.2f, 0.6f, 0.8f))

        assertEquals(0.2f, rect?.left ?: 0f, 0.001f)
        assertEquals(0.1f, rect?.top ?: 0f, 0.001f)
        assertEquals(0.8f, rect?.right ?: 0f, 0.001f)
        assertEquals(0.6f, rect?.bottom ?: 0f, 0.001f)
        assertNull(normalizedPersonBox(floatArrayOf(0f, 0f, 0f, 0f)))
    }

    @Test
    fun cropHeadUsesTopCenterOfDetectedBody() {
        val source = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val body = NormalizedRect(0.2f, 0.1f, 0.8f, 0.9f)

        val crop = cropHead(source, body, 0.25f)

        assertEquals(20, crop?.width)
        assertEquals(20, crop?.height)
    }

    @Test
    fun normalizedFaceBoxUsesBitmapBounds() {
        val bitmap = Bitmap.createBitmap(100, 200, Bitmap.Config.ARGB_8888)
        val rect = normalizedFaceBox(Rect(25, 40, 75, 90), bitmap)

        assertEquals(0.25f, rect?.left ?: 0f, 0.001f)
        assertEquals(0.2f, rect?.top ?: 0f, 0.001f)
        assertEquals(0.75f, rect?.right ?: 0f, 0.001f)
        assertEquals(0.45f, rect?.bottom ?: 0f, 0.001f)
    }

    @Test
    fun cropFaceAddsMarginAndClampsToTheImage() {
        val source = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)

        val crop = cropFace(source, Rect(20, 20, 60, 60))
        assertEquals(46, crop?.width)
        assertEquals(46, crop?.height)

        val clamped = cropFace(source, Rect(0, 0, 10, 10))
        assertEquals(10, clamped?.width)
        assertNull(cropFace(source, Rect(10, 10, 10, 10)))
    }

    @Test
    fun faceBoxExpansionCoversUpperBody() {
        val face = NormalizedRect(0.4f, 0.2f, 0.6f, 0.4f)

        val body = face.expandedToUpperBody()

        assertEquals(0.34f, body.left, 0.001f)
        assertEquals(0.14f, body.top, 0.001f)
        assertEquals(0.66f, body.right, 0.001f)
        assertEquals(0.96f, body.bottom, 0.001f)
    }
}
