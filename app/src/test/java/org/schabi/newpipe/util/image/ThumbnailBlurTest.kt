package org.schabi.newpipe.util.image

import android.graphics.Bitmap
import android.graphics.Color
import androidx.preference.PreferenceManager
import coil3.size.Size
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class ThumbnailBlurTest {
    private val context get() = RuntimeEnvironment.getApplication()

    private fun setStrength(strength: String) {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .putString(ThumbnailBlur.PREFERENCE_KEY, strength)
            .commit()
    }

    @After
    fun disableBlur() {
        setStrength(ThumbnailBlur.OFF)
    }

    @Test
    fun blurIsDisabledByDefault() {
        PreferenceManager.getDefaultSharedPreferences(context)
            .edit()
            .remove(ThumbnailBlur.PREFERENCE_KEY)
            .commit()

        assertEquals(ThumbnailBlur.OFF, ThumbnailBlur.strength(context))
        assertFalse(ThumbnailBlur.isEnabled(context))
        assertEquals(0, ThumbnailBlur.radius(context))
    }

    @Test
    fun strongerStrengthsUseLargerRadii() {
        assertEquals(0, ThumbnailBlur.radius(ThumbnailBlur.OFF))
        assertTrue(
            ThumbnailBlur.radius(ThumbnailBlur.LIGHT) <
                ThumbnailBlur.radius(ThumbnailBlur.MEDIUM)
        )
        assertTrue(
            ThumbnailBlur.radius(ThumbnailBlur.MEDIUM) <
                ThumbnailBlur.radius(ThumbnailBlur.STRONG)
        )
    }

    @Test
    fun selectingAStrengthEnablesBlur() {
        setStrength(ThumbnailBlur.MEDIUM)

        assertTrue(ThumbnailBlur.isEnabled(context))
        assertEquals(ThumbnailBlur.radius(ThumbnailBlur.MEDIUM), ThumbnailBlur.radius(context))
    }

    @Test
    fun memoryCacheKeyIsScopedPerStrength() {
        assertEquals("url", ThumbnailBlur.memoryCacheKey("url", 0))
        assertEquals(
            "wizestream-thumbnail-blur:8:url",
            ThumbnailBlur.memoryCacheKey("url", 8)
        )
        assertNull(ThumbnailBlur.memoryCacheKey(null, 16))
    }

    @Test
    fun thumbnailRequestsUseBlurScopedMemoryCacheKey() {
        setStrength(ThumbnailBlur.MEDIUM)
        val url = "https://example.com/thumb.jpg"

        val blurred = CoilHelper.getImageRequest(context, url, 0, blur = true).build()
        assertEquals(ThumbnailBlur.memoryCacheKey(url, 16), blurred.memoryCacheKey)

        val plain = CoilHelper.getImageRequest(context, url, 0).build()
        assertEquals(url, plain.memoryCacheKey)
    }

    @Test
    fun transformationBlursASharpEdge() = runBlocking {
        val width = 32
        val height = 8
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                bitmap.setPixel(x, y, if (x < width / 2) Color.BLACK else Color.WHITE)
            }
        }

        val blurred = ThumbnailBlurTransformation(4).transform(bitmap, Size(width, height))

        assertEquals(width, blurred.width)
        assertEquals(height, blurred.height)
        // the black side of the edge picks up some of the white side
        assertTrue(Color.red(blurred.getPixel(width / 2 - 1, height / 2)) > 0)
        // far away from the edge the image stays black
        assertEquals(0, Color.red(blurred.getPixel(0, height / 2)))
        // a radius of 0 is a no-op
        assertEquals(bitmap, ThumbnailBlurTransformation(0).transform(bitmap, Size(width, height)))
    }
}
