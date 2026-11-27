package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

private const val NSFW_MODEL = "nsfw.tflite"
private const val GENDER_MODEL = "gender.tflite"
private const val NSFW_INPUT_SIZE = 224
private const val GENDER_INPUT_SIZE = 128
private const val FACE_MARGIN_RATIO = 0.08f

/** A normalized rectangle in bitmap coordinates, values in `0..1`. */
data class NormalizedRect(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/** Approximates a full body from a face-sized region. */
internal fun NormalizedRect.expandedToBody(): NormalizedRect {
    val width = right - left
    val height = bottom - top
    if (height >= 0.25f && height >= width * 1.2f) return this

    val centerX = (left + right) / 2f
    val halfWidth = (width * 2.2f).coerceAtLeast(height * 1.4f)
    val expandedTop = (top - height * 0.6f).coerceIn(0f, 1f)
    val expandedBottom = (bottom + height * 6.5f).coerceIn(0f, 1f)
    return NormalizedRect(
        left = (centerX - halfWidth).coerceIn(0f, 1f),
        top = expandedTop,
        right = (centerX + halfWidth).coerceIn(0f, 1f),
        bottom = expandedBottom
    )
}

/** On-device classification used to decide which content must be blurred. */
interface ContentClassifier {
    /** Probability in `0..1` that [bitmap] is explicit, or `null` when unavailable. */
    fun nsfwScore(context: Context, bitmap: Bitmap): Float?

    /** Normalized regions containing a face classified as female. */
    fun femaleFaceRegions(context: Context, bitmap: Bitmap): List<NormalizedRect>
}

/**
 * On-device implementation.
 *
 * NSFW: GantMan/nsfw_model MobileNetV2 TFLite, RGB in `0..1`, five outputs.
 * Faces: ML Kit. Gender: UTKFace TFLite on a tight face crop.
 * Interpreters are serialized because Coil runs transformations on several threads.
 */
object OnDeviceContentClassifier : ContentClassifier {
    private const val TAG = "SensitiveContent"
    private val initLock = Any()
    private val nsfwLock = Any()
    private val genderLock = Any()

    @Volatile
    private var initialized = false
    private var nsfwInterpreter: Interpreter? = null
    private var genderInterpreter: Interpreter? = null
    private var faceDetector: FaceDetector? = null

    override fun nsfwScore(context: Context, bitmap: Bitmap): Float? {
        ensureInitialized(context)
        val interpreter = nsfwInterpreter ?: return null
        return runCatching {
            val output = Array(1) { FloatArray(5) }
            synchronized(nsfwLock) {
                interpreter.run(nsfwInputBuffer(bitmap), output)
            }
            val scores = output[0]
            // [drawings, hentai, neutral, porn, sexy]
            maxOf(scores[1], scores[3], scores[4], 1f - scores[2] - scores[0])
        }.onFailure { Log.w(TAG, "NSFW inference failed", it) }.getOrNull()
    }

    override fun femaleFaceRegions(context: Context, bitmap: Bitmap): List<NormalizedRect> {
        ensureInitialized(context)
        val detector = faceDetector ?: return emptyList()
        val interpreter = genderInterpreter ?: return emptyList()
        return runCatching {
            val image = InputImage.fromBitmap(bitmap, 0)
            val faces = Tasks.await(detector.process(image))
            faces.mapNotNull { face ->
                val box = face.boundingBox
                val crop = cropFace(bitmap, box) ?: return@mapNotNull null
                val probabilities = classifyGender(interpreter, crop) ?: return@mapNotNull null
                if (probabilities[1] < SensitiveContent.FEMALE_THRESHOLD) {
                    return@mapNotNull null
                }
                NormalizedRect(
                    left = (box.left.toFloat() / bitmap.width).coerceIn(0f, 1f),
                    top = (box.top.toFloat() / bitmap.height).coerceIn(0f, 1f),
                    right = (box.right.toFloat() / bitmap.width).coerceIn(0f, 1f),
                    bottom = (box.bottom.toFloat() / bitmap.height).coerceIn(0f, 1f)
                ).expandedToBody()
            }
        }.onFailure { Log.w(TAG, "Face/gender inference failed", it) }.getOrDefault(emptyList())
    }

    fun reset() {
        synchronized(initLock) {
            synchronized(nsfwLock) {
                runCatching { nsfwInterpreter?.close() }
                nsfwInterpreter = null
            }
            synchronized(genderLock) {
                runCatching { genderInterpreter?.close() }
                genderInterpreter = null
            }
            initialized = false
        }
    }

    private fun ensureInitialized(context: Context) {
        if (initialized) return
        synchronized(initLock) {
            if (initialized) return
            val appContext = context.applicationContext
            if (!SensitiveContentModelManager.isInstalled(appContext)) {
                SensitiveContentModelManager.ensureDownloaded(appContext)
                return
            }
            nsfwInterpreter = loadInterpreter(SensitiveContentModelManager.modelFile(appContext, NSFW_MODEL))
            genderInterpreter = loadInterpreter(SensitiveContentModelManager.modelFile(appContext, GENDER_MODEL))
            if (faceDetector == null) {
                faceDetector = runCatching {
                    FaceDetection.getClient(
                        FaceDetectorOptions.Builder()
                            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                            .setMinFaceSize(0.05f)
                            .build()
                    )
                }.onFailure { Log.w(TAG, "Face detector unavailable", it) }.getOrNull()
            }
            initialized = nsfwInterpreter != null || genderInterpreter != null
        }
    }

    private fun loadInterpreter(file: File): Interpreter? = runCatching {
        if (!file.isFile) return@runCatching null
        FileInputStream(file).channel.use { channel ->
            val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            Interpreter(mapped, Interpreter.Options().apply { setNumThreads(2) })
        }
    }.onFailure { Log.w(TAG, "Could not load ${file.name}", it) }.getOrNull()

    private fun classifyGender(interpreter: Interpreter, crop: Bitmap): FloatArray? = runCatching {
        val output = Array(1) { FloatArray(2) }
        synchronized(genderLock) {
            interpreter.run(genderInputBuffer(crop), output)
        }
        output[0]
    }.onFailure { Log.w(TAG, "Gender inference failed", it) }.getOrNull()
}

/** Scales to 224x224 RGB floats in `0..1` for the NSFW model. */
internal fun nsfwInputBuffer(bitmap: Bitmap): ByteBuffer = rgbInputBuffer(bitmap, NSFW_INPUT_SIZE)

/** Scales to 128x128 RGB floats in `0..1` for the gender model. */
internal fun genderInputBuffer(bitmap: Bitmap): ByteBuffer = rgbInputBuffer(bitmap, GENDER_INPUT_SIZE)

private fun rgbInputBuffer(bitmap: Bitmap, size: Int): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
    val pixels = IntArray(size * size)
    scaled.getPixels(pixels, 0, size, 0, 0, size, size)

    val buffer = ByteBuffer
        .allocateDirect(size * size * 3 * 4)
        .order(ByteOrder.nativeOrder())
    for (pixel in pixels) {
        buffer.putFloat(Color.red(pixel) / 255f)
        buffer.putFloat(Color.green(pixel) / 255f)
        buffer.putFloat(Color.blue(pixel) / 255f)
    }
    buffer.rewind()
    return buffer
}

/** Crops [box] out of [source] with a small margin for the gender model. */
internal fun cropFace(source: Bitmap, box: Rect, marginRatio: Float = FACE_MARGIN_RATIO): Bitmap? {
    if (box.width() <= 0 || box.height() <= 0) {
        return null
    }
    val marginX = (box.width() * marginRatio).toInt()
    val marginY = (box.height() * marginRatio).toInt()
    val left = (box.left - marginX).coerceAtLeast(0)
    val top = (box.top - marginY).coerceAtLeast(0)
    val right = (box.right + marginX).coerceAtMost(source.width)
    val bottom = (box.bottom + marginY).coerceAtMost(source.height)
    if (right - left < 2 || bottom - top < 2) {
        return null
    }
    return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
}
