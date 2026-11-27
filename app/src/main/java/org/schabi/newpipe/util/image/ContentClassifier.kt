package org.schabi.newpipe.util.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

private const val NSFW_MODEL = "nsfw.tflite"
private const val PERSON_MODEL = "person.tflite"
private const val GENDER_MODEL = "gender.tflite"
private const val NSFW_INPUT_SIZE = 224
private const val PERSON_INPUT_SIZE = 320
private const val GENDER_INPUT_SIZE = 128
private const val MAX_PERSON_DETECTIONS = 25
private const val PERSON_CLASS = 0
private const val PERSON_THRESHOLD = 0.5f
private const val HEAD_REGION_FRACTION = 0.3f

/** A normalized rectangle in bitmap coordinates, values in `0..1`. */
data class NormalizedRect(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/** On-device classification used to decide which content must be blurred. */
interface ContentClassifier {
    /** Probability in `0..1` that [bitmap] is explicit, or `null` when unavailable. */
    fun nsfwScore(context: Context, bitmap: Bitmap): Float?

    /** Normalized full-body regions of people classified as female. */
    fun femaleBodyRegions(context: Context, bitmap: Bitmap): List<NormalizedRect>
}

/**
 * On-device implementation.
 *
 * NSFW: GantMan/nsfw_model MobileNetV2 TFLite, RGB in `0..1`, five outputs.
 * Bodies: EfficientDet Lite0 TFLite COCO person boxes.
 * Gender: UTKFace TFLite on the head region of each detected body.
 * Interpreters are serialized because Coil runs transformations on several threads.
 */
object OnDeviceContentClassifier : ContentClassifier {
    private const val TAG = "SensitiveContent"
    private val initLock = Any()
    private val nsfwLock = Any()
    private val personLock = Any()
    private val genderLock = Any()

    @Volatile
    private var initialized = false
    private var nsfwInterpreter: Interpreter? = null
    private var personInterpreter: Interpreter? = null
    private var genderInterpreter: Interpreter? = null

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

    override fun femaleBodyRegions(context: Context, bitmap: Bitmap): List<NormalizedRect> {
        ensureInitialized(context)
        val detector = personInterpreter ?: return emptyList()
        val interpreter = genderInterpreter ?: return emptyList()
        return runCatching {
            detectPeople(detector, bitmap).mapNotNull { region ->
                val crop = cropHead(bitmap, region) ?: return@mapNotNull null
                val probabilities = classifyGender(interpreter, crop) ?: return@mapNotNull null
                if (probabilities[1] < SensitiveContent.FEMALE_THRESHOLD) {
                    return@mapNotNull null
                }
                region
            }
        }.onFailure { Log.w(TAG, "Person/gender inference failed", it) }
            .getOrDefault(emptyList())
    }

    fun reset() {
        synchronized(initLock) {
            synchronized(nsfwLock) {
                runCatching { nsfwInterpreter?.close() }
                nsfwInterpreter = null
            }
            synchronized(personLock) {
                runCatching { personInterpreter?.close() }
                personInterpreter = null
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
            personInterpreter = loadInterpreter(SensitiveContentModelManager.modelFile(appContext, PERSON_MODEL))
            genderInterpreter = loadInterpreter(SensitiveContentModelManager.modelFile(appContext, GENDER_MODEL))
            initialized = nsfwInterpreter != null && personInterpreter != null && genderInterpreter != null
        }
    }

    private fun loadInterpreter(file: File): Interpreter? = runCatching {
        if (!file.isFile) return@runCatching null
        FileInputStream(file).channel.use { channel ->
            val mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            Interpreter(mapped, Interpreter.Options().apply { setNumThreads(2) })
        }
    }.onFailure { Log.w(TAG, "Could not load ${file.name}", it) }.getOrNull()

    private fun detectPeople(interpreter: Interpreter, bitmap: Bitmap): List<NormalizedRect> {
        val boxes = Array(1) { Array(MAX_PERSON_DETECTIONS) { FloatArray(4) } }
        val classes = Array(1) { FloatArray(MAX_PERSON_DETECTIONS) }
        val scores = Array(1) { FloatArray(MAX_PERSON_DETECTIONS) }
        val count = Array(1) { FloatArray(1) }
        synchronized(personLock) {
            interpreter.run(personInputBuffer(bitmap), arrayOf<Any>(boxes, classes, scores, count))
        }
        val total = count[0][0].toInt().coerceIn(0, MAX_PERSON_DETECTIONS)
        return (0 until total).mapNotNull { index ->
            if (classes[0][index].toInt() != PERSON_CLASS ||
                scores[0][index] < PERSON_THRESHOLD
            ) {
                return@mapNotNull null
            }
            normalizedPersonBox(boxes[0][index])
        }
    }

    private fun classifyGender(interpreter: Interpreter, crop: Bitmap): FloatArray? = runCatching {
        val output = Array(1) { FloatArray(2) }
        synchronized(genderLock) {
            interpreter.run(genderInputBuffer(crop), output)
        }
        output[0]
    }.onFailure { Log.w(TAG, "Gender inference failed", it) }.getOrNull()
}

/** Converts EfficientDet's `[ymin, xmin, ymax, xmax]` output to a normalized rect. */
internal fun normalizedPersonBox(box: FloatArray): NormalizedRect? {
    val left = box[1].coerceIn(0f, 1f)
    val top = box[0].coerceIn(0f, 1f)
    val right = box[3].coerceIn(0f, 1f)
    val bottom = box[2].coerceIn(0f, 1f)
    if (right - left < 0.02f || bottom - top < 0.02f) {
        return null
    }
    return NormalizedRect(left = left, top = top, right = right, bottom = bottom)
}

/** Scales to 224x224 RGB floats in `0..1` for the NSFW model. */
internal fun nsfwInputBuffer(bitmap: Bitmap): ByteBuffer = rgbInputBuffer(bitmap, NSFW_INPUT_SIZE)

/** Scales to 128x128 RGB floats in `0..1` for the gender model. */
internal fun genderInputBuffer(bitmap: Bitmap): ByteBuffer = rgbInputBuffer(bitmap, GENDER_INPUT_SIZE)

/** Scales to 320x320 RGB bytes for EfficientDet Lite0. */
internal fun personInputBuffer(bitmap: Bitmap): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, PERSON_INPUT_SIZE, PERSON_INPUT_SIZE, true)
    val pixels = IntArray(PERSON_INPUT_SIZE * PERSON_INPUT_SIZE)
    scaled.getPixels(pixels, 0, PERSON_INPUT_SIZE, 0, 0, PERSON_INPUT_SIZE, PERSON_INPUT_SIZE)
    if (scaled !== bitmap) scaled.recycle()

    val buffer = ByteBuffer
        .allocateDirect(PERSON_INPUT_SIZE * PERSON_INPUT_SIZE * 3)
        .order(ByteOrder.nativeOrder())
    for (pixel in pixels) {
        buffer.put((pixel shr 16 and 0xFF).toByte())
        buffer.put((pixel shr 8 and 0xFF).toByte())
        buffer.put((pixel and 0xFF).toByte())
    }
    buffer.rewind()
    return buffer
}

private fun rgbInputBuffer(bitmap: Bitmap, size: Int): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
    val pixels = IntArray(size * size)
    scaled.getPixels(pixels, 0, size, 0, 0, size, size)
    if (scaled !== bitmap) scaled.recycle()

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

/** Crops the head area from a detected body for the gender model. */
internal fun cropHead(
    source: Bitmap,
    region: NormalizedRect,
    fraction: Float = HEAD_REGION_FRACTION
): Bitmap? {
    val left = (region.left * source.width).toInt()
    val right = (region.right * source.width).toInt()
    val top = (region.top * source.height).toInt()
    val bottom = (region.bottom * source.height).toInt()
    val bodyWidth = right - left
    val bodyHeight = bottom - top
    if (bodyWidth <= 0 || bodyHeight <= 0) {
        return null
    }

    val side = minOf(bodyWidth, (bodyHeight * fraction).toInt())
        .coerceAtLeast(8)
        .coerceAtMost(source.width)
    val centerX = (left + right) / 2
    val cropLeft = (centerX - side / 2).coerceIn(0, source.width - side)
    val cropTop = top.coerceIn(0, source.height - side)
    return Bitmap.createBitmap(source, cropLeft, cropTop, side, side)
}
