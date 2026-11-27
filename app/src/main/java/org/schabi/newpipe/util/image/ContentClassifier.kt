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
private const val PERSON_MODEL = "person.tflite"
private const val GENDER_MODEL = "gender.tflite"
private const val NSFW_INPUT_SIZE = 224
private const val PERSON_INPUT_SIZE = 320
private const val GENDER_INPUT_SIZE = 128
private const val MAX_PERSON_DETECTIONS = 25
private const val PERSON_CLASS = 0
private const val PERSON_THRESHOLD = 0.5f
private const val HEAD_REGION_FRACTION = 0.3f
private const val FACE_MARGIN_RATIO = 0.08f

/** A normalized rectangle in bitmap coordinates, values in `0..1`. */
data class NormalizedRect(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/** Covers the visible upper body when only a face or half body is in frame. */
internal fun NormalizedRect.expandedToUpperBody(): NormalizedRect {
    val width = right - left
    val height = bottom - top
    val centerX = (left + right) / 2f
    val halfWidth = (width * 1.6f).coerceAtLeast(height * 0.6f) / 2f
    return NormalizedRect(
        left = (centerX - halfWidth).coerceIn(0f, 1f),
        top = (top - height * 0.3f).coerceIn(0f, 1f),
        right = (centerX + halfWidth).coerceIn(0f, 1f),
        bottom = (bottom + height * 2.8f).coerceIn(0f, 1f)
    )
}

private fun NormalizedRect.containsCenter(other: NormalizedRect): Boolean {
    val x = (other.left + other.right) / 2f
    val y = (other.top + other.bottom) / 2f
    return x in left..right && y in top..bottom
}

/** On-device classification used to decide which content must be blurred. */
interface ContentClassifier {
    /** Probability in `0..1` that [bitmap] is explicit, or `null` when unavailable. */
    fun nsfwScore(context: Context, bitmap: Bitmap): Float?

    /** Normalized regions of women/girls: detected bodies, half bodies or faces. */
    fun femaleBodyRegions(context: Context, bitmap: Bitmap): List<NormalizedRect>
}

/**
 * On-device implementation.
 *
 * NSFW: GantMan/nsfw_model MobileNetV2 TFLite, RGB in `0..1`, five outputs.
 * Bodies: EfficientDet Lite0 TFLite COCO person boxes.
 * Faces: ML Kit on-device detector for close-ups and partial bodies.
 * Gender: UTKFace TFLite on a crop of the detected body or face.
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

    override fun femaleBodyRegions(context: Context, bitmap: Bitmap): List<NormalizedRect> {
        ensureInitialized(context)
        val people = personInterpreter?.let { detectPeople(it, bitmap) }.orEmpty()
        val bodies = people.mapNotNull { region ->
            val crop = cropHead(bitmap, region) ?: return@mapNotNull null
            if (isFemale(crop)) region else null
        }

        val faces = faceDetector?.let { detectFaces(it, bitmap) }.orEmpty()
        val partialBodies = faces.mapNotNull { box ->
            val region = normalizedFaceBox(box, bitmap) ?: return@mapNotNull null
            if (bodies.any { it.containsCenter(region) }) {
                return@mapNotNull null
            }
            val crop = cropFace(bitmap, box) ?: return@mapNotNull null
            if (isFemale(crop)) region.expandedToUpperBody() else null
        }
        return bodies + partialBodies
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
            runCatching { faceDetector?.close() }
            faceDetector = null
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
        val outputs = mapOf<Int, Any>(0 to boxes, 1 to classes, 2 to scores, 3 to count)
        synchronized(personLock) {
            interpreter.runForMultipleInputsOutputs(arrayOf(personInputBuffer(bitmap)), outputs)
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

    private fun detectFaces(detector: FaceDetector, bitmap: Bitmap): List<Rect> = runCatching {
        Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0))).map { it.boundingBox }
    }.onFailure { Log.w(TAG, "Face detection failed", it) }.getOrDefault(emptyList())

    private fun isFemale(crop: Bitmap): Boolean {
        val interpreter = genderInterpreter ?: return false
        val probabilities = runCatching {
            val output = Array(1) { FloatArray(2) }
            synchronized(genderLock) {
                interpreter.run(genderInputBuffer(crop), output)
            }
            output[0]
        }.onFailure { Log.w(TAG, "Gender inference failed", it) }.getOrNull()
        return probabilities != null && probabilities[1] >= SensitiveContent.FEMALE_THRESHOLD
    }
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

/** Converts an ML Kit face box to normalized coordinates. */
internal fun normalizedFaceBox(box: Rect, bitmap: Bitmap): NormalizedRect? {
    if (box.width() <= 0 || box.height() <= 0) {
        return null
    }
    return NormalizedRect(
        left = (box.left.toFloat() / bitmap.width).coerceIn(0f, 1f),
        top = (box.top.toFloat() / bitmap.height).coerceIn(0f, 1f),
        right = (box.right.toFloat() / bitmap.width).coerceIn(0f, 1f),
        bottom = (box.bottom.toFloat() / bitmap.height).coerceIn(0f, 1f)
    )
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

/** Crops a detected face with a small margin for the gender model. */
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
