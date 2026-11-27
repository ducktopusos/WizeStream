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

/** On-device classification used to decide whether a thumbnail must be blurred. */
interface ContentClassifier {
    /** Probability in `0..1` that [bitmap] is explicit, or `null` when unavailable. */
    fun nsfwScore(context: Context, bitmap: Bitmap): Float?

    /** Whether [bitmap] contains a face classified as female. */
    fun hasFemaleFace(context: Context, bitmap: Bitmap): Boolean

    /** Normalized face regions classified as female. Default falls back to none. */
    fun femaleFaceRegions(context: Context, bitmap: Bitmap): List<NormalizedRect> = emptyList()
}

/**
 * Bundled-model implementation.
 *
 * * Explicit content: GantMan/nsfw_model MobileNetV2 140x224 (`nsfw.tflite`,
 *   MIT), the same model family used by nsfwjs/HaramBlur. Input `1x224x224x3`
 *   RGB in `0..1`, output `[drawings, hentai, neutral, porn, sexy]`.
 * * Faces: ML Kit face detection, the on-device equivalent of the BlazeFace
 *   detector used by HaramBlur.
 * * Gender: `models/gender.tflite` from shubham0204's MIT-licensed
 *   Age-Gender_Estimation project (UTKFace, input `1x128x128x3` in `0..1`,
 *   output `[male, female]`). It expects a tight face crop.
 *
 * Inference is CPU-only, offline and serialized: the TFLite interpreters are
 * not thread-safe and Coil runs transformations on several threads.
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
            val p = output[0]
            // [drawings, hentai, neutral, porn, sexy]
            maxOf(p[1], p[3], p[4], 1f - p[2] - p[0])
        }.onFailure { Log.w(TAG, "NSFW inference failed", it) }.getOrNull()
    }

    override fun hasFemaleFace(context: Context, bitmap: Bitmap): Boolean = femaleFaceRegions(context, bitmap).isNotEmpty()

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
                val female = probabilities[1]
                if (female >= SensitiveContent.FEMALE_THRESHOLD) {
                    expandFaceToBody(
                        NormalizedRect(
                            left = (box.left.toFloat() / bitmap.width).coerceIn(0f, 1f),
                            top = (box.top.toFloat() / bitmap.height).coerceIn(0f, 1f),
                            right = (box.right.toFloat() / bitmap.width).coerceIn(0f, 1f),
                            bottom = (box.bottom.toFloat() / bitmap.height).coerceIn(0f, 1f)
                        )
                    )
                } else {
                    null
                }
            }
        }.onFailure { Log.w(TAG, "Face/gender inference failed", it) }.getOrDefault(emptyList())
    }

    private fun expandFaceToBody(region: NormalizedRect): NormalizedRect {
        val w = region.right - region.left
        val h = region.bottom - region.top
        if (h >= 0.25f && h >= w * 1.2f) return region
        val cx = (region.left + region.right) / 2f
        val top = (region.top - h * 0.6f).coerceIn(0f, 1f)
        val bottom = (region.bottom + h * 6.5f).coerceIn(0f, 1f)
        val halfWidth = (w * 2.2f).coerceAtLeast(h * 1.4f)
        return NormalizedRect(
            left = (cx - halfWidth).coerceIn(0f, 1f),
            top = top,
            right = (cx + halfWidth).coerceIn(0f, 1f),
            bottom = bottom
        )
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
            // Keep face detector; it is lightweight and independent of model files.
            initialized = false
        }
    }

    private fun ensureInitialized(context: Context) {
        if (initialized) {
            return
        }
        synchronized(initLock) {
            if (initialized) {
                return
            }
            val appContext = context.applicationContext
            if (!SensitiveContentModelManager.isInstalled(appContext)) {
                // Download on demand when an AI mode is active.
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

/**
 * Scales to 224x224 RGB in 0..1, matching the GantMan/nsfw_model MobileNetV2
 * input used by nsfwjs/HaramBlur.
 */
internal fun nsfwInputBuffer(bitmap: Bitmap): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, NSFW_INPUT_SIZE, NSFW_INPUT_SIZE, true)
    val pixels = IntArray(NSFW_INPUT_SIZE * NSFW_INPUT_SIZE)
    scaled.getPixels(pixels, 0, NSFW_INPUT_SIZE, 0, 0, NSFW_INPUT_SIZE, NSFW_INPUT_SIZE)

    val buffer = ByteBuffer
        .allocateDirect(4 * NSFW_INPUT_SIZE * NSFW_INPUT_SIZE * 3)
        .order(ByteOrder.nativeOrder())
    for (pixel in pixels) {
        buffer.putFloat(Color.red(pixel) / 255f)
        buffer.putFloat(Color.green(pixel) / 255f)
        buffer.putFloat(Color.blue(pixel) / 255f)
    }
    buffer.rewind()
    return buffer
}

/** Scales to 128x128 and converts to RGB floats in `0..1`, as the gender model expects. */
internal fun genderInputBuffer(bitmap: Bitmap): ByteBuffer {
    val scaled = Bitmap.createScaledBitmap(bitmap, GENDER_INPUT_SIZE, GENDER_INPUT_SIZE, true)
    val pixels = IntArray(GENDER_INPUT_SIZE * GENDER_INPUT_SIZE)
    scaled.getPixels(pixels, 0, GENDER_INPUT_SIZE, 0, 0, GENDER_INPUT_SIZE, GENDER_INPUT_SIZE)

    val buffer = ByteBuffer
        .allocateDirect(1 * GENDER_INPUT_SIZE * GENDER_INPUT_SIZE * 3 * 4)
        .order(ByteOrder.nativeOrder())
    for (pixel in pixels) {
        buffer.putFloat(Color.red(pixel) / 255f)
        buffer.putFloat(Color.green(pixel) / 255f)
        buffer.putFloat(Color.blue(pixel) / 255f)
    }
    buffer.rewind()
    return buffer
}

/**
 * Crops [box] out of [source] with a small margin. The gender model was trained
 * on tight UTKFace-style crops, so background-heavy crops mislead it.
 */
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
