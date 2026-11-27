package org.schabi.newpipe.util.image

import android.content.Context
import android.os.Handler
import android.os.Looper
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

object SensitiveContentModelManager {
    enum class Status { NOT_INSTALLED, DOWNLOADING, INSTALLED, FAILED }

    data class State(val status: Status, val message: String? = null)

    private const val RELEASE_BASE =
        "https://github.com/ducktopusos/WizeStream/releases/download/sensitive-content-models-v1"

    private val models = listOf(
        Model(
            name = "nsfw.tflite",
            url = "$RELEASE_BASE/nsfw.tflite",
            sha256 = "380f98f7685f9d8a386f8cc595b6dfcb972989aae3d1b8b270d3a4a5b96fab40"
        ),
        Model(
            name = "person.tflite",
            url = "$RELEASE_BASE/person.tflite",
            sha256 = "2e04c53bfeac0ac2a30c057c7e2a777594ce39baaac35a92f74fb1e8c4fc4e0b"
        ),
        Model(
            name = "gender.tflite",
            url = "$RELEASE_BASE/gender.tflite",
            sha256 = "7ab000d530a005ccd605a2826485b23d08690d0e4fb23bc4589a5b03861750c0"
        )
    )

    private data class Model(val name: String, val url: String, val sha256: String)

    private val downloading = AtomicBoolean(false)

    @Volatile
    private var verifiedInstall: Boolean? = null

    @Volatile
    private var state: State = State(Status.NOT_INSTALLED)
    private val listeners = mutableSetOf<(State) -> Unit>()
    private val mainHandler = Handler(Looper.getMainLooper())

    fun modelsDir(context: Context): File = File(context.noBackupFilesDir ?: context.filesDir, "sensitive_content_models")

    fun modelFile(context: Context, name: String): File = File(modelsDir(context), name)

    fun isInstalled(context: Context): Boolean {
        verifiedInstall?.let { return it }
        val ok = models.all { model ->
            val file = modelFile(context, model.name)
            file.isFile && file.length() > 0 && sha256(file) == model.sha256.lowercase()
        }
        verifiedInstall = ok
        return ok
    }

    fun getState(context: Context): State {
        if (state.status != Status.DOWNLOADING) {
            state = if (isInstalled(context)) {
                State(Status.INSTALLED)
            } else {
                State(Status.NOT_INSTALLED)
            }
        }
        return state
    }

    fun addListener(listener: (State) -> Unit) {
        synchronized(listeners) { listeners.add(listener) }
    }

    fun removeListener(listener: (State) -> Unit) {
        synchronized(listeners) { listeners.remove(listener) }
    }

    fun ensureDownloaded(context: Context, force: Boolean = false) {
        val appContext = context.applicationContext
        if (force) verifiedInstall = null
        if (!force && isInstalled(appContext)) {
            updateState(State(Status.INSTALLED))
            return
        }
        if (!downloading.compareAndSet(false, true)) return

        updateState(State(Status.DOWNLOADING))
        Thread {
            try {
                val dir = modelsDir(appContext)
                if (!dir.exists()) dir.mkdirs()
                models.forEach { downloadModel(it, File(dir, it.name)) }
                verifiedInstall = null
                OnDeviceContentClassifier.reset()
                updateState(State(Status.INSTALLED))
            } catch (t: Throwable) {
                updateState(State(Status.FAILED, t.message))
            } finally {
                downloading.set(false)
            }
        }.start()
    }

    private fun downloadModel(model: Model, target: File) {
        val tmp = File(target.parentFile, target.name + ".tmp")
        val connection = (URL(model.url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }
        try {
            connection.inputStream.use { input ->
                FileOutputStream(tmp).use { output -> input.copyTo(output) }
            }
            if (!sha256(tmp).equals(model.sha256, ignoreCase = true)) {
                tmp.delete()
                throw IllegalStateException("Checksum mismatch for ${model.name}")
            }
            if (target.exists()) target.delete()
            if (!tmp.renameTo(target)) {
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
        } finally {
            connection.disconnect()
            if (tmp.exists() && target.exists()) tmp.delete()
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun updateState(newState: State) {
        state = newState
        val snapshot = synchronized(listeners) { listeners.toList() }
        mainHandler.post { snapshot.forEach { it(newState) } }
    }
}
