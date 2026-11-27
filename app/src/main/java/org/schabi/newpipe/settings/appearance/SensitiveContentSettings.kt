package org.schabi.newpipe.settings.appearance

import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import androidx.preference.ListPreference
import androidx.preference.Preference
import org.schabi.newpipe.R
import org.schabi.newpipe.settings.BasePreferenceFragment
import org.schabi.newpipe.util.image.SensitiveContent
import org.schabi.newpipe.util.image.SensitiveContentModelManager

object SensitiveContentSettings {
    private const val MODELS_KEY = "sensitive_content_models"

    @JvmStatic
    fun bind(fragment: BasePreferenceFragment) {
        val modePref = fragment.findPreference<ListPreference>(SensitiveContent.PREFERENCE_KEY)
        val intensityPref = fragment.findPreference<ListPreference>(SensitiveContent.INTENSITY_KEY)
        val modelsPref = fragment.findPreference<Preference>(MODELS_KEY)

        fun updateVisibility() {
            val mode = modePref?.value ?: SensitiveContent.OFF
            intensityPref?.isVisible = mode != SensitiveContent.OFF
            modelsPref?.isVisible = SensitiveContent.requiresAiModels(mode)
            updateModelSummary(fragment, modelsPref)
        }

        modePref?.onPreferenceChangeListener = Preference.OnPreferenceChangeListener { _, newValue ->
            val mode = newValue?.toString() ?: SensitiveContent.OFF
            if (SensitiveContent.requiresAiModels(mode)) {
                SensitiveContentModelManager.ensureDownloaded(fragment.requireContext())
            }
            fragment.view?.post { updateVisibility() }
            true
        }

        modelsPref?.onPreferenceClickListener = Preference.OnPreferenceClickListener {
            SensitiveContentModelManager.ensureDownloaded(fragment.requireContext(), force = true)
            updateModelSummary(fragment, modelsPref)
            true
        }

        val listener: (SensitiveContentModelManager.State) -> Unit = {
            if (fragment.isAdded) updateModelSummary(fragment, modelsPref)
        }
        SensitiveContentModelManager.addListener(listener)
        fragment.viewLifecycleOwnerLiveData.observe(fragment) { owner ->
            owner?.lifecycle?.addObserver(object : DefaultLifecycleObserver {
                override fun onDestroy(owner: LifecycleOwner) {
                    SensitiveContentModelManager.removeListener(listener)
                }
            })
        }

        updateVisibility()
    }

    @JvmStatic
    fun refresh(fragment: BasePreferenceFragment) {
        updateModelSummary(fragment, fragment.findPreference(MODELS_KEY))
    }

    private fun updateModelSummary(fragment: BasePreferenceFragment, modelsPref: Preference?) {
        val pref = modelsPref ?: return
        val state = SensitiveContentModelManager.getState(fragment.requireContext())
        pref.summary = when (state.status) {
            SensitiveContentModelManager.Status.INSTALLED ->
                fragment.getString(R.string.sensitive_content_models_installed)

            SensitiveContentModelManager.Status.DOWNLOADING ->
                fragment.getString(R.string.sensitive_content_models_downloading)

            SensitiveContentModelManager.Status.FAILED ->
                state.message?.let {
                    fragment.getString(R.string.sensitive_content_models_failed) + " " + it
                } ?: fragment.getString(R.string.sensitive_content_models_failed)

            SensitiveContentModelManager.Status.NOT_INSTALLED ->
                fragment.getString(R.string.sensitive_content_models_not_installed)
        }
    }
}
