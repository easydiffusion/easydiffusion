// Saving settings
let saveSettingsCheckbox = document.getElementById("auto_save_settings")
let saveSettingsConfigTable = document.getElementById("save-settings-config-table")
let saveSettingsConfigOverlay = document.getElementById("save-settings-config")

const SETTINGS_KEY = "user_settings"
var SETTINGS_SHOULD_SAVE_MAP = {} // key=id. dict initialized in initSettings
var SETTINGS_VALUES = {} // key=id. dict initialized in initSettings
var SETTINGS_DEFAULTS = {} // key=id. dict initialized in initSettings
var SETTINGS_TO_SAVE = [] // list of elements initialized by initSettings
var SETTINGS_IDS_LIST = [
    "seed",
    "random_seed",
    "num_outputs_total",
    "num_outputs_parallel",
    "stable_diffusion_model",
    "sampler",
    "width",
    "height",
    "num_inference_steps",
    "guidance_scale_slider",
    "prompt_strength_slider",
    "output_format",
    "negative_prompt",
    "stream_image_progress",
    "use_face_correction",
    "use_upscale",
    "show_only_filtered_image",
    "upscale_model",
    "preview-image",
    "modifier-card-size-slider",
    "theme",
    "save_to_disk",
    "diskPath",
    "sound_toggle",
    "turbo",
    "use_cpu",
    "use_full_precision",
    "auto_save_settings"
]

async function initSettings() {
    SETTINGS_IDS_LIST.forEach(id => SETTINGS_TO_SAVE.push(document.getElementById(id)))
    SETTINGS_TO_SAVE.forEach(element => {
        SETTINGS_SHOULD_SAVE_MAP[element.id] = true
        SETTINGS_DEFAULTS[element.id] = getSetting(element)
        SETTINGS_VALUES[element.id] = getSetting(element)
        element.addEventListener("input", settingChangeHandler)
        element.addEventListener("change", settingChangeHandler)
    })
    loadSettings()
    fillSaveSettingsConfigTable()
}

function getSetting(element) {
    if (element instanceof String) {
        element = SETTINGS_TO_SAVE.find(e => e.id == element);
    }
    if (element.type == "checkbox") {
        return element.checked
    }
    return element.value
}
function setSetting(element, value) {
    if (getSetting(element) == value) {
        return // no setting necessary
    }
    if (element.type == "checkbox") {
        element.checked = value
    }
    else {
        element.value = value
    }
    element.dispatchEvent(new Event("input"))
    element.dispatchEvent(new Event("change"))
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        values: SETTINGS_VALUES,
        should_save: SETTINGS_SHOULD_SAVE_MAP
    }))
}


var CURRENTLY_LOADING_SETTINGS = false
function loadSettings() {
    if (!saveSettingsCheckbox.checked) {
        return
    }
    var saved_settings = JSON.parse(localStorage.getItem(SETTINGS_KEY))
    if (saved_settings) {
        var values = saved_settings.values
        var should_save = saved_settings.should_save
        CURRENTLY_LOADING_SETTINGS = true
        SETTINGS_TO_SAVE.forEach(element => {
            if (element.id in values) {
                SETTINGS_SHOULD_SAVE_MAP[element.id] = should_save[element.id]
                SETTINGS_VALUES[element.id] = values[element.id]
                if (SETTINGS_SHOULD_SAVE_MAP[element.id]) {
                    setSetting(element, SETTINGS_VALUES[element.id])
                }
            }
        })
        CURRENTLY_LOADING_SETTINGS = false
    }
    else {
        saveSettings()
    }
}

document.querySelector('#restoreDefaultSettingsBtn').addEventListener('click', loadDefaultSettings)
function loadDefaultSettings() {
    CURRENTLY_LOADING_SETTINGS = true
    SETTINGS_TO_SAVE.forEach(element => {
        SETTINGS_VALUES[element.id] = SETTINGS_DEFAULTS[element.id]
        setSetting(element, SETTINGS_VALUES[element.id])
    })
    CURRENTLY_LOADING_SETTINGS = false
    saveSettings()
}

function settingChangeHandler(event) {
    if (!CURRENTLY_LOADING_SETTINGS) {
        var element = event.target
        var value = getSetting(element)
        if (value != SETTINGS_VALUES[element.id]) {
            SETTINGS_VALUES[element.id] = value
            saveSettings()
        }
    }
}

function fillSaveSettingsConfigTable() {
    SETTINGS_TO_SAVE.forEach(element => {
        var caption = element.id
        var label = document.querySelector(`label[for='${element.id}']`)
        if (label) {
            caption = label.innerText
            var truncate_length = 25
            if (caption.length > truncate_length) {
                caption = caption.substring(0, truncate_length - 3) + "..."
            }
        }
        var default_value = SETTINGS_DEFAULTS[element.id]
        var checkbox_id = `shouldsave_${element.id}`
        var is_checked = SETTINGS_SHOULD_SAVE_MAP[element.id] ? "checked" : ""
        var newrow = `<tr><td><label for="${checkbox_id}">${caption}</label></td><td><input id="${checkbox_id}" name="${checkbox_id}" ${is_checked} type="checkbox" ></td><td><small>(${default_value})</small></td></tr>`
        saveSettingsConfigTable.insertAdjacentHTML("beforeend", newrow)
        var checkbox = document.getElementById(checkbox_id)
        checkbox.addEventListener("input", event => {
            SETTINGS_SHOULD_SAVE_MAP[element.id] = checkbox.checked
            saveSettings()
        })
    })
}

document.getElementById("save-settings-config-close-btn").addEventListener('click', () => {
    saveSettingsConfigOverlay.style.display = 'none'
})
document.getElementById("configureSettingsSaveBtn").addEventListener('click', () => {
    saveSettingsConfigOverlay.style.display = 'block'
})
saveSettingsConfigOverlay.addEventListener('click', (event) => {
    if (event.target.id == saveSettingsConfigOverlay.id) {
        saveSettingsConfigOverlay.style.display = 'none'
    }
})