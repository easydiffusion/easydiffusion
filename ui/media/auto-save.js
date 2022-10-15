// Saving settings 

let autoSaveSettingsField = document.querySelector('#auto_save_settings')
let configureSettingsSaveBtn = document.querySelector('#configureSettingsSaveBtn')
let restoreDefaultSettingsBtn = document.querySelector('#restoreDefaultSettingsBtn')
let saveSettingsConfigOverlay = document.querySelector('#save-settings-config')
let saveSettingsConfigTable = document.querySelector('#save-settings-config-table')
let saveSettingsConfigCloseBtn = document.querySelector('#save-settings-config-close-btn')


const SETTINGS_KEY = "user_settings";
var SETTINGS_SHOULD_SAVE_MAP = {}; // key=id. dict initialized in initSettings
var SETTINGS_VALUES = {}; // key=id. dict initialized in initSettings
var SETTINGS_DEFAULTS = {}; // key=id. dict initialized in initSettings
var SETTINGS_TO_SAVE = [
    promptField,
    seedField,
    randomSeedField,
    numOutputsTotalField,
    numOutputsParallelField,
    stableDiffusionModelField,
    samplerField,
    widthField,
    heightField,
    numInferenceStepsField,
    guidanceScaleSlider,
    promptStrengthSlider,
    outputFormatField,
    negativePromptField,
    streamImageProgressField,
    useFaceCorrectionField,
    useUpscalingField,
    showOnlyFilteredImageField,
    upscaleModelField,
    previewImageField,
    modifierCardSizeSlider
];

function getSetting(element) {
    if (element.type == "checkbox") {
        return element.checked;
    }
    return element.value;
}
function setSetting(element, value) {
    if (getSetting(element) == value) {
        return; // no setting necessary
    }
    if (element.type == "checkbox") {
        element.checked = value;
    }
    else {
        element.value = value;
    }
    element.dispatchEvent(new Event("input"));
    element.dispatchEvent(new Event("change"));
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        values: SETTINGS_VALUES,
        should_save: SETTINGS_SHOULD_SAVE_MAP
    }));
}

var CURRENTLY_LOADING_SETTINGS = false;
function loadSettings() {
    if (!autoSaveSettingsField.checked) {
        return;
    }
    var saved_settings = JSON.parse(localStorage.getItem(SETTINGS_KEY));
    if (saved_settings) {
        var values = saved_settings.values;
        var should_save = saved_settings.should_save;
        CURRENTLY_LOADING_SETTINGS = true;
        SETTINGS_TO_SAVE.forEach(element => {
            if (element.id in values) {
                SETTINGS_SHOULD_SAVE_MAP[element.id] = should_save[element.id];
                SETTINGS_VALUES[element.id] = values[element.id];
                if (SETTINGS_SHOULD_SAVE_MAP[element.id]) {
                    setSetting(element, SETTINGS_VALUES[element.id]);
                }
            }
        });
        CURRENTLY_LOADING_SETTINGS = false;
    }
    else {
        saveSettings();
    }
}

restoreDefaultSettingsBtn.addEventListener('click', loadDefaultSettings);
function loadDefaultSettings() {
    CURRENTLY_LOADING_SETTINGS = true;
    SETTINGS_TO_SAVE.forEach(element => {
        SETTINGS_VALUES[element.id] = SETTINGS_DEFAULTS[element.id];
        setSetting(element, SETTINGS_VALUES[element.id]);
    });
    CURRENTLY_LOADING_SETTINGS = false;
    saveSettings();
}

function settingChangeHandler(event) {
    if (!CURRENTLY_LOADING_SETTINGS) {
        var element = event.target;
        var value = getSetting(element);
        if (value != SETTINGS_VALUES[element.id]) {
            SETTINGS_VALUES[element.id] = value;
            saveSettings();
        }
    }
}
async function initSettings() {
    SETTINGS_TO_SAVE.forEach(element => {
        SETTINGS_SHOULD_SAVE_MAP[element.id] = true;
        SETTINGS_DEFAULTS[element.id] = getSetting(element);
        SETTINGS_VALUES[element.id] = getSetting(element);
        element.addEventListener("input", settingChangeHandler);
        element.addEventListener("change", settingChangeHandler);
    });
    loadSettings();
    fillSaveSettingsConfigTable();
}

function fillSaveSettingsConfigTable() {
    SETTINGS_TO_SAVE.forEach(element => {
        var caption = element.id;
        var label = document.querySelector(`label[for='${element.id}']`);
        if (label) {
            caption = label.innerText;
            var truncate_length = 25;
            if (caption.length > truncate_length) {
                caption = caption.substring(0, truncate_length - 3) + "...";
            }
        }
        var default_value = SETTINGS_DEFAULTS[element.id];
        var checkbox_id = `shouldsave_${element.id}`;
        var is_checked = SETTINGS_SHOULD_SAVE_MAP[element.id] ? "checked" : "";
        var newrow = `<tr><td><label for="${checkbox_id}">${caption}</label></td><td><input id="${checkbox_id}" name="${checkbox_id}" ${is_checked} type="checkbox" ></td><td><small>(${default_value})</small></td></tr>`;
        saveSettingsConfigTable.insertAdjacentHTML("beforeend", newrow);
        var checkbox = document.getElementById(checkbox_id)
        checkbox.addEventListener("input", event => {
            SETTINGS_SHOULD_SAVE_MAP[element.id] = checkbox.checked;
            saveSettings();
        });
    });
}

saveSettingsConfigCloseBtn.addEventListener('click', () => {
    saveSettingsConfigOverlay.style.display = 'none';
});
configureSettingsSaveBtn.addEventListener('click', () => {
    saveSettingsConfigOverlay.style.display = 'block';
});
saveSettingsConfigOverlay.addEventListener('click', (event) => {
    if (event.target.id == saveSettingsConfigOverlay.id) {
        saveSettingsConfigOverlay.style.display = 'none';
    }
});