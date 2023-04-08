/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
var ParameterType = {
    checkbox: "checkbox",
    select: "select",
    select_multiple: "select_multiple",
    slider: "slider",
    custom: "custom",
};

/**
 * JSDoc style
 * @typedef {object} Parameter
 * @property {string} id
 * @property {keyof ParameterType} type
 * @property {string | (parameter: Parameter) => (HTMLElement | string)} label
 * @property {string | (parameter: Parameter) => (HTMLElement | string) | undefined} note
 * @property {(parameter: Parameter) => (HTMLElement | string) | undefined} render
 * @property {string | undefined} icon
 * @property {number|boolean|string} default
 * @property {boolean?} saveInAppConfig
 */


/** @type {Array.<Parameter>} */
var PARAMETERS = [
    {
        id: "theme",
        type: ParameterType.select,
        label: "Theme",
        default: "theme-default",
        note: "customize the look and feel of the ui",
        options: [ // Note: options expanded dynamically
            {
                value: "theme-default",
                label: "Default"
            }
        ],
        icon: "fa-palette"
    },
    {
        id: "save_to_disk",
        type: ParameterType.checkbox,
        label: "Auto-Save Images",
        note: "automatically saves images to the specified location",
        icon: "fa-download",
        default: false,
    },
    {
        id: "diskPath",
        type: ParameterType.custom,
        label: "Save Location",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="30" disabled>`
        }
    },
    {
        id: "metadata_output_format",
        type: ParameterType.select,
        label: "Metadata format",
        note: "will be saved to disk in this format",
        default: "txt",
        options: [
            {
                value: "none",
                label: "none"
            },
            {
                value: "txt",
                label: "txt"
            },
            {
                value: "json",
                label: "json"
            },
            {
                value: "embed",
                label: "embed"
            },
            {
                value: "embed,txt",
                label: "embed & txt",
            },
            {
                value: "embed,json",
                label: "embed & json",
            },
        ],
    },
    {
        id: "block_nsfw",
        type: ParameterType.checkbox,
        label: "Block NSFW images",
        note: "blurs out NSFW images",
        icon: "fa-land-mine-on",
        default: false,
    },
    {
        id: "sound_toggle",
        type: ParameterType.checkbox,
        label: "Enable Sound",
        note: "plays a sound on task completion",
        icon: "fa-volume-low",
        default: true,
    },
    {
        id: "process_order_toggle",
        type: ParameterType.checkbox,
        label: "Process newest jobs first",
        note: "reverse the normal processing order",
        icon: "fa-arrow-down-short-wide",
        default: false,
    },
    {
        id: "ui_open_browser_on_start",
        type: ParameterType.checkbox,
        label: "Open browser on startup",
        note: "starts the default browser on startup",
        icon: "fa-window-restore",
        default: true,
        saveInAppConfig: true,
    },
    {
        id: "vram_usage_level",
        type: ParameterType.select,
        label: "GPU Memory Usage",
        note: "Faster performance requires more GPU memory (VRAM)<br/><br/>" +
              "<b>Balanced:</b> nearly as fast as High, much lower VRAM usage<br/>" +
              "<b>High:</b> fastest, maximum GPU memory usage</br>" +
              "<b>Low:</b> slowest, recommended for GPUs with 3 to 4 GB memory",
        icon: "fa-forward",
        default: "balanced",
        options: [
            {value: "balanced", label: "Balanced"},
            {value: "high", label: "High"},
            {value: "low", label: "Low"}
        ],
    },
    {
        id: "use_cpu",
        type: ParameterType.checkbox,
        label: "Use CPU (not GPU)",
        note: "warning: this will be *very* slow",
        icon: "fa-microchip",
        default: false,
    },
    {
        id: "auto_pick_gpus",
        type: ParameterType.checkbox,
        label: "Automatically pick the GPUs (experimental)",
        default: false,
    },
    {
        id: "use_gpus",
        type: ParameterType.select_multiple,
        label: "GPUs to use (experimental)",
        note: "to process in parallel",
        default: false,
    },
    {
        id: "auto_save_settings",
        type: ParameterType.checkbox,
        label: "Auto-Save Settings",
        note: "restores settings on browser load",
        icon: "fa-gear",
        default: true,
    },
    {
        id: "confirm_dangerous_actions",
        type: ParameterType.checkbox,
        label: "Confirm dangerous actions",
        note: "Actions that might lead to data loss must either be clicked with the shift key pressed, or confirmed in an 'Are you sure?' dialog",
        icon: "fa-check-double",
        default: true,
    },
    {
        id: "listen_to_network",
        type: ParameterType.checkbox,
        label: "Make Stable Diffusion available on your network",
        note: "Other devices on your network can access this web page",
        icon: "fa-network-wired",
        default: true,
        saveInAppConfig: true,
    },
    {
        id: "listen_port",
        type: ParameterType.custom,
        label: "Network port",
        note: "Port that this server listens to. The '9000' part in 'http://localhost:9000'",
        icon: "fa-anchor",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="6" value="9000" onkeypress="preventNonNumericalInput(event)">`
        },
        saveInAppConfig: true,
    },
    {
        id: "use_beta_channel",
        type: ParameterType.checkbox,
        label: "Beta channel",
        note: "Get the latest features immediately (but could be less stable). Please restart the program after changing this.",
        icon: "fa-fire",
        default: false,
    },
    {
        id: "test_diffusers",
        type: ParameterType.checkbox,
        label: "Test Diffusers",
        note: "<b>Experimental! Can have bugs!</b> Use upcoming features (like LoRA) in our new engine. Please press Save, then restart the program after changing this.",
        icon: "fa-bolt",
        default: false,
        saveInAppConfig: true,
    },
];

function getParameterSettingsEntry(id) {
    let parameter = PARAMETERS.filter(p => p.id === id)
    if (parameter.length === 0) {
        return
    }
    return parameter[0].settingsEntry
}

function sliderUpdate(event) {
    if (event.srcElement.id.endsWith('-input')) {
        let slider = document.getElementById(event.srcElement.id.slice(0,-6))
        slider.value = event.srcElement.value
        slider.dispatchEvent(new Event("change"))
    } else {
        let field = document.getElementById(event.srcElement.id+'-input')
        field.value = event.srcElement.value
        field.dispatchEvent(new Event("change"))
    }
}

/**
 * @param {Parameter} parameter 
 * @returns {string | HTMLElement}
 */
function getParameterElement(parameter) {
    switch (parameter.type) {
        case ParameterType.checkbox:
            var is_checked = parameter.default ? " checked" : "";
            return `<input id="${parameter.id}" name="${parameter.id}"${is_checked} type="checkbox">`
        case ParameterType.select:
        case ParameterType.select_multiple:
            var options = (parameter.options || []).map(option => `<option value="${option.value}">${option.label}</option>`).join("")
            var multiple = (parameter.type == ParameterType.select_multiple ? 'multiple' : '')
            return `<select id="${parameter.id}" name="${parameter.id}" ${multiple}>${options}</select>`
        case ParameterType.slider:
            return `<input id="${parameter.id}" name="${parameter.id}" class="editor-slider" type="range" value="${parameter.default}" min="${parameter.slider_min}" max="${parameter.slider_max}" oninput="sliderUpdate(event)"> <input id="${parameter.id}-input" name="${parameter.id}-input" size="4" value="${parameter.default}" pattern="^[0-9\.]+$" onkeypress="preventNonNumericalInput(event)" oninput="sliderUpdate(event)">&nbsp;${parameter.slider_unit}`
        case ParameterType.custom:
            return parameter.render(parameter)
        default:
            console.error(`Invalid type ${parameter.type} for parameter ${parameter.id}`);
            return "ERROR: Invalid Type"
    }
}

let parametersTable = document.querySelector("#system-settings .parameters-table")
/**
 * fill in the system settings popup table
 * @param {Array<Parameter> | undefined} parameters
 * */
function initParameters(parameters) {
    parameters.forEach(parameter => {
        const element = getParameterElement(parameter)
        const elementWrapper = createElement('div')
        if (element instanceof Node) {
            elementWrapper.appendChild(element)
        } else {
            elementWrapper.innerHTML = element
        }

        const note = typeof parameter.note === 'function' ? parameter.note(parameter) : parameter.note
        const noteElements = []
        if (note) {
            const noteElement = createElement('small')
            if (note instanceof Node) {
                noteElement.appendChild(note)
            } else {
                noteElement.innerHTML = note || ''
            }
            noteElements.push(noteElement)
        }

        const icon = parameter.icon ? [createElement('i', undefined, ['fa', parameter.icon])] : []

        const label = typeof parameter.label === 'function' ? parameter.label(parameter) : parameter.label
        const labelElement = createElement('label', { for: parameter.id })
        if (label instanceof Node) {
            labelElement.appendChild(label)
        } else {
            labelElement.innerHTML = label
        }

        const newrow = createElement(
            'div',
            { 'data-setting-id': parameter.id, 'data-save-in-app-config': parameter.saveInAppConfig },
            undefined,
            [
                createElement('div', undefined, undefined, icon),
                createElement('div', undefined, undefined, [labelElement, ...noteElements]),
                elementWrapper,
            ]
        )
        parametersTable.appendChild(newrow)
        parameter.settingsEntry = newrow
    })
}

initParameters(PARAMETERS)

// listen to parameters from plugins
PARAMETERS.addEventListener('push', (...items) => {
    initParameters(items)
    
    if (items.find(item => item.saveInAppConfig)) {
        console.log('Reloading app config for new parameters', items.map(p => p.id))
        getAppConfig()
    }
})

let vramUsageLevelField = document.querySelector('#vram_usage_level')
let useCPUField = document.querySelector('#use_cpu')
let autoPickGPUsField = document.querySelector('#auto_pick_gpus')
let useGPUsField = document.querySelector('#use_gpus')
let saveToDiskField = document.querySelector('#save_to_disk')
let diskPathField = document.querySelector('#diskPath')
let metadataOutputFormatField = document.querySelector('#metadata_output_format')
let listenToNetworkField = document.querySelector("#listen_to_network")
let listenPortField = document.querySelector("#listen_port")
let useBetaChannelField = document.querySelector("#use_beta_channel")
let uiOpenBrowserOnStartField = document.querySelector("#ui_open_browser_on_start")
let confirmDangerousActionsField = document.querySelector("#confirm_dangerous_actions")
let testDiffusers = document.querySelector("#test_diffusers")

let saveSettingsBtn = document.querySelector('#save-system-settings-btn')


async function changeAppConfig(configDelta) {
    try {
        let res = await fetch('/app_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configDelta)
        })
        res = await res.json()

        console.log('set config status response', res)
    } catch (e) {
        console.log('set config status error', e)
    }
}

async function getAppConfig() {
    try {
        let res = await fetch('/get/app_config')
        const config = await res.json()

        applySettingsFromConfig(config)

        // custom overrides
        if (config.update_branch === 'beta') {
            useBetaChannelField.checked = true
            document.querySelector("#updateBranchLabel").innerText = "(beta)"
        } else {
            getParameterSettingsEntry("test_diffusers").style.display = "none"
        }
        if (config.ui && config.ui.open_browser_on_start === false) {
            uiOpenBrowserOnStartField.checked = false
        }
        if (config.net && config.net.listen_to_network === false) {
            listenToNetworkField.checked = false
        }
        if (config.net && config.net.listen_port !== undefined) {
            listenPortField.value = config.net.listen_port
        }
        if (config.test_diffusers === undefined || config.update_branch === 'main') {
            testDiffusers.checked = false
            document.querySelector("#lora_model_container").style.display = 'none'
            document.querySelector("#lora_alpha_container").style.display = 'none'
        } else {
            testDiffusers.checked = config.test_diffusers && config.update_branch !== 'main'
            document.querySelector("#lora_model_container").style.display = (testDiffusers.checked ? '' : 'none')
            document.querySelector("#lora_alpha_container").style.display = (testDiffusers.checked && loraModelField.value !== "" ? '' : 'none')
        }

        console.log('get config status response', config)

        return config
    } catch (e) {
        console.log('get config status error', e)

        return {}
    }
}

function applySettingsFromConfig(config) {
    Array.from(parametersTable.children).forEach(parameterRow => {
        if (parameterRow.dataset.settingId in config && parameterRow.dataset.saveInAppConfig === 'true') {
            const configValue = config[parameterRow.dataset.settingId]
            const parameterElement = document.getElementById(parameterRow.dataset.settingId) ||
                parameterRow.querySelector('input') || parameterRow.querySelector('select')

            switch (parameterElement?.tagName) {
                case 'INPUT':
                    if (parameterElement.type === 'checkbox') {
                        parameterElement.checked = configValue
                    } else {
                        parameterElement.value = configValue
                    }
                    parameterElement.dispatchEvent(new Event('change'))
                    break
                case 'SELECT':
                    if (Array.isArray(configValue)) {
                        Array.from(parameterElement.options).forEach(option => {
                            if (configValue.includes(option.value || option.text)) {
                                option.selected = true
                            }
                        })
                    } else {
                        parameterElement.value = configValue
                    }
                    parameterElement.dispatchEvent(new Event('change'))
                    break
            }
        }
    })
}

saveToDiskField.addEventListener('change', function(e) {
    diskPathField.disabled = !this.checked
    metadataOutputFormatField.disabled = !this.checked
})

function getCurrentRenderDeviceSelection() {
    let selectedGPUs = $('#use_gpus').val()

    if (useCPUField.checked && !autoPickGPUsField.checked) {
        return 'cpu'
    }
    if (autoPickGPUsField.checked || selectedGPUs.length == 0) {
        return 'auto'
    }

    return selectedGPUs.join(',')
}

useCPUField.addEventListener('click', function() {
    let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
    let autoPickGPUSettingEntry = getParameterSettingsEntry('auto_pick_gpus')
    if (this.checked) {
        gpuSettingEntry.style.display = 'none'
        autoPickGPUSettingEntry.style.display = 'none'
        autoPickGPUsField.setAttribute('data-old-value', autoPickGPUsField.checked)
        autoPickGPUsField.checked = false
    } else if (useGPUsField.options.length >= MIN_GPUS_TO_SHOW_SELECTION) {
        gpuSettingEntry.style.display = ''
        autoPickGPUSettingEntry.style.display = ''
        let oldVal = autoPickGPUsField.getAttribute('data-old-value')
        if (oldVal === null || oldVal === undefined) { // the UI started with CPU selected by default
            autoPickGPUsField.checked = true
        } else {
            autoPickGPUsField.checked = (oldVal === 'true')
        }
        gpuSettingEntry.style.display = (autoPickGPUsField.checked ? 'none' : '')
    }
})

useGPUsField.addEventListener('click', function() {
    let selectedGPUs = $('#use_gpus').val()
    autoPickGPUsField.checked = (selectedGPUs.length === 0)
})

autoPickGPUsField.addEventListener('click', function() {
    if (this.checked) {
        $('#use_gpus').val([])
    }

    let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
    gpuSettingEntry.style.display = (this.checked ? 'none' : '')
})

async function setDiskPath(defaultDiskPath, force=false) {
    var diskPath = getSetting("diskPath")
    if (force || diskPath == '' || diskPath == undefined || diskPath == "undefined") {
        setSetting("diskPath", defaultDiskPath)
    }
}

function setDeviceInfo(devices) {
    let cpu = devices.all.cpu.name
    let allGPUs = Object.keys(devices.all).filter(d => d != 'cpu')
    let activeGPUs = Object.keys(devices.active)

    function ID_TO_TEXT(d) {
        let info = devices.all[d]
        if ("mem_free" in info && "mem_total" in info) {
            return `${info.name} <small>(${d}) (${info.mem_free.toFixed(1)}Gb free / ${info.mem_total.toFixed(1)} Gb total)</small>`
        } else {
            return `${info.name} <small>(${d}) (no memory info)</small>`
        }
    }

    allGPUs = allGPUs.map(ID_TO_TEXT)
    activeGPUs = activeGPUs.map(ID_TO_TEXT)

    let systemInfoEl = document.querySelector('#system-info')
    systemInfoEl.querySelector('#system-info-cpu').innerText = cpu
    systemInfoEl.querySelector('#system-info-gpus-all').innerHTML = allGPUs.join('</br>')
    systemInfoEl.querySelector('#system-info-rendering-devices').innerHTML = activeGPUs.join('</br>')
}

function setHostInfo(hosts) {
    let port = listenPortField.value
    hosts = hosts.map(addr => `http://${addr}:${port}/`).map(url => `<div><a href="${url}">${url}</a></div>`)
    document.querySelector('#system-info-server-hosts').innerHTML = hosts.join('')
}

async function getSystemInfo() {
    try {
        const res = await SD.getSystemInfo()
        let devices = res['devices']

        let allDeviceIds = Object.keys(devices['all']).filter(d => d !== 'cpu')
        let activeDeviceIds = Object.keys(devices['active']).filter(d => d !== 'cpu')

        if (activeDeviceIds.length === 0) {
            useCPUField.checked = true
        }

        if (allDeviceIds.length < MIN_GPUS_TO_SHOW_SELECTION || useCPUField.checked) {
            let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
            gpuSettingEntry.style.display = 'none'
            let autoPickGPUSettingEntry = getParameterSettingsEntry('auto_pick_gpus')
            autoPickGPUSettingEntry.style.display = 'none'
        }

        if (allDeviceIds.length === 0) {
            useCPUField.checked = true
            useCPUField.disabled = true // no compatible GPUs, so make the CPU mandatory
        }

        autoPickGPUsField.checked = (devices['config'] === 'auto')

        useGPUsField.innerHTML = ''
        allDeviceIds.forEach(device => {
            let deviceName = devices['all'][device]['name']
            let deviceOption = `<option value="${device}">${deviceName} (${device})</option>`
            useGPUsField.insertAdjacentHTML('beforeend', deviceOption)
        })

        if (autoPickGPUsField.checked) {
            let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
            gpuSettingEntry.style.display = 'none'
        } else {
            $('#use_gpus').val(activeDeviceIds)
        }

        setDeviceInfo(devices)
        setHostInfo(res['hosts'])
        let force = false
        if (res['enforce_output_dir'] !== undefined) {
            force = res['enforce_output_dir']
            if (force == true) {
               saveToDiskField.checked = true
               metadataOutputFormatField.disabled = false
            }
            saveToDiskField.disabled = force
            diskPathField.disabled = force
        }
        setDiskPath(res['default_output_dir'], force)
    } catch (e) {
        console.log('error fetching devices', e)
    }
}

saveSettingsBtn.addEventListener('click', function() {
    if (listenPortField.value == '') {
        alert('The network port field must not be empty.')
        return
    }
    if (listenPortField.value < 1 || listenPortField.value > 65535) {
        alert('The network port must be a number from 1 to 65535')
        return
    }
    const updateBranch = (useBetaChannelField.checked ? 'beta' : 'main')

    const updateAppConfigRequest = {
        'render_devices': getCurrentRenderDeviceSelection(),
        'update_branch': updateBranch,
    }

    Array.from(parametersTable.children).forEach(parameterRow => {
        if (parameterRow.dataset.saveInAppConfig === 'true') {
            const parameterElement = document.getElementById(parameterRow.dataset.settingId) ||
                parameterRow.querySelector('input') || parameterRow.querySelector('select')

            switch (parameterElement?.tagName) {
                case 'INPUT':
                    if (parameterElement.type === 'checkbox') {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.checked
                    } else {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.value
                    }
                    break
                case 'SELECT':
                    if (parameterElement.multiple) {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = Array.from(parameterElement.options)
                            .filter(option => option.selected)
                            .map(option => option.value || option.text)
                    } else {
                        updateAppConfigRequest[parameterRow.dataset.settingId] = parameterElement.value
                    }
                    break
                default:
                    console.error(`Setting parameter ${parameterRow.dataset.settingId} couldn't be saved to app.config - element #${parameter.id} is a <${parameterElement?.tagName} /> instead of a <input /> or a <select />!`)
                    break
            }
        }
    })

    const savePromise = changeAppConfig(updateAppConfigRequest)
    saveSettingsBtn.classList.add('active')
    Promise.all([savePromise, asyncDelay(300)]).then(() => saveSettingsBtn.classList.remove('active'))
})
