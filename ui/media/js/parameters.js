/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
 var ParameterType = {
    checkbox: "checkbox",
    select: "select",
    select_multiple: "select_multiple",
    custom: "custom",
};

/**
 * JSDoc style
 * @typedef {object} Parameter
 * @property {string} id
 * @property {ParameterType} type
 * @property {string} label
 * @property {?string} note
 * @property {number|boolean|string} default
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
                value: "txt",
                label: "txt"
            },
            {
                value: "json",
                label: "json"
            }
        ],
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
    },
    {
        id: "listen_port",
        type: ParameterType.custom,
        label: "Network port",
        note: "Port that this server listens to. The '9000' part in 'http://localhost:9000'",
        icon: "fa-anchor",
        render: (parameter) => {
            return `<input id="${parameter.id}" name="${parameter.id}" size="6" value="9000" onkeypress="preventNonNumericalInput(event)">`
        }
    },
    {
        id: "use_beta_channel",
        type: ParameterType.checkbox,
        label: "Beta channel",
        note: "Get the latest features immediately (but could be less stable). Please restart the program after changing this.",
        icon: "fa-fire",
        default: false,
    },
];

function getParameterSettingsEntry(id) {
    let parameter = PARAMETERS.filter(p => p.id === id)
    if (parameter.length === 0) {
        return
    }
    return parameter[0].settingsEntry
}

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
        case ParameterType.custom:
            return parameter.render(parameter)
        default:
            console.error(`Invalid type for parameter ${parameter.id}`);
            return "ERROR: Invalid Type"
    }
}

let parametersTable = document.querySelector("#system-settings .parameters-table")
/* fill in the system settings popup table */
function initParameters() {
    PARAMETERS.forEach(parameter => {
        var element = getParameterElement(parameter)
        var note = parameter.note ? `<small>${parameter.note}</small>` : "";
        var icon = parameter.icon ? `<i class="fa ${parameter.icon}"></i>` : "";
        var newrow = document.createElement('div')
        newrow.innerHTML = `
            <div>${icon}</div>
            <div><label for="${parameter.id}">${parameter.label}</label>${note}</div>
            <div>${element}</div>`
        parametersTable.appendChild(newrow)
        parameter.settingsEntry = newrow
    })
}

initParameters()

let vramUsageLevelField = document.querySelector('#vram_usage_level')
let useCPUField = document.querySelector('#use_cpu')
let autoPickGPUsField = document.querySelector('#auto_pick_gpus')
let useGPUsField = document.querySelector('#use_gpus')
let saveToDiskField = document.querySelector('#save_to_disk')
let diskPathField = document.querySelector('#diskPath')
let listenToNetworkField = document.querySelector("#listen_to_network")
let listenPortField = document.querySelector("#listen_port")
let useBetaChannelField = document.querySelector("#use_beta_channel")
let uiOpenBrowserOnStartField = document.querySelector("#ui_open_browser_on_start")
let confirmDangerousActionsField = document.querySelector("#confirm_dangerous_actions")

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

        if (config.update_branch === 'beta') {
            useBetaChannelField.checked = true
            document.querySelector("#updateBranchLabel").innerText = "(beta)"
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

        console.log('get config status response', config)
    } catch (e) {
        console.log('get config status error', e)
    }
}

saveToDiskField.addEventListener('change', function(e) {
    diskPathField.disabled = !this.checked
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

async function setDiskPath(defaultDiskPath) {
    var diskPath = getSetting("diskPath")
    if (diskPath == '' || diskPath == undefined || diskPath == "undefined") {
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
        setDiskPath(res['default_output_dir'])
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
    let updateBranch = (useBetaChannelField.checked ? 'beta' : 'main')
    changeAppConfig({
        'render_devices': getCurrentRenderDeviceSelection(),
        'update_branch': updateBranch,
        'ui_open_browser_on_start': uiOpenBrowserOnStartField.checked,
        'listen_to_network': listenToNetworkField.checked,
        'listen_port': listenPortField.value
    })
    saveSettingsBtn.classList.add('active')
    asyncDelay(300).then(() => saveSettingsBtn.classList.remove('active'))
})
