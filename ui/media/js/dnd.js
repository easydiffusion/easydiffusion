"use strict" // Opt in to a restricted variant of JavaScript

const EXT_REGEX = /(?:\.([^.]+))?$/
const TEXT_EXTENSIONS = ['txt', 'json']
const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'tga']

function parseBoolean(stringValue) {
    if (typeof stringValue === 'boolean') {
        return stringValue
    }
    if (typeof stringValue === 'number') {
        return stringValue !== 0
    }
    if (typeof stringValue !== 'string') {
        return false
    }
    switch(stringValue?.toLowerCase()?.trim()) {
        case "true":
        case "yes":
        case "on":
        case "1":
          return true;

        case "false":
        case "no":
        case "off":
        case "0":
        case null:
        case undefined:
          return false;
    }
    try {
        return Boolean(JSON.parse(stringValue));
    } catch {
        return Boolean(stringValue)
    }
}

const TASK_MAPPING = {
    prompt: { name: 'Prompt',
        setUI: (prompt) => {
            promptField.value = prompt
        },
        readUI: () => promptField.value,
        parse: (val) => val
    },
    negative_prompt: { name: 'Negative Prompt',
        setUI: (negative_prompt) => {
            negativePromptField.value = negative_prompt
        },
        readUI: () => negativePromptField.value,
        parse: (val) => val
    },
    width: { name: 'Width',
        setUI: (width) => {
            const oldVal = widthField.value
            widthField.value = width
            if (!widthField.value) {
                widthField.value = oldVal
            }
        },
        readUI: () => parseInt(widthField.value),
        parse: (val) => parseInt(val)
    },
    height: { name: 'Height',
        setUI: (height) => {
            const oldVal = heightField.value
            heightField.value = height
            if (!heightField.value) {
                heightField.value = oldVal
            }
        },
        readUI: () => parseInt(heightField.value),
        parse: (val) => parseInt(val)
    },
    seed: { name: 'Seed',
        setUI: (seed) => {
            if (!seed) {
                randomSeedField.checked = true
                seedField.disabled = true
                return
            }
            randomSeedField.checked = false
            seedField.disabled = false
            seedField.value = seed
        },
        readUI: () => (randomSeedField.checked ? Math.floor(Math.random() * 10000000) : parseInt(seedField.value)),
        parse: (val) => parseInt(val)
    },
    num_inference_steps: { name: 'Steps',
        setUI: (num_inference_steps) => {
            numInferenceStepsField.value = num_inference_steps
        },
        readUI: () => parseInt(numInferenceStepsField.value),
        parse: (val) => parseInt(val)
    },
    guidance_scale: { name: 'Guidance Scale',
        setUI: (guidance_scale) => {
            guidanceScaleField.value = guidance_scale
            updateGuidanceScaleSlider()
        },
        readUI: () => parseFloat(guidanceScaleField.value),
        parse: (val) => parseFloat(val)
    },
    prompt_strength: { name: 'Prompt Strength',
        setUI: (prompt_strength) => {
            promptStrengthField.value = prompt_strength
            updatePromptStrengthSlider()
        },
        readUI: () => parseFloat(promptStrengthField.value),
        parse: (val) => parseFloat(val)
    },

    init_image:  { name: 'Initial Image',
        setUI: (init_image) => {
            initImagePreview.src = init_image
        },
        readUI: () => initImagePreview.src,
        parse: (val) => val
    },
    mask:  { name: 'Mask',
        setUI: (mask) => {
            inpaintingEditor.setImg(mask)
            maskSetting.checked = Boolean(mask)
        },
        readUI: () => (maskSetting.checked ? inpaintingEditor.getImg() : undefined),
        parse: (val) => val
    },

    use_face_correction: { name: 'Use Face Correction',
        setUI: (use_face_correction) => {
            useFaceCorrectionField.checked = parseBoolean(use_face_correction)
        },
        readUI: () => useFaceCorrectionField.checked,
        parse: (val) => parseBoolean(val)
    },
    use_upscale: { name: 'Use Upscaling',
        setUI: (use_upscale) => {
            const oldVal = upscaleModelField.value
            upscaleModelField.value = use_upscale
            if (upscaleModelField.value) { // Is a valid value for the field.
                useUpscalingField.checked = true
                upscaleModelField.disabled = false
            } else { // Not a valid value, restore the old value and disable the filter.
                upscaleModelField.disabled = true
                upscaleModelField.value = oldVal
                useUpscalingField.checked = false
            }
        },
        readUI: () => (useUpscalingField.checked ? upscaleModelField.value : undefined),
        parse: (val) => val
    },
    sampler: { name: 'Sampler',
        setUI: (sampler) => {
            samplerField.value = sampler
        },
        readUI: () => samplerField.value,
        parse: (val) => val
    },
    use_stable_diffusion_model: { name: 'Stable Diffusion model',
        setUI: (use_stable_diffusion_model) => {
            const oldVal = stableDiffusionModelField.value

            let pathIdx = use_stable_diffusion_model.lastIndexOf('/') // Linux, Mac paths
            if (pathIdx < 0) {
                pathIdx = use_stable_diffusion_model.lastIndexOf('\\') // Windows paths.
            }
            if (pathIdx >= 0) {
                use_stable_diffusion_model = use_stable_diffusion_model.slice(pathIdx + 1)
            }
            const modelExt = '.ckpt'
            if (use_stable_diffusion_model.endsWith(modelExt)) {
                use_stable_diffusion_model = use_stable_diffusion_model.slice(0, use_stable_diffusion_model.length - modelExt.length)
            }

            stableDiffusionModelField.value = use_stable_diffusion_model

            if (!stableDiffusionModelField.value) {
                stableDiffusionModelField.value = oldVal
            }
        },
        readUI: () => stableDiffusionModelField.value,
        parse: (val) => val
    },
    use_vae_model: { name: 'VAE model',
        setUI: (use_vae_model) => {
            const oldVal = vaeModelField.value

            if (Boolean(use_vae_model)) {
                let pathIdx = use_vae_model.lastIndexOf('/') // Linux, Mac paths
                if (pathIdx < 0) {
                    pathIdx = use_vae_model.lastIndexOf('\\') // Windows paths.
                }
                if (pathIdx >= 0) {
                    use_vae_model = use_vae_model.slice(pathIdx + 1)
                }
                const modelExt = '.ckpt'
                if (use_vae_model.endsWith(modelExt)) {
                    use_vae_model = use_vae_model.slice(0, use_vae_model.length - modelExt.length)
                }
                use_vae_model = Boolean(use_vae_model) ? use_vae_model : oldVal
            }
            vaeModelField.value = use_vae_model
        },
        readUI: () => vaeModelField.value,
        parse: (val) => val
    },

    numOutputsParallel: { name: 'Parallel Images',
        setUI: (numOutputsParallel) => {
            numOutputsParallelField.value = numOutputsParallel
        },
        readUI: () => parseInt(numOutputsParallelField.value),
        parse: (val) => val
    },

    use_cpu: { name: 'Use CPU',
        setUI: (use_cpu) => {
            useCPUField.checked = use_cpu
        },
        readUI: () => useCPUField.checked,
        parse: (val) => val
    },
    turbo: { name: 'Turbo',
        setUI: (turbo) => {
            turboField.checked = turbo
        },
        readUI: () => turboField.checked,
        parse: (val) => Boolean(val)
    },
    use_full_precision: { name: 'Use Full Precision',
        setUI: (use_full_precision) => {
            useFullPrecisionField.checked = use_full_precision
        },
        readUI: () => useFullPrecisionField.checked,
        parse: (val) => Boolean(val)
    },

    stream_image_progress: { name: 'Stream Image Progress',
        setUI: (stream_image_progress) => {
            streamImageProgressField.checked = (parseInt(numOutputsTotalField.value) > 50 ? false : stream_image_progress)
        },
        readUI: () => streamImageProgressField.checked,
        parse: (val) => Boolean(val)
    },
    show_only_filtered_image: { name: 'Show only the corrected/upscaled image',
        setUI: (show_only_filtered_image) => {
            showOnlyFilteredImageField.checked = show_only_filtered_image
        },
        readUI: () => showOnlyFilteredImageField.checked,
        parse: (val) => Boolean(val)
    },
    output_format: { name: 'Output Format',
        setUI: (output_format) => {
            outputFormatField.value = output_format
        },
        readUI: () => outputFormatField.value,
        parse: (val) => val
    },
    save_to_disk_path: { name: 'Save to disk path',
        setUI: (save_to_disk_path) => {
            saveToDiskField.checked = Boolean(save_to_disk_path)
            diskPathField.value = save_to_disk_path
        },
        readUI: () => diskPathField.value,
        parse: (val) => val
    }
}
function restoreTaskToUI(task) {
    if ('numOutputsTotal' in task) {
        numOutputsTotalField.value = task.numOutputsTotal
    }
    if ('seed' in task) {
        randomSeedField.checked = false
        seedField.value = task.seed
    }
    if (!('reqBody' in task)) {
        return
    }
    for (const key in TASK_MAPPING) {
        if (key in task.reqBody) {
            TASK_MAPPING[key].setUI(task.reqBody[key])
        }
    }
}
function readUI() {
    const reqBody = {}
    for (const key in TASK_MAPPING) {
        reqBody[key] = TASK_MAPPING[key].readUI()
    }
    return {
        'numOutputsTotal': parseInt(numOutputsTotalField.value),
        'seed': TASK_MAPPING['seed'].readUI(),
        'reqBody': reqBody
    }
}

const TASK_TEXT_MAPPING = {
    width: 'Width',
    height: 'Height',
    seed: 'Seed',
    num_inference_steps: 'Steps',
    guidance_scale: 'Guidance Scale',
    prompt_strength: 'Prompt Strength',
    use_face_correction: 'Use Face Correction',
    use_upscale: 'Use Upscaling',
    sampler: 'Sampler',
    negative_prompt: 'Negative Prompt',
    use_stable_diffusion_model: 'Stable Diffusion model'
}
const afterPromptRe = /^\s*Width\s*:\s*\d+\s*(?:\r\n|\r|\n)+\s*Height\s*:\s*\d+\s*(\r\n|\r|\n)+Seed\s*:\s*\d+\s*$/igm
function parseTaskFromText(str) {
    const taskReqBody = {}

    // Prompt
    afterPromptRe.lastIndex = 0
    const match = afterPromptRe.exec(str)
    if (match) {
        let prompt = str.slice(0, match.index)
        str = str.slice(prompt.length)
        taskReqBody.prompt = prompt.trim()
        console.log('Prompt:', taskReqBody.prompt)
    }
    for (const key in TASK_TEXT_MAPPING) {
        const name = TASK_TEXT_MAPPING[key];
        let val = undefined

        const reName = new RegExp(`${name}\\ *:\\ *(.*)(?:\\r\\n|\\r|\\n)*`, 'igm')
        const match = reName.exec(str);
        if (match) {
            str = str.slice(0, match.index) + str.slice(match.index + match[0].length)
            val = match[1]
        }
        if (val !== undefined) {
            taskReqBody[key] = TASK_MAPPING[key].parse(val.trim())
            console.log(TASK_MAPPING[key].name + ':', taskReqBody[key])
            if (!str) {
                break
            }
        }
    }
    if (Object.keys(taskReqBody).length <= 0) {
        return undefined
    }
    const task = { reqBody: taskReqBody }
    if ('seed' in taskReqBody) {
        task.seed = taskReqBody.seed
    }
    return task
}

async function readFile(file, i) {
    const fileContent = (await file.text()).trim()

    // JSON File.
    if (fileContent.startsWith('{') && fileContent.endsWith('}')) {
        try {
            const task = JSON.parse(fileContent)
            restoreTaskToUI(task)
        } catch (e) {
            console.warn(`file[${i}]:${file.name} - File couldn't be parsed.`, e)
        }
        return
    }

    // Normal txt file.
    const task = parseTaskFromText(fileContent)
    if (task) {
        restoreTaskToUI(task)
    } else {
        console.warn(`file[${i}]:${file.name} - File couldn't be parsed.`)
    }
}

function dropHandler(ev) {
    console.log('Content dropped...')
    let items = []

    if (ev?.dataTransfer?.items) { // Use DataTransferItemList interface
        items = Array.from(ev.dataTransfer.items)
        items = items.filter(item => item.kind === 'file')
        items = items.map(item => item.getAsFile())
    } else if (ev?.dataTransfer?.files) { // Use DataTransfer interface
        items = Array.from(ev.dataTransfer.files)
    }

    items.forEach(item => {item.file_ext = EXT_REGEX.exec(item.name.toLowerCase())[1]})

    let text_items = items.filter(item => TEXT_EXTENSIONS.includes(item.file_ext))
    let image_items = items.filter(item => IMAGE_EXTENSIONS.includes(item.file_ext))

    if (image_items.length > 0 && ev.target == initImageSelector) {
        return // let the event bubble up, so that the Init Image filepicker can receive this
    }

    ev.preventDefault() // Prevent default behavior (Prevent file/content from being opened)
    text_items.forEach(readFile)
}
function dragOverHandler(ev) {
    console.log('Content in drop zone')

    // Prevent default behavior (Prevent file/content from being opened)
    ev.preventDefault()

    ev.dataTransfer.dropEffect = "copy"

    let img = new Image()
    img.src = location.host + '/media/images/favicon-32x32.png'
    ev.dataTransfer.setDragImage(img, 16, 16)
}

document.addEventListener("drop", dropHandler)
document.addEventListener("dragover", dragOverHandler)

const TASK_REQ_NO_EXPORT = [
    "use_cpu",
    "turbo",
    "use_full_precision",
    "save_to_disk_path"
]

// Adds a copy icon if the browser grants permission to write to clipboard.
function checkWriteToClipboardPermission (result) {
    if (result.state == "granted" || result.state == "prompt") {
        const resetSettings = document.getElementById('reset-image-settings')
        const copyIcon = document.createElement('i')
        // copyIcon.id = 'copy-image-settings'
        copyIcon.className = 'fa-solid fa-clipboard section-button'
        copyIcon.innerHTML = `<span class="simple-tooltip right">Copy Image Settings</span>`
        copyIcon.addEventListener('click', (event) => {
            event.stopPropagation()
            const uiState = readUI()
            TASK_REQ_NO_EXPORT.forEach((key) => delete uiState.reqBody[key])
            if (uiState.reqBody.init_image && !IMAGE_REGEX.test(uiState.reqBody.init_image)) {
                delete uiState.reqBody.init_image
                delete uiState.reqBody.prompt_strength
            }
            navigator.clipboard.writeText(JSON.stringify(uiState, undefined, 4))
        })
        resetSettings.parentNode.insertBefore(copyIcon, resetSettings)
    }
}
navigator.permissions.query({ name: "clipboard-write" }).then(checkWriteToClipboardPermission, (e) => {
    if (e instanceof TypeError && typeof navigator?.clipboard?.writeText === 'function') {
        // Fix for firefox https://bugzilla.mozilla.org/show_bug.cgi?id=1560373
        checkWriteToClipboardPermission({state:"granted"})
    }
})
