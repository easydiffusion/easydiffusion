"use strict" // Opt in to a restricted variant of JavaScript

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
                return
            }
            randomSeedField.checked = false
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
            useFaceCorrectionField.checked = Boolean(use_face_correction)
        },
        readUI: () => useFaceCorrectionField.checked,
        parse: (val) => val
    },
    use_upscale: { name: 'Use Upscaling',
        setUI: (use_upscale) => {
            useUpscalingField.checked = Boolean(use_upscale)
            upscaleModelField.value = use_upscale
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

            const pathIdx = use_stable_diffusion_model.lastIndexOf('/')
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
const lineEndRe = /(?:\r\n|\r|\n)+/igm
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
        if (str.startsWith(name + ':')) {
            // Backend format, faster
            lineEndRe.lastIndex = 0
            const endMatch = lineEndRe.exec(str)
            if (endMatch) {
                val = str.slice(name.length + 1, endMatch.index)
                str = str.slice(endMatch.index + endMatch[0].length)
            } else {
                val = str.slice(name.length + 1)
                str = ""
            }
        } else {
            // User formatted, use regex to get all cases, but slower.
            const reName = new RegExp(`${name}\\s*:\\s*(.*)(?:\\r\\n|\\r|\\n)*`, 'igm')
            const match = reName.exec(str);
            if (match) {
                str = str.slice(0, match.index) + str.slice(match.index + match[0].length)
                val = match[1]
            }
        }
        if (val) {
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
    // Prevent default behavior (Prevent file/content from being opened)
    ev.preventDefault()

    if (ev?.dataTransfer?.items) { // Use DataTransferItemList interface
        Array.from(ev.dataTransfer.items).forEach(function(item, i) {
            if (item.kind === 'file') {
                const file = item.getAsFile()
                readFile(file, i)
            }
        })
    } else if (ev?.dataTransfer?.files) { // Use DataTransfer interface
        Array.from(ev.dataTransfer.files).forEach(readFile)
    }
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

// Adds a copy icon if the browser grants permission to write to clipboard.
function checkWriteToClipboardPermission (result) {
    if (result.state == "granted" || result.state == "prompt") {
        const style = document.createElement('style');
        style.textContent = `
        #copy-image-settings {
            cursor: pointer;
            padding: 8px;
            opacity: 1;
            transition: opacity 0.5;
        }
        .collapsible:not(.active) #copy-image-settings {
            display: none;
        }
        #copy-image-settings.hidden {
            opacity: 0;
            pointer-events: none;
        }`;
        document.head.append(style);
        const resetSettings = document.getElementById('reset-image-settings')
        const copyIcon = document.createElement('i')
        copyIcon.id = 'copy-image-settings'
        copyIcon.className = 'fa-solid fa-clipboard'
        copyIcon.innerHTML = `<span class="simple-tooltip right">Copy Image Settings</span>`
        copyIcon.addEventListener('click', (event) => {
            event.stopPropagation()
            navigator.clipboard.writeText(JSON.stringify(readUI(), undefined, 4))
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
