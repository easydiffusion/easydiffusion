"use strict" // Opt in to a restricted variant of JavaScript

const EXT_REGEX = /(?:\.([^.]+))?$/
const TEXT_EXTENSIONS = ["txt", "json"]
const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "tga", "webp"]

function parseBoolean(stringValue) {
    if (typeof stringValue === "boolean") {
        return stringValue
    }
    if (typeof stringValue === "number") {
        return stringValue !== 0
    }
    if (typeof stringValue !== "string") {
        return false
    }
    switch (stringValue?.toLowerCase()?.trim()) {
        case "true":
        case "yes":
        case "on":
        case "1":
            return true

        case "false":
        case "no":
        case "off":
        case "0":
        case "none":
        case null:
        case undefined:
            return false
    }
    try {
        return Boolean(JSON.parse(stringValue))
    } catch {
        return Boolean(stringValue)
    }
}

// keep in sync with `ui/easydiffusion/utils/save_utils.py`
const TASK_MAPPING = {
    prompt: {
        name: "Prompt",
        setUI: (prompt) => {
            promptField.value = prompt
        },
        readUI: () => promptField.value,
        parse: (val) => val,
    },
    negative_prompt: {
        name: "Negative Prompt",
        setUI: (negative_prompt) => {
            negativePromptField.value = negative_prompt
        },
        readUI: () => negativePromptField.value,
        parse: (val) => val,
    },
    active_tags: {
        name: "Image Modifiers",
        setUI: (active_tags) => {
            refreshModifiersState(active_tags)
        },
        readUI: () => activeTags.map((x) => x.name),
        parse: (val) => val,
    },
    inactive_tags: {
        name: "Inactive Image Modifiers",
        setUI: (inactive_tags) => {
            refreshInactiveTags(inactive_tags)
        },
        readUI: () => activeTags.filter((tag) => tag.inactive === true).map((x) => x.name),
        parse: (val) => val,
    },
    width: {
        name: "Width",
        setUI: (width) => {
            const oldVal = widthField.value
            widthField.value = width
            if (!widthField.value) {
                widthField.value = oldVal
            }
            widthField.dispatchEvent(new Event("change"))
        },
        readUI: () => parseInt(widthField.value),
        parse: (val) => parseInt(val),
    },
    height: {
        name: "Height",
        setUI: (height) => {
            const oldVal = heightField.value
            heightField.value = height
            if (!heightField.value) {
                heightField.value = oldVal
            }
            heightField.dispatchEvent(new Event("change"))
        },
        readUI: () => parseInt(heightField.value),
        parse: (val) => parseInt(val),
    },
    seed: {
        name: "Seed",
        setUI: (seed) => {
            if (!seed) {
                randomSeedField.checked = true
                seedField.disabled = true
                seedField.value = 0
                return
            }
            randomSeedField.checked = false
            randomSeedField.dispatchEvent(new Event("change")) // let plugins know that the state of the random seed toggle changed
            seedField.disabled = false
            seedField.value = seed
        },
        readUI: () => parseInt(seedField.value), // just return the value the user is seeing in the UI
        parse: (val) => parseInt(val),
    },
    num_inference_steps: {
        name: "Steps",
        setUI: (num_inference_steps) => {
            numInferenceStepsField.value = num_inference_steps
        },
        readUI: () => parseInt(numInferenceStepsField.value),
        parse: (val) => parseInt(val),
    },
    guidance_scale: {
        name: "Guidance Scale",
        setUI: (guidance_scale) => {
            guidanceScaleField.value = guidance_scale
            updateGuidanceScaleSlider()
        },
        readUI: () => parseFloat(guidanceScaleField.value),
        parse: (val) => parseFloat(val),
    },
    prompt_strength: {
        name: "Prompt Strength",
        setUI: (prompt_strength) => {
            promptStrengthField.value = prompt_strength
            updatePromptStrengthSlider()
        },
        readUI: () => parseFloat(promptStrengthField.value),
        parse: (val) => parseFloat(val),
    },

    init_image: {
        name: "Initial Image",
        setUI: (init_image) => {
            initImagePreview.src = init_image
        },
        readUI: () => initImagePreview.src,
        parse: (val) => val,
    },
    mask: {
        name: "Mask",
        setUI: (mask) => {
            setTimeout(() => {
                // add a delay to insure this happens AFTER the main image loads (which reloads the inpainter)
                imageInpainter.setImg(mask)
            }, 250)
            maskSetting.checked = Boolean(mask)
        },
        readUI: () => (maskSetting.checked ? imageInpainter.getImg() : undefined),
        parse: (val) => val,
    },
    preserve_init_image_color_profile: {
        name: "Preserve Color Profile",
        setUI: (preserve_init_image_color_profile) => {
            applyColorCorrectionField.checked = parseBoolean(preserve_init_image_color_profile)
        },
        readUI: () => applyColorCorrectionField.checked,
        parse: (val) => parseBoolean(val),
    },

    use_face_correction: {
        name: "Use Face Correction",
        setUI: (use_face_correction) => {
            const oldVal = gfpganModelField.value
            console.log("use face correction", use_face_correction)
            if (use_face_correction == null || use_face_correction == "None") {
                gfpganModelField.disabled = true
                useFaceCorrectionField.checked = false
            } else {
                gfpganModelField.value = getModelPath(use_face_correction, [".pth"])
                if (gfpganModelField.value) {
                    // Is a valid value for the field.
                    useFaceCorrectionField.checked = true
                    gfpganModelField.disabled = false
                } else {
                    // Not a valid value, restore the old value and disable the filter.
                    gfpganModelField.disabled = true
                    gfpganModelField.value = oldVal
                    useFaceCorrectionField.checked = false
                }
            }

            //useFaceCorrectionField.checked = parseBoolean(use_face_correction)
        },
        readUI: () => (useFaceCorrectionField.checked ? gfpganModelField.value : undefined),
        parse: (val) => val,
    },
    use_upscale: {
        name: "Use Upscaling",
        setUI: (use_upscale) => {
            const oldVal = upscaleModelField.value
            upscaleModelField.value = getModelPath(use_upscale, [".pth"])
            if (upscaleModelField.value) {
                // Is a valid value for the field.
                useUpscalingField.checked = true
                upscaleModelField.disabled = false
                upscaleAmountField.disabled = false
            } else {
                // Not a valid value, restore the old value and disable the filter.
                upscaleModelField.disabled = true
                upscaleAmountField.disabled = true
                upscaleModelField.value = oldVal
                useUpscalingField.checked = false
            }
        },
        readUI: () => (useUpscalingField.checked ? upscaleModelField.value : undefined),
        parse: (val) => val,
    },
    upscale_amount: {
        name: "Upscale By",
        setUI: (upscale_amount) => {
            upscaleAmountField.value = upscale_amount
        },
        readUI: () => upscaleAmountField.value,
        parse: (val) => val,
    },
    latent_upscaler_steps: {
        name: "Latent Upscaler Steps",
        setUI: (latent_upscaler_steps) => {
            latentUpscalerStepsField.value = latent_upscaler_steps
        },
        readUI: () => latentUpscalerStepsField.value,
        parse: (val) => val,
    },
    sampler_name: {
        name: "Sampler",
        setUI: (sampler_name) => {
            samplerField.value = sampler_name
        },
        readUI: () => samplerField.value,
        parse: (val) => val,
    },
    use_stable_diffusion_model: {
        name: "Stable Diffusion model",
        setUI: (use_stable_diffusion_model) => {
            const oldVal = stableDiffusionModelField.value

            use_stable_diffusion_model = getModelPath(use_stable_diffusion_model, [".ckpt", ".safetensors"])
            stableDiffusionModelField.value = use_stable_diffusion_model

            if (!stableDiffusionModelField.value) {
                stableDiffusionModelField.value = oldVal
            }
        },
        readUI: () => stableDiffusionModelField.value,
        parse: (val) => val,
    },
    clip_skip: {
        name: "Clip Skip",
        setUI: (value) => {
            clip_skip.checked = value
        },
        readUI: () => clip_skip.checked,
        parse: (val) => Boolean(val),
    },
    tiling: {
        name: "Tiling",
        setUI: (val) => {
            tilingField.value = val
        },
        readUI: () => tilingField.value,
        parse: (val) => val,
    },
    use_vae_model: {
        name: "VAE model",
        setUI: (use_vae_model) => {
            const oldVal = vaeModelField.value
            use_vae_model =
                use_vae_model === undefined || use_vae_model === null || use_vae_model === "None" ? "" : use_vae_model

            if (use_vae_model !== "") {
                use_vae_model = getModelPath(use_vae_model, [".vae.pt", ".ckpt"])
                use_vae_model = use_vae_model !== "" ? use_vae_model : oldVal
            }
            vaeModelField.value = use_vae_model
        },
        readUI: () => vaeModelField.value,
        parse: (val) => val,
    },
    use_controlnet_model: {
        name: "ControlNet model",
        setUI: (use_controlnet_model) => {
            controlnetModelField.value = getModelPath(use_controlnet_model, [".pth", ".safetensors"])
        },
        readUI: () => controlnetModelField.value,
        parse: (val) => val,
    },
    control_filter_to_apply: {
        name: "ControlNet Filter",
        setUI: (control_filter_to_apply) => {
            controlImageFilterField.value = control_filter_to_apply
        },
        readUI: () => controlImageFilterField.value,
        parse: (val) => val,
    },
    use_lora_model: {
        name: "LoRA model",
        setUI: (use_lora_model) => {
            let modelPaths = []
            use_lora_model = Array.isArray(use_lora_model) ? use_lora_model : [use_lora_model]
            use_lora_model.forEach((m) => {
                if (m.includes("models\\lora\\")) {
                    m = m.split("models\\lora\\")[1]
                } else if (m.includes("models\\\\lora\\\\")) {
                    m = m.split("models\\\\lora\\\\")[1]
                } else if (m.includes("models/lora/")) {
                    m = m.split("models/lora/")[1]
                }
                m = m.replaceAll("\\\\", "/")
                m = getModelPath(m, [".ckpt", ".safetensors"])
                modelPaths.push(m)
            })
            loraModelField.modelNames = modelPaths
        },
        readUI: () => {
            return loraModelField.modelNames
        },
        parse: (val) => {
            val = !val || val === "None" ? "" : val
            if (typeof val === "string" && val.includes(",")) {
                val = val.split(",")
                val = val.map((v) => v.trim())
                val = val.map((v) => v.replaceAll("\\", "\\\\"))
                val = val.map((v) => v.replaceAll('"', ""))
                val = val.map((v) => v.replaceAll("'", ""))
                val = val.map((v) => '"' + v + '"')
                val = "[" + val + "]"
                val = JSON.parse(val)
            }
            val = Array.isArray(val) ? val : [val]
            return val
        },
    },
    lora_alpha: {
        name: "LoRA Strength",
        setUI: (lora_alpha) => {
            lora_alpha = Array.isArray(lora_alpha) ? lora_alpha : [lora_alpha]
            loraModelField.modelWeights = lora_alpha
        },
        readUI: () => {
            return loraModelField.modelWeights
        },
        parse: (val) => {
            if (typeof val === "string" && val.includes(",")) {
                val = "[" + val.replaceAll("'", '"') + "]"
                val = JSON.parse(val)
            }
            val = Array.isArray(val) ? val : [val]
            val = val.map((e) => parseFloat(e))
            return val
        },
    },
    use_hypernetwork_model: {
        name: "Hypernetwork model",
        setUI: (use_hypernetwork_model) => {
            const oldVal = hypernetworkModelField.value
            use_hypernetwork_model =
                use_hypernetwork_model === undefined ||
                use_hypernetwork_model === null ||
                use_hypernetwork_model === "None"
                    ? ""
                    : use_hypernetwork_model

            if (use_hypernetwork_model !== "") {
                use_hypernetwork_model = getModelPath(use_hypernetwork_model, [".pt"])
                use_hypernetwork_model = use_hypernetwork_model !== "" ? use_hypernetwork_model : oldVal
            }
            hypernetworkModelField.value = use_hypernetwork_model
            hypernetworkModelField.dispatchEvent(new Event("change"))
        },
        readUI: () => hypernetworkModelField.value,
        parse: (val) => val,
    },
    hypernetwork_strength: {
        name: "Hypernetwork Strength",
        setUI: (hypernetwork_strength) => {
            hypernetworkStrengthField.value = hypernetwork_strength
            updateHypernetworkStrengthSlider()
        },
        readUI: () => parseFloat(hypernetworkStrengthField.value),
        parse: (val) => parseFloat(val),
    },

    num_outputs: {
        name: "Parallel Images",
        setUI: (num_outputs) => {
            numOutputsParallelField.value = num_outputs
        },
        readUI: () => parseInt(numOutputsParallelField.value),
        parse: (val) => val,
    },

    use_cpu: {
        name: "Use CPU",
        setUI: (use_cpu) => {
            useCPUField.checked = use_cpu
        },
        readUI: () => useCPUField.checked,
        parse: (val) => val,
    },

    stream_image_progress: {
        name: "Stream Image Progress",
        setUI: (stream_image_progress) => {
            streamImageProgressField.checked = parseInt(numOutputsTotalField.value) > 50 ? false : stream_image_progress
        },
        readUI: () => streamImageProgressField.checked,
        parse: (val) => Boolean(val),
    },
    show_only_filtered_image: {
        name: "Show only the corrected/upscaled image",
        setUI: (show_only_filtered_image) => {
            showOnlyFilteredImageField.checked = show_only_filtered_image
        },
        readUI: () => showOnlyFilteredImageField.checked,
        parse: (val) => Boolean(val),
    },
    output_format: {
        name: "Output Format",
        setUI: (output_format) => {
            outputFormatField.value = output_format
        },
        readUI: () => outputFormatField.value,
        parse: (val) => val,
    },
    save_to_disk_path: {
        name: "Save to disk path",
        setUI: (save_to_disk_path) => {
            saveToDiskField.checked = Boolean(save_to_disk_path)
            diskPathField.value = save_to_disk_path
        },
        readUI: () => diskPathField.value,
        parse: (val) => val,
    },
}

function restoreTaskToUI(task, fieldsToSkip) {
    fieldsToSkip = fieldsToSkip || []

    if ("numOutputsTotal" in task) {
        numOutputsTotalField.value = task.numOutputsTotal
    }
    if ("seed" in task) {
        randomSeedField.checked = false
        seedField.value = task.seed
    }
    if (!("reqBody" in task)) {
        return
    }
    for (const key in TASK_MAPPING) {
        if (key in task.reqBody && !fieldsToSkip.includes(key)) {
            TASK_MAPPING[key].setUI(task.reqBody[key])
        }
    }

    // properly reset fields not present in the task
    if (!("use_hypernetwork_model" in task.reqBody)) {
        hypernetworkModelField.value = ""
        hypernetworkModelField.dispatchEvent(new Event("change"))
    }

    if (!("use_lora_model" in task.reqBody)) {
        loraModelField.modelNames = []
        loraModelField.modelWeights = []
    }

    // restore the original prompt if provided (e.g. use settings), fallback to prompt as needed (e.g. copy/paste or d&d)
    promptField.value = task.reqBody.original_prompt
    if (!("original_prompt" in task.reqBody)) {
        promptField.value = task.reqBody.prompt
    }
    promptField.dispatchEvent(new Event("input"))

    // properly reset checkboxes
    if (!("use_face_correction" in task.reqBody)) {
        useFaceCorrectionField.checked = false
        gfpganModelField.disabled = true
    }
    if (!("use_upscale" in task.reqBody)) {
        useUpscalingField.checked = false
    }
    if (!("mask" in task.reqBody) && maskSetting.checked) {
        maskSetting.checked = false
        maskSetting.dispatchEvent(new Event("click"))
    }
    upscaleModelField.disabled = !useUpscalingField.checked
    upscaleAmountField.disabled = !useUpscalingField.checked

    // hide/show source picture as needed
    if (IMAGE_REGEX.test(initImagePreview.src) && task.reqBody.init_image == undefined) {
        // hide source image
        initImageClearBtn.dispatchEvent(new Event("click"))
    } else if (task.reqBody.init_image !== undefined) {
        // listen for inpainter loading event, which happens AFTER the main image loads (which reloads the inpainter)
        initImagePreview.addEventListener(
            "load",
            function() {
                if (Boolean(task.reqBody.mask)) {
                    imageInpainter.setImg(task.reqBody.mask)
                    maskSetting.checked = true
                }
            },
            { once: true }
        )
        initImagePreview.src = task.reqBody.init_image
    }

    // hide/show controlnet picture as needed
    if (IMAGE_REGEX.test(controlImagePreview.src) && task.reqBody.control_image == undefined) {
        // hide source image
        controlImageClearBtn.dispatchEvent(new Event("click"))
    } else if (task.reqBody.control_image !== undefined) {
        // listen for inpainter loading event, which happens AFTER the main image loads (which reloads the inpai
        controlImagePreview.src = task.reqBody.control_image
    }
}
function readUI() {
    const reqBody = {}
    for (const key in TASK_MAPPING) {
        if (testDiffusers.checked && (key === "use_hypernetwork_model" || key === "hypernetwork_strength")) {
            continue
        }

        reqBody[key] = TASK_MAPPING[key].readUI()
    }
    return {
        numOutputsTotal: parseInt(numOutputsTotalField.value),
        seed: TASK_MAPPING["seed"].readUI(),
        reqBody: reqBody,
    }
}
function getModelPath(filename, extensions) {
    if (typeof filename !== "string") {
        return
    }

    let pathIdx
    if (filename.includes("/models/stable-diffusion/")) {
        pathIdx = filename.indexOf("/models/stable-diffusion/") + 25 // Linux, Mac paths
    } else if (filename.includes("\\models\\stable-diffusion\\")) {
        pathIdx = filename.indexOf("\\models\\stable-diffusion\\") + 25 // Linux, Mac paths
    }
    if (pathIdx >= 0) {
        filename = filename.slice(pathIdx)
    }
    extensions.forEach((ext) => {
        if (filename.endsWith(ext)) {
            filename = filename.slice(0, filename.length - ext.length)
        }
    })
    return filename
}

const TASK_TEXT_MAPPING = {
    prompt: "Prompt",
    width: "Width",
    height: "Height",
    seed: "Seed",
    num_inference_steps: "Steps",
    guidance_scale: "Guidance Scale",
    prompt_strength: "Prompt Strength",
    use_face_correction: "Use Face Correction",
    use_upscale: "Use Upscaling",
    upscale_amount: "Upscale By",
    sampler_name: "Sampler",
    negative_prompt: "Negative Prompt",
    use_stable_diffusion_model: "Stable Diffusion model",
    use_hypernetwork_model: "Hypernetwork model",
    hypernetwork_strength: "Hypernetwork Strength",
    use_lora_model: "LoRA model",
    lora_alpha: "LoRA Strength",
    use_controlnet_model: "ControlNet model",
    control_filter_to_apply: "ControlNet Filter",
}
function parseTaskFromText(str) {
    const taskReqBody = {}

    const lines = str.split("\n")
    if (lines.length === 0) {
        return
    }

    // Prompt
    let knownKeyOnFirstLine = false
    for (let key in TASK_TEXT_MAPPING) {
        if (lines[0].startsWith(TASK_TEXT_MAPPING[key] + ":")) {
            knownKeyOnFirstLine = true
            break
        }
    }
    if (!knownKeyOnFirstLine) {
        taskReqBody.prompt = lines[0]
        console.log("Prompt:", taskReqBody.prompt)
    }

    for (const key in TASK_TEXT_MAPPING) {
        if (key in taskReqBody) {
            continue
        }

        const name = TASK_TEXT_MAPPING[key]
        let val = undefined

        const reName = new RegExp(`${name}\\ *:\\ *(.*)(?:\\r\\n|\\r|\\n)*`, "igm")
        const match = reName.exec(str)
        if (match) {
            str = str.slice(0, match.index) + str.slice(match.index + match[0].length)
            val = match[1]
        }
        if (val !== undefined) {
            taskReqBody[key] = TASK_MAPPING[key].parse(val.trim())
            console.log(TASK_MAPPING[key].name + ":", taskReqBody[key])
            if (!str) {
                break
            }
        }
    }
    if (Object.keys(taskReqBody).length <= 0) {
        return undefined
    }
    const task = { reqBody: taskReqBody }
    if ("seed" in taskReqBody) {
        task.seed = taskReqBody.seed
    }
    return task
}

async function parseContent(text) {
    text = text.trim()
    if (text.startsWith("{") && text.endsWith("}")) {
        try {
            const task = JSON.parse(text)
            if (!("reqBody" in task)) {
                // support the format saved to the disk, by the UI
                task.reqBody = Object.assign({}, task)
            }
            restoreTaskToUI(task)
            return true
        } catch (e) {
            console.warn(`JSON text content couldn't be parsed.`, e)
        }
        return false
    }
    // Normal txt file.
    const task = parseTaskFromText(text)
    if (text.toLowerCase().includes("seed:") && task) {
        // only parse valid task content
        restoreTaskToUI(task)
        return true
    } else {
        console.warn(`Raw text content couldn't be parsed.`)
        promptField.value = text
        return false
    }
}

async function readFile(file, i) {
    console.log(`Event %o reading file[${i}]:${file.name}...`)
    const fileContent = (await file.text()).trim()
    return await parseContent(fileContent)
}

function dropHandler(ev) {
    console.log("Content dropped...")
    let items = []

    if (ev?.dataTransfer?.items) {
        // Use DataTransferItemList interface
        items = Array.from(ev.dataTransfer.items)
        items = items.filter((item) => item.kind === "file")
        items = items.map((item) => item.getAsFile())
    } else if (ev?.dataTransfer?.files) {
        // Use DataTransfer interface
        items = Array.from(ev.dataTransfer.files)
    }

    items.forEach((item) => {
        item.file_ext = EXT_REGEX.exec(item.name.toLowerCase())[1]
    })

    let text_items = items.filter((item) => TEXT_EXTENSIONS.includes(item.file_ext))
    let image_items = items.filter((item) => IMAGE_EXTENSIONS.includes(item.file_ext))

    if (image_items.length > 0 && ev.target == initImageSelector) {
        return // let the event bubble up, so that the Init Image filepicker can receive this
    }

    ev.preventDefault() // Prevent default behavior (Prevent file/content from being opened)
    text_items.forEach(readFile)
}
function dragOverHandler(ev) {
    console.log("Content in drop zone")

    // Prevent default behavior (Prevent file/content from being opened)
    ev.preventDefault()

    ev.dataTransfer.dropEffect = "copy"

    let img = new Image()
    img.src = "//" + location.host + "/media/images/favicon-32x32.png"
    ev.dataTransfer.setDragImage(img, 16, 16)
}

document.addEventListener("drop", dropHandler)
document.addEventListener("dragover", dragOverHandler)

const TASK_REQ_NO_EXPORT = ["use_cpu", "save_to_disk_path"]
const resetSettings = document.getElementById("reset-image-settings")

function checkReadTextClipboardPermission(result) {
    if (result.state != "granted" && result.state != "prompt") {
        return
    }
    // PASTE ICON
    const pasteIcon = document.createElement("i")
    pasteIcon.className = "fa-solid fa-paste section-button"
    pasteIcon.innerHTML = `<span class="simple-tooltip top-left">Paste Image Settings</span>`
    pasteIcon.addEventListener("click", async (event) => {
        event.stopPropagation()
        // Add css class 'active'
        pasteIcon.classList.add("active")
        // In 350 ms remove the 'active' class
        asyncDelay(350).then(() => pasteIcon.classList.remove("active"))

        // Retrieve clipboard content and try to parse it
        const text = await navigator.clipboard.readText()
        await parseContent(text)
    })
    resetSettings.parentNode.insertBefore(pasteIcon, resetSettings)
}
navigator.permissions
    .query({ name: "clipboard-read" })
    .then(checkReadTextClipboardPermission, (reason) => console.log("clipboard-read is not available. %o", reason))

document.addEventListener("paste", async (event) => {
    if (event.target) {
        const targetTag = event.target.tagName.toLowerCase()
        // Disable when targeting input elements.
        if (targetTag === "input" || targetTag === "textarea") {
            return
        }
    }
    const paste = (event.clipboardData || window.clipboardData).getData("text")
    const selection = window.getSelection()
    if (paste != "" && selection.toString().trim().length <= 0 && (await parseContent(paste))) {
        event.preventDefault()
        return
    }
})

// Adds a copy and a paste icon if the browser grants permission to write to clipboard.
function checkWriteToClipboardPermission(result) {
    if (result.state != "granted" && result.state != "prompt") {
        return
    }
    // COPY ICON
    const copyIcon = document.createElement("i")
    copyIcon.className = "fa-solid fa-clipboard section-button"
    copyIcon.innerHTML = `<span class="simple-tooltip top-left">Copy Image Settings</span>`
    copyIcon.addEventListener("click", (event) => {
        event.stopPropagation()
        // Add css class 'active'
        copyIcon.classList.add("active")
        // In 350 ms remove the 'active' class
        asyncDelay(350).then(() => copyIcon.classList.remove("active"))
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
// Determine which access we have to the clipboard. Clipboard access is only available on localhost or via TLS.
navigator.permissions.query({ name: "clipboard-write" }).then(checkWriteToClipboardPermission, (e) => {
    if (e instanceof TypeError && typeof navigator?.clipboard?.writeText === "function") {
        // Fix for firefox https://bugzilla.mozilla.org/show_bug.cgi?id=1560373
        checkWriteToClipboardPermission({ state: "granted" })
    }
})
