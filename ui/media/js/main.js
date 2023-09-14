"use strict" // Opt in to a restricted variant of JavaScript
const MAX_INIT_IMAGE_DIMENSION = 768
const MIN_GPUS_TO_SHOW_SELECTION = 2

const IMAGE_REGEX = new RegExp("data:image/[A-Za-z]+;base64")

const spinnerPacmanHtml =
    '<div class="loadingio-spinner-bean-eater-x0y3u8qky4n"><div class="ldio-8f673ktaleu"><div><div></div><div></div><div></div></div><div><div></div><div></div><div></div></div></div></div>'

const taskConfigSetup = {
    taskConfig: {
        seed: { value: ({ seed }) => seed, label: "Seed" },
        dimensions: { value: ({ reqBody }) => `${reqBody?.width}x${reqBody?.height}`, label: "Dimensions" },
        sampler_name: "Sampler",
        num_inference_steps: "Inference Steps",
        guidance_scale: "Guidance Scale",
        use_stable_diffusion_model: "Model",
        clip_skip: {
            label: "Clip Skip",
            visible: ({ reqBody }) => reqBody?.clip_skip,
            value: ({ reqBody }) => "yes",
        },
        tiling: {
            label: "Tiling",
            visible: ({ reqBody }) =>
                reqBody?.tiling != "none" && reqBody?.tiling !== null && reqBody?.tiling !== undefined,
            value: ({ reqBody }) => reqBody?.tiling,
        },
        use_vae_model: {
            label: "VAE",
            visible: ({ reqBody }) => reqBody?.use_vae_model !== undefined && reqBody?.use_vae_model.trim() !== "",
        },
        negative_prompt: {
            label: "Negative Prompt",
            visible: ({ reqBody }) => reqBody?.negative_prompt !== undefined && reqBody?.negative_prompt.trim() !== "",
        },
        prompt_strength: "Prompt Strength",
        use_face_correction: "Fix Faces",
        upscale: {
            value: ({ reqBody }) => `${reqBody?.use_upscale} (${reqBody?.upscale_amount || 4}x)`,
            label: "Upscale",
            visible: ({ reqBody }) => !!reqBody?.use_upscale,
        },
        use_hypernetwork_model: "Hypernetwork",
        hypernetwork_strength: {
            label: "Hypernetwork Strength",
            visible: ({ reqBody }) => !!reqBody?.use_hypernetwork_model,
        },
        use_lora_model: { label: "Lora Model", visible: ({ reqBody }) => !!reqBody?.use_lora_model },
        lora_alpha: { label: "Lora Strength", visible: ({ reqBody }) => !!reqBody?.use_lora_model },
        preserve_init_image_color_profile: "Preserve Color Profile",
        strict_mask_border: "Strict Mask Border",
        use_controlnet_model: "ControlNet Model",
    },
    pluginTaskConfig: {},
    getCSSKey: (key) =>
        key
            .split("_")
            .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
            .join(""),
}

let imageCounter = 0
let imageRequest = []

let promptField = document.querySelector("#prompt")
let promptsFromFileSelector = document.querySelector("#prompt_from_file")
let promptsFromFileBtn = document.querySelector("#promptsFromFileBtn")
let negativePromptField = document.querySelector("#negative_prompt")
let numOutputsTotalField = document.querySelector("#num_outputs_total")
let numOutputsParallelField = document.querySelector("#num_outputs_parallel")
let numInferenceStepsField = document.querySelector("#num_inference_steps")
let guidanceScaleSlider = document.querySelector("#guidance_scale_slider")
let guidanceScaleField = document.querySelector("#guidance_scale")
let outputQualitySlider = document.querySelector("#output_quality_slider")
let outputQualityField = document.querySelector("#output_quality")
let outputQualityRow = document.querySelector("#output_quality_row")
let randomSeedField = document.querySelector("#random_seed")
let seedField = document.querySelector("#seed")
let widthField = document.querySelector("#width")
let heightField = document.querySelector("#height")
let customWidthField = document.querySelector("#custom-width")
let customHeightField = document.querySelector("#custom-height")
let recentResolutionsButton = document.querySelector("#recent-resolutions-button")
let recentResolutionsPopup = document.querySelector("#recent-resolutions-popup")
let recentResolutionList = document.querySelector("#recent-resolution-list")
let commonResolutionList = document.querySelector("#common-resolution-list")
let resizeSlider = document.querySelector("#resize-slider")
let enlargeButtons = document.querySelector("#enlarge-buttons")
let swapWidthHeightButton = document.querySelector("#swap-width-height")
let smallImageWarning = document.querySelector("#small_image_warning")
let initImageSelector = document.querySelector("#init_image")
let initImagePreview = document.querySelector("#init_image_preview")
let initImageSizeBox = document.querySelector("#init_image_size_box")
let maskImageSelector = document.querySelector("#mask")
let maskImagePreview = document.querySelector("#mask_preview")
let controlImageSelector = document.querySelector("#control_image")
let controlImagePreview = document.querySelector("#control_image_preview")
let controlImageClearBtn = document.querySelector(".control_image_clear")
let controlImageContainer = document.querySelector("#control_image_wrapper")
let controlImageFilterField = document.querySelector("#control_image_filter")
let applyColorCorrectionField = document.querySelector("#apply_color_correction")
let strictMaskBorderField = document.querySelector("#strict_mask_border")
let colorCorrectionSetting = document.querySelector("#apply_color_correction_setting")
let strictMaskBorderSetting = document.querySelector("#strict_mask_border_setting")
let promptStrengthSlider = document.querySelector("#prompt_strength_slider")
let promptStrengthField = document.querySelector("#prompt_strength")
let samplerField = document.querySelector("#sampler_name")
let samplerSelectionContainer = document.querySelector("#samplerSelection")
let useFaceCorrectionField = document.querySelector("#use_face_correction")
let gfpganModelField = new ModelDropdown(document.querySelector("#gfpgan_model"), ["gfpgan", "codeformer"], "", false)
let useUpscalingField = document.querySelector("#use_upscale")
let upscaleModelField = document.querySelector("#upscale_model")
let upscaleAmountField = document.querySelector("#upscale_amount")
let latentUpscalerSettings = document.querySelector("#latent_upscaler_settings")
let latentUpscalerStepsSlider = document.querySelector("#latent_upscaler_steps_slider")
let latentUpscalerStepsField = document.querySelector("#latent_upscaler_steps")
let codeformerFidelitySlider = document.querySelector("#codeformer_fidelity_slider")
let codeformerFidelityField = document.querySelector("#codeformer_fidelity")
let stableDiffusionModelField = new ModelDropdown(document.querySelector("#stable_diffusion_model"), "stable-diffusion")
let clipSkipField = document.querySelector("#clip_skip")
let tilingField = document.querySelector("#tiling")
let controlnetModelField = new ModelDropdown(document.querySelector("#controlnet_model"), "controlnet", "None", false)
let vaeModelField = new ModelDropdown(document.querySelector("#vae_model"), "vae", "None")
let loraModelField = new MultiModelSelector(document.querySelector("#lora_model"), "lora", "LoRA", 0.5, 0.02)
let hypernetworkModelField = new ModelDropdown(document.querySelector("#hypernetwork_model"), "hypernetwork", "None")
let hypernetworkStrengthSlider = document.querySelector("#hypernetwork_strength_slider")
let hypernetworkStrengthField = document.querySelector("#hypernetwork_strength")
let outputFormatField = document.querySelector("#output_format")
let outputLosslessField = document.querySelector("#output_lossless")
let outputLosslessContainer = document.querySelector("#output_lossless_container")
let blockNSFWField = document.querySelector("#block_nsfw")
let showOnlyFilteredImageField = document.querySelector("#show_only_filtered_image")
let updateBranchLabel = document.querySelector("#updateBranchLabel")
let streamImageProgressField = document.querySelector("#stream_image_progress")
let thumbnailSizeField = document.querySelector("#thumbnail_size-input")
let autoscrollBtn = document.querySelector("#auto_scroll_btn")
let autoScroll = document.querySelector("#auto_scroll")
let embeddingsButton = document.querySelector("#embeddings-button")
let negativeEmbeddingsButton = document.querySelector("#negative-embeddings-button")
let embeddingsDialog = document.querySelector("#embeddings-dialog")
let embeddingsDialogCloseBtn = embeddingsDialog.querySelector("#embeddings-dialog-close-button")
let embeddingsSearchBox = document.querySelector("#embeddings-search-box")
let embeddingsList = document.querySelector("#embeddings-list")
let embeddingsModeField = document.querySelector("#embeddings-mode")
let embeddingsCardSizeSelector = document.querySelector("#embedding-card-size-selector")
let addEmbeddingsThumb = document.querySelector("#add-embeddings-thumb")
let addEmbeddingsThumbInput = document.querySelector("#add-embeddings-thumb-input")

let positiveEmbeddingText = document.querySelector("#positive-embedding-text")
let negativeEmbeddingText = document.querySelector("#negative-embedding-text")
let embeddingsCollapsiblesBtn = document.querySelector("#embeddings-action-collapsibles-btn")

let makeImageBtn = document.querySelector("#makeImage")
let stopImageBtn = document.querySelector("#stopImage")
let renderButtons = document.querySelector("#render-buttons")

let imagesContainer = document.querySelector("#current-images")
let initImagePreviewContainer = document.querySelector("#init_image_preview_container")
let initImageClearBtn = document.querySelector(".init_image_clear")
let promptStrengthContainer = document.querySelector("#prompt_strength_container")

let initialText = document.querySelector("#initial-text")
let supportBanner = document.querySelector("#supportBanner")
let versionText = document.querySelector("#version")
let previewTools = document.querySelector("#preview-tools")
let clearAllPreviewsBtn = document.querySelector("#clear-all-previews")
let showDownloadDialogBtn = document.querySelector("#show-download-popup")
let saveAllImagesDialog = document.querySelector("#download-images-dialog")
let saveAllImagesBtn = document.querySelector("#save-all-images")
let saveAllImagesCloseBtn = document.querySelector("#download-images-close-button")
let saveAllZipToggle = document.querySelector("#zip_toggle")
let saveAllTreeToggle = document.querySelector("#tree_toggle")
let saveAllJSONToggle = document.querySelector("#json_toggle")
let saveAllFoldersOption = document.querySelector("#download-add-folders")
let splashScreenPopup = document.querySelector("#splash-screen")
let useAsThumbDialog = document.querySelector("#use-as-thumb-dialog")
let useAsThumbDialogCloseBtn = document.querySelector("#use-as-thumb-dialog-close-button")
let useAsThumbImageContainer = document.querySelector("#use-as-thumb-img-container")
let useAsThumbSelect = document.querySelector("#use-as-thumb-select")
let useAsThumbSaveBtn = document.querySelector("#use-as-thumb-save")
let useAsThumbCancelBtn = document.querySelector("#use-as-thumb-cancel")

let maskSetting = document.querySelector("#enable_mask")

let imagePreview = document.querySelector("#preview")
let imagePreviewContent = document.querySelector("#preview-content")

let undoButton = document.querySelector("#undo")
let undoBuffer = []
const UNDO_LIMIT = 20
const MAX_IMG_UNDO_ENTRIES = 5

let IMAGE_STEP_SIZE = 64

let loraModels = []

imagePreview.addEventListener("drop", function(ev) {
    const data = ev.dataTransfer?.getData("text/plain")
    if (!data) {
        return
    }
    const movedTask = document.getElementById(data)
    if (!movedTask) {
        return
    }
    ev.preventDefault()
    let moveTarget = ev.target
    while (moveTarget && typeof moveTarget === "object" && moveTarget.parentNode !== imagePreviewContent) {
        moveTarget = moveTarget.parentNode
    }
    if (moveTarget === initialText || moveTarget === previewTools) {
        moveTarget = null
    }
    if (moveTarget === movedTask) {
        return
    }
    if (moveTarget) {
        const childs = Array.from(imagePreviewContent.children)
        if (moveTarget.nextSibling && childs.indexOf(movedTask) < childs.indexOf(moveTarget)) {
            // Move after the target if lower than current position.
            moveTarget = moveTarget.nextSibling
        }
    }
    const newNode = imagePreviewContent.insertBefore(movedTask, moveTarget || previewTools.nextSibling)
    if (newNode === movedTask) {
        return
    }
    imagePreviewContent.removeChild(movedTask)
    const task = htmlTaskMap.get(movedTask)
    if (task) {
        htmlTaskMap.delete(movedTask)
    }
    if (task) {
        htmlTaskMap.set(newNode, task)
    }
})

let showConfigToggle = document.querySelector("#configToggleBtn")
// let configBox = document.querySelector('#config')
// let outputMsg = document.querySelector('#outputMsg')

let soundToggle = document.querySelector("#sound_toggle")

let serverStatusColor = document.querySelector("#server-status-color")
let serverStatusMsg = document.querySelector("#server-status-msg")

function getLocalStorageBoolItem(key, fallback) {
    let item = localStorage.getItem(key)
    if (item === null) {
        return fallback
    }

    return item === "true" ? true : false
}

function handleBoolSettingChange(key) {
    return function(e) {
        localStorage.setItem(key, e.target.checked.toString())
    }
}

function handleStringSettingChange(key) {
    return function(e) {
        localStorage.setItem(key, e.target.value.toString())
    }
}

function isSoundEnabled() {
    return getSetting("sound_toggle")
}

function getSavedDiskPath() {
    return getSetting("diskPath")
}

function setStatus(statusType, msg, msgType) {}

function setServerStatus(event) {
    switch (event.type) {
        case "online":
            serverStatusColor.style.color = "var(--status-green)"
            serverStatusMsg.style.color = "var(--status-green)"
            serverStatusMsg.innerText = "Stable Diffusion is " + event.message
            break
        case "busy":
            serverStatusColor.style.color = "var(--status-orange)"
            serverStatusMsg.style.color = "var(--status-orange)"
            serverStatusMsg.innerText = "Stable Diffusion is " + event.message
            break
        case "error":
            serverStatusColor.style.color = "var(--status-red)"
            serverStatusMsg.style.color = "var(--status-red)"
            serverStatusMsg.innerText = "Stable Diffusion has stopped"
            break
    }
    if (SD.serverState.devices) {
        document.dispatchEvent(new CustomEvent("system_info_update", { detail: SD.serverState.devices }))
    }
}

// shiftOrConfirm(e, prompt, fn)
//   e      : MouseEvent
//   prompt : Text to be shown as prompt. Should be a question to which "yes" is a good answer.
//   fn     : function to be called if the user confirms the dialog or has the shift key pressed
//   allowSkip: Allow skipping the dialog using the shift key or the confirm_dangerous_actions setting (default: true)
//
// If the user had the shift key pressed while clicking, the function fn will be executed.
// If the setting "confirm_dangerous_actions" in the system settings is disabled, the function
// fn will be executed.
// Otherwise, a confirmation dialog is shown. If the user confirms, the function fn will also
// be executed.
function shiftOrConfirm(e, prompt, fn, allowSkip = true) {
    e.stopPropagation()
    let tip = allowSkip
        ? '<small>Tip: To skip this dialog, use shift-click or disable the "Confirm dangerous actions" setting in the Settings tab.</small>'
        : ""
    if (allowSkip && (e.shiftKey || !confirmDangerousActionsField.checked)) {
        fn(e)
    } else {
        confirm(tip, prompt, () => {
            fn(e)
        })
    }
}

function undoableRemove(element, doubleUndo = false) {
    let data = {
        element: element,
        parent: element.parentNode,
        prev: element.previousSibling,
        next: element.nextSibling,
        doubleUndo: doubleUndo,
    }
    undoBuffer.push(data)
    if (undoBuffer.length > UNDO_LIMIT) {
        // Remove item from memory and also remove it from the data structures
        let item = undoBuffer.shift()
        htmlTaskMap.delete(item.element)
        item.element.querySelectorAll("[data-imagecounter]").forEach((img) => {
            delete imageRequest[img.dataset["imagecounter"]]
        })
    }
    element.remove()
    if (undoBuffer.length != 0) {
        undoButton.classList.remove("displayNone")
    }
}

function undoRemove() {
    let data = undoBuffer.pop()
    if (!data) {
        return
    }
    if (data.next == null) {
        data.parent.appendChild(data.element)
    } else {
        data.parent.insertBefore(data.element, data.next)
    }
    if (data.doubleUndo) {
        undoRemove()
    }
    if (undoBuffer.length == 0) {
        undoButton.classList.add("displayNone")
    }
    updateInitialText()
}

undoButton.addEventListener("click", () => {
    undoRemove()
})

document.addEventListener("keydown", function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "z" && e.target == document.body) {
        undoRemove()
    }
})

function showImages(reqBody, res, outputContainer, livePreview) {
    let imageItemElements = outputContainer.querySelectorAll(".imgItem")
    if (typeof res != "object") return
    res.output.reverse()
    res.output.forEach((result, index) => {
        const imageData = result?.data || result?.path + "?t=" + Date.now(),
            imageSeed = result?.seed,
            imagePrompt = reqBody.prompt,
            imageInferenceSteps = reqBody.num_inference_steps,
            imageGuidanceScale = reqBody.guidance_scale,
            imageWidth = reqBody.width,
            imageHeight = reqBody.height

        if (!imageData.includes("/")) {
            // res contained no data for the image, stop execution
            setStatus("request", "invalid image", "error")
            return
        }

        let imageItemElem = index < imageItemElements.length ? imageItemElements[index] : null
        if (!imageItemElem) {
            imageItemElem = document.createElement("div")
            imageItemElem.className = "imgItem"
            imageItemElem.innerHTML = `
                <div class="imgContainer">
                    <img/>
                    <div class="imgItemInfo">
                        <div>
                            <span class="imgInfoLabel imgExpandBtn"><i class="fa-solid fa-expand"></i></span><span class="imgInfoLabel imgSeedLabel"></span>
                        </div>
                    </div>
                    <button class="imgPreviewItemClearBtn image_clear_btn"><i class="fa-solid fa-xmark"></i></button>
                    <span class="img_bottom_label"></span>
                    <div class="spinner displayNone"><center>${spinnerPacmanHtml}</center><div class="spinnerStatus"></div></div>
                </div>
            `
            outputContainer.appendChild(imageItemElem)
            const imageRemoveBtn = imageItemElem.querySelector(".imgPreviewItemClearBtn")
            let parentTaskContainer = imageRemoveBtn.closest(".imageTaskContainer")
            imageRemoveBtn.addEventListener("click", (e) => {
                undoableRemove(imageItemElem)
                let allHidden = true
                let children = parentTaskContainer.querySelectorAll(".imgItem")
                for (let x = 0; x < children.length; x++) {
                    let child = children[x]
                    if (child.style.display != "none") {
                        allHidden = false
                    }
                }
                if (allHidden === true) {
                    const req = htmlTaskMap.get(parentTaskContainer)
                    if (!req.isProcessing || req.batchesDone == req.batchCount) {
                        undoableRemove(parentTaskContainer, true)
                    }
                }
            })
        }
        const imageElem = imageItemElem.querySelector("img")
        imageElem.src = imageData
        imageElem.width = parseInt(imageWidth)
        imageElem.height = parseInt(imageHeight)
        imageElem.setAttribute("data-prompt", imagePrompt)
        imageElem.setAttribute("data-steps", imageInferenceSteps)
        imageElem.setAttribute("data-guidance", imageGuidanceScale)

        imageElem.addEventListener("load", function() {
            imageItemElem.querySelector(".img_bottom_label").innerText = `${this.naturalWidth} x ${this.naturalHeight}`
        })

        const imageInfo = imageItemElem.querySelector(".imgItemInfo")
        imageInfo.style.visibility = livePreview ? "hidden" : "visible"

        if ("seed" in result && !imageElem.hasAttribute("data-seed")) {
            const imageExpandBtn = imageItemElem.querySelector(".imgExpandBtn")
            imageExpandBtn.addEventListener("click", function() {
                function previousImage(img) {
                    const allImages = Array.from(outputContainer.parentNode.querySelectorAll(".imgItem img"))
                    const index = allImages.indexOf(img)
                    return allImages.slice(0, index).reverse()[0]
                }

                function nextImage(img) {
                    const allImages = Array.from(outputContainer.parentNode.querySelectorAll(".imgItem img"))
                    const index = allImages.indexOf(img)
                    return allImages.slice(index + 1)[0]
                }

                function imageModalParameter(img) {
                    const previousImg = previousImage(img)
                    const nextImg = nextImage(img)

                    return {
                        src: img.src,
                        previous: previousImg ? () => imageModalParameter(previousImg) : undefined,
                        next: nextImg ? () => imageModalParameter(nextImg) : undefined,
                    }
                }

                imageModal(imageModalParameter(imageElem))
            })

            const req = Object.assign({}, reqBody, {
                seed: result?.seed || reqBody.seed,
            })
            imageElem.setAttribute("data-seed", req.seed)
            imageElem.setAttribute("data-imagecounter", ++imageCounter)
            imageRequest[imageCounter] = req
            const imageSeedLabel = imageItemElem.querySelector(".imgSeedLabel")
            imageSeedLabel.innerText = "Seed: " + req.seed

            const imageUndoBuffer = []
            const imageRedoBuffer = []
            let buttons = [
                { text: "Use as Input", on_click: onUseAsInputClick },
                { text: "Use for Controlnet", on_click: onUseForControlnetClick },
                [
                    {
                        html: '<i class="fa-solid fa-download"></i> Download Image',
                        on_click: onDownloadImageClick,
                        class: "download-img",
                    },
                    {
                        html: '<i class="fa-solid fa-download"></i> JSON',
                        on_click: onDownloadJSONClick,
                        class: "download-json",
                    },
                ],
                { text: "Make Similar Images", on_click: onMakeSimilarClick },
                { text: "Draw another 25 steps", on_click: onContinueDrawingClick },
                [
                    { html: '<i class="fa-solid fa-undo"></i> Undo', on_click: onUndoFilter },
                    { html: '<i class="fa-solid fa-redo"></i> Redo', on_click: onRedoFilter },
                    { text: "Upscale", on_click: onUpscaleClick },
                    { text: "Fix Faces", on_click: onFixFacesClick },
                ],
                { 
                    text: "Use as Thumbnail",
                    on_click: onUseAsThumbnailClick,
                    filter: (req, img) => "use_embeddings_model" in req || "use_lora_model" in req
                },
            ]

            // include the plugins
            buttons = buttons.concat(PLUGINS["IMAGE_INFO_BUTTONS"])

            const imgItemInfo = imageItemElem.querySelector(".imgItemInfo")
            const img = imageItemElem.querySelector("img")
            const spinner = imageItemElem.querySelector(".spinner")
            const spinnerStatus = imageItemElem.querySelector(".spinnerStatus")
            const tools = {
                spinner: spinner,
                spinnerStatus: spinnerStatus,
                undoBuffer: imageUndoBuffer,
                redoBuffer: imageRedoBuffer,
            }
            const createButton = function(btnInfo) {
                if (Array.isArray(btnInfo)) {
                    const wrapper = document.createElement("div")
                    btnInfo.map(createButton).forEach((buttonElement) => wrapper.appendChild(buttonElement))
                    return wrapper
                }

                const isLabel = btnInfo.type === "label"

                const newButton = document.createElement(isLabel ? "span" : "button")
                newButton.classList.add("tasksBtns")

                if (btnInfo.html) {
                    const html = typeof btnInfo.html === "function" ? btnInfo.html() : btnInfo.html
                    if (html instanceof HTMLElement) {
                        newButton.appendChild(html)
                    } else {
                        newButton.innerHTML = html
                    }
                } else {
                    newButton.innerText = typeof btnInfo.text === "function" ? btnInfo.text() : btnInfo.text
                }

                if (btnInfo.on_click || !isLabel) {
                    newButton.addEventListener("click", function(event) {
                        btnInfo.on_click.bind(newButton)(req, img, event, tools)
                    })
                    if (btnInfo.on_click === onUndoFilter) {
                        tools["undoButton"] = newButton
                        newButton.classList.add("displayNone")
                    }
                    if (btnInfo.on_click === onRedoFilter) {
                        tools["redoButton"] = newButton
                        newButton.classList.add("displayNone")
                    }
                }

                if (btnInfo.class !== undefined) {
                    if (Array.isArray(btnInfo.class)) {
                        newButton.classList.add(...btnInfo.class)
                    } else {
                        newButton.classList.add(btnInfo.class)
                    }
                }
                return newButton
            }
            buttons.forEach((btn) => {
                if (Array.isArray(btn)) {
                    btn = btn.filter((btnInfo) => !btnInfo.filter || btnInfo.filter(req, img) === true)
                    if (btn.length === 0) {
                        return
                    }
                } else if (btn.filter && btn.filter(req, img) === false) {
                    return
                }

                try {
                    imgItemInfo.appendChild(createButton(btn))
                } catch (err) {
                    console.error("Error creating image info button from plugin: ", btn, err)
                }
            })
        }
    })
}

function onUseAsInputClick(req, img) {
    const imgData = img.src

    initImageSelector.value = null
    initImagePreview.src = imgData

    maskSetting.checked = false
}

function onUseForControlnetClick(req, img) {
    controlImagePreview.src = img.src
}

function getDownloadFilename(img, suffix) {
    const imageSeed = img.getAttribute("data-seed")
    const imagePrompt = img.getAttribute("data-prompt")
    const imageInferenceSteps = img.getAttribute("data-steps")
    const imageGuidanceScale = img.getAttribute("data-guidance")

    return createFileName(imagePrompt, imageSeed, imageInferenceSteps, imageGuidanceScale, suffix)
}

function onDownloadJSONClick(req, img) {
    const name = getDownloadFilename(img, "json")
    const blob = new Blob([JSON.stringify(req, null, 2)], { type: "text/plain" })
    saveAs(blob, name)
}

function onDownloadImageClick(req, img) {
    const name = getDownloadFilename(img, req["output_format"])
    const blob = dataURItoBlob(img.src)
    saveAs(blob, name)
}

function modifyCurrentRequest(...reqDiff) {
    const newTaskRequest = getCurrentUserRequest()

    newTaskRequest.reqBody = Object.assign(newTaskRequest.reqBody, ...reqDiff, {
        use_cpu: useCPUField.checked,
    })
    newTaskRequest.seed = newTaskRequest.reqBody.seed

    return newTaskRequest
}

function onMakeSimilarClick(req, img) {
    const newTaskRequest = modifyCurrentRequest(req, {
        num_outputs: 1,
        num_inference_steps: 50,
        guidance_scale: 7.5,
        prompt_strength: 0.7,
        init_image: img.src,
        seed: Math.floor(Math.random() * 10000000),
    })

    newTaskRequest.numOutputsTotal = 5
    newTaskRequest.batchCount = 5

    delete newTaskRequest.reqBody.mask

    createTask(newTaskRequest)
}

// gets a flat list of all models of a certain type, ignoring directories
function getAllModelNames(type) {
    function f(tree) {
        if (tree == undefined) {
            return []
        }
        let result = []
        tree.forEach((e) => {
            if (typeof e == "object") {
                result = result.concat(f(e[1]))
            } else {
                result.push(e)
            }
        })
        return result
    }
    return f(modelsOptions[type])
}

// gets a flattened list of all models of a certain type. e.g. "path/subpath/modelname"
// use the filter to search for all models having a certain name.
function getAllModelPathes(type, filter = "") {
    function f(tree, prefix) {
        if (tree == undefined) {
            return []
        }
        let result = []
        tree.forEach((e) => {
            if (typeof e == "object") {
                result = result.concat(f(e[1], prefix + e[0] + "/"))
            } else {
                if (filter == "" || e == filter) {
                    result.push(prefix + e)
                }
            }
        })
        return result
    }
    return f(modelsOptions[type], "")
}

function onUseAsThumbnailClick(req, img) {
    let scale = 1
    let targetWidth = img.naturalWidth
    let targetHeight = img.naturalHeight
    let resize = false
    onUseAsThumbnailClick.img = img

    if (typeof onUseAsThumbnailClick.croppr == "undefined") {
        onUseAsThumbnailClick.croppr = new Croppr("#use-as-thumb-image", {
            aspectRatio: 1,
            minSize: [384, 384, "px"],
            startSize: [512, 512, "px"],
            returnMode: "real",
        })
    }

    if (img.naturalWidth > img.naturalHeight) {
        if (img.naturalWidth > 768) {
            scale = 768 / img.naturalWidth
            targetWidth = 768
            targetHeight = (img.naturalHeight * scale) >>> 0
            resize = true
        }
    } else {
        if (img.naturalHeight > 768) {
            scale = 768 / img.naturalHeight
            targetHeight = 768
            targetWidth = (img.naturalWidth * scale) >>> 0
            resize = true
        }
    }

    onUseAsThumbnailClick.croppr.options.minSize = { width: (384 * scale) >>> 0, height: (384 * scale) >>> 0 }
    onUseAsThumbnailClick.croppr.options.startSize = { width: (512 * scale) >>> 0, height: (512 * scale) >>> 0 }

    if (resize) {
        const canvas = document.createElement("canvas")
        canvas.width = targetWidth
        canvas.height = targetHeight
        const ctx = canvas.getContext("2d")
        ctx.drawImage(img, 0, 0, targetWidth, targetHeight)

        onUseAsThumbnailClick.croppr.setImage(canvas.toDataURL("image/png"))
    } else {
        onUseAsThumbnailClick.croppr.setImage(img.src)
    }

    useAsThumbSelect.innerHTML=""

    if ("use_embeddings_model" in req) {
        let embeddings = req.use_embeddings_model.map((e) => e.split("/").pop())

        let embOptions = document.createElement("optgroup")
        embOptions.label = "Embeddings"
        embOptions.replaceChildren(
            ...embeddings.map((e) => {
                let option = document.createElement("option")
                option.innerText = e
                option.dataset["type"] = "embeddings"
                return option
            })
        )
        useAsThumbSelect.appendChild(embOptions)
    }


    if ("use_lora_model" in req) {
        let LORA = req.use_lora_model
        if (typeof LORA == "string") {
            LORA = [LORA]
        }
        LORA = LORA.map((e) => e.split("/").pop())

        let loraOptions = document.createElement("optgroup")
        loraOptions.label = "LORA"
        loraOptions.replaceChildren(
            ...LORA.map((e) => {
                let option = document.createElement("option")
                option.innerText = e
                option.dataset["type"] = "lora"
                return option
            })
        )
        useAsThumbSelect.appendChild(loraOptions)
    }

    useAsThumbDialog.showModal()
    onUseAsThumbnailClick.scale = scale
}

modalDialogCloseOnBackdropClick(useAsThumbDialog)
makeDialogDraggable(useAsThumbDialog)

useAsThumbDialogCloseBtn.addEventListener("click", () => {
    useAsThumbDialog.close()
})

useAsThumbCancelBtn.addEventListener("click", () => {
    useAsThumbDialog.close()
})

const Bucket = {
    upload(path, blob) {
        const formData = new FormData()
        formData.append("file", blob)
        return fetch(`bucket/${path}`, {
            method: "POST",
            body: formData,
        })
    },

    getImageAsDataURL(path) {
        return fetch(`bucket/${path}`)
            .then((response) => {
                if (response.status == 200) {
                    return response.blob()
                } else {
                    throw new Error("Bucket error")
                }
            })
            .then((blob) => {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader()
                    reader.onload = () => resolve(reader.result)
                    reader.onerror = reject
                    reader.readAsDataURL(blob)
                })
            })
    },

    getList(path) {
        return fetch(`bucket/${path}`)
            .then((response) => (response.status == 200 ? response.json() : []))
    },

    store(path, data) {
        return Bucket.upload(`${path}.json`, JSON.stringify(data))
    },

    retrieve(path) {
        return fetch(`bucket/${path}.json`)
            .then((response) => (response.status == 200 ? response.json() : null))
    },
}

useAsThumbSaveBtn.addEventListener("click", (e) => {
    let scale = 1 / onUseAsThumbnailClick.scale
    let crop = onUseAsThumbnailClick.croppr.getValue()

    let len = Math.max(crop.width * scale, 384)
    let profileName = profileNameField.value

    cropImageDataUrl(onUseAsThumbnailClick.img.src, crop.x * scale, crop.y * scale, len, len)
        .then((thumb) => fetch(thumb))
        .then((response) => response.blob())
        .then(async function(blob) {
            let options = useAsThumbSelect.selectedOptions
            let promises = []
            for (let embedding of options) {
                promises.push(
                    Bucket.upload(`${profileName}/${embedding.dataset["type"]}/${embedding.value}.png`, blob)
                )
            }
            return Promise.all(promises)
        })
        .then(() => {
            useAsThumbDialog.close()
        })
        .catch((error) => {
            console.error(error)
            showToast("Couldn't save thumbnail.<br>" + error)
        })
})

function enqueueImageVariationTask(req, img, reqDiff) {
    const imageSeed = img.getAttribute("data-seed")

    const newRequestBody = {
        num_outputs: 1, // this can be user-configurable in the future
        seed: imageSeed,
    }

    // If the user is editing pictures, stop modifyCurrentRequest from importing
    // new values by setting the missing properties to undefined
    if (!("init_image" in req) && !("init_image" in reqDiff)) {
        newRequestBody.init_image = undefined
        newRequestBody.mask = undefined
    } else if (!("mask" in req) && !("mask" in reqDiff)) {
        newRequestBody.mask = undefined
    }

    const newTaskRequest = modifyCurrentRequest(req, reqDiff, newRequestBody)
    newTaskRequest.numOutputsTotal = 1 // this can be user-configurable in the future
    newTaskRequest.batchCount = 1

    createTask(newTaskRequest)
}

function applyInlineFilter(filterName, path, filterParams, img, statusText, tools) {
    const filterReq = {
        image: img.src,
        filter: filterName,
        model_paths: {},
        filter_params: filterParams,
        output_format: outputFormatField.value,
        output_quality: parseInt(outputQualityField.value),
        output_lossless: outputLosslessField.checked,
    }
    filterReq.model_paths[filterName] = path

    if (saveToDiskField.checked && diskPathField.value.trim() !== "") {
        filterReq.save_to_disk_path = diskPathField.value.trim()
    }

    tools.spinnerStatus.innerText = statusText
    tools.spinner.classList.remove("displayNone")

    SD.filter(filterReq, (e) => {
        if (e.status === "succeeded") {
            let prevImg = img.src
            img.src = e.output[0]
            tools.spinner.classList.add("displayNone")

            if (prevImg.length > 0) {
                tools.undoBuffer.push(prevImg)
                tools.redoBuffer = []

                if (tools.undoBuffer.length > MAX_IMG_UNDO_ENTRIES) {
                    let n = tools.undoBuffer.length
                    tools.undoBuffer.splice(0, n - MAX_IMG_UNDO_ENTRIES)
                }

                tools.undoButton.classList.remove("displayNone")
                tools.redoButton.classList.add("displayNone")
            }
        } else if (e.status == "failed") {
            alert("Error running upscale: " + e.detail)
            tools.spinner.classList.add("displayNone")
        }
    })
}

function moveImageBetweenBuffers(img, fromBuffer, toBuffer, fromButton, toButton) {
    if (fromBuffer.length === 0) {
        return
    }

    let src = fromBuffer.pop()
    if (src.length > 0) {
        toBuffer.push(img.src)
        img.src = src
    }

    if (fromBuffer.length === 0) {
        fromButton.classList.add("displayNone")
    }
    if (toBuffer.length > 0) {
        toButton.classList.remove("displayNone")
    }
}

function onUndoFilter(req, img, e, tools) {
    moveImageBetweenBuffers(img, tools.undoBuffer, tools.redoBuffer, tools.undoButton, tools.redoButton)
}

function onRedoFilter(req, img, e, tools) {
    moveImageBetweenBuffers(img, tools.redoBuffer, tools.undoBuffer, tools.redoButton, tools.undoButton)
}

function onUpscaleClick(req, img, e, tools) {
    let path = upscaleModelField.value
    let scale = parseInt(upscaleAmountField.value)
    let filterName = path.toLowerCase().includes("realesrgan") ? "realesrgan" : "latent_upscaler"
    let statusText = "Upscaling by " + scale + "x using " + filterName
    applyInlineFilter(filterName, path, { scale: scale }, img, statusText, tools)
}

function onFixFacesClick(req, img, e, tools) {
    let path = gfpganModelField.value
    let filterName = path.toLowerCase().includes("gfpgan") ? "gfpgan" : "codeformer"
    let statusText = "Fixing faces with " + filterName
    applyInlineFilter(filterName, path, {}, img, statusText, tools)
}

function onContinueDrawingClick(req, img) {
    enqueueImageVariationTask(req, img, {
        num_inference_steps: parseInt(req.num_inference_steps) + 25,
    })
}

function makeImage() {
    if (typeof performance == "object" && performance.mark) {
        performance.mark("click-makeImage")
    }

    if (!SD.isServerAvailable()) {
        alert("The server is not available.")
        return
    }
    if (!randomSeedField.checked && seedField.value == "") {
        alert('The "Seed" field must not be empty.')
        seedField.classList.add("validation-failed")
        return
    }
    seedField.classList.remove("validation-failed")

    if (numInferenceStepsField.value == "") {
        alert('The "Inference Steps" field must not be empty.')
        numInferenceStepsField.classList.add("validation-failed")
        return
    }
    numInferenceStepsField.classList.remove("validation-failed")

    if (controlnetModelField.value === "" && IMAGE_REGEX.test(controlImagePreview.src)) {
        alert("Please choose a ControlNet model, to use the ControlNet image.")
        document.getElementById("controlnet_model").classList.add("validation-failed")
        return
    }
    document.getElementById("controlnet_model").classList.remove("validation-failed")

    if (numOutputsTotalField.value == "" || numOutputsTotalField.value == 0) {
        numOutputsTotalField.value = 1
    }
    if (numOutputsParallelField.value == "" || numOutputsParallelField.value == 0) {
        numOutputsParallelField.value = 1
    }
    if (guidanceScaleField.value == "") {
        guidanceScaleField.value = guidanceScaleSlider.value / 10
    }
    if (hypernetworkStrengthField.value == "") {
        hypernetworkStrengthField.value = hypernetworkStrengthSlider.value / 100
    }
    const taskTemplate = getCurrentUserRequest()
    const newTaskRequests = getPrompts().map((prompt) =>
        Object.assign({}, taskTemplate, {
            reqBody: Object.assign({ prompt: prompt }, taskTemplate.reqBody),
        })
    )
    newTaskRequests.forEach(setEmbeddings)
    newTaskRequests.forEach(createTask)

    updateInitialText()

    const countBeforeBanner = localStorage.getItem("countBeforeBanner") || 1
    if (countBeforeBanner <= 0) {
        // supportBanner.classList.remove("displayNone")
    } else {
        localStorage.setItem("countBeforeBanner", countBeforeBanner - 1)
    }
}

/* Hover effect for the init image in the task list */
function createInitImageHover(taskEntry, task) {
    taskEntry.querySelectorAll(".task-initimg").forEach((thumb) => {
        let thumbimg = thumb.querySelector("img")
        let img = createElement("img", { src: thumbimg.src })
        thumb.querySelector(".task-fs-initimage").appendChild(img)
        let div = createElement("div", undefined, ["top-right"])
        div.innerHTML = `
            <button class="useAsInputBtn">Use as Input</button>
            <br>
            <button class="useForControlnetBtn">Use for Controlnet</button>
            <br>
            <button class="downloadPreviewImg">Download</button>`
        div.querySelector(".useAsInputBtn").addEventListener("click", (e) => {
            e.preventDefault()
            onUseAsInputClick(null, img)
        })
        div.querySelector(".useForControlnetBtn").addEventListener("click", (e) => {
            e.preventDefault()
            controlImagePreview.src = img.src
        })
        div.querySelector(".downloadPreviewImg").addEventListener("click", (e) => {
            e.preventDefault()

            const name = "image." + task.reqBody["output_format"]
            const blob = dataURItoBlob(img.src)
            saveAs(blob, name)
        })
        thumb.querySelector(".task-fs-initimage").appendChild(div)
    })
    return

    var $tooltip = $(taskEntry.querySelector(".task-fs-initimage"))
    var img = document.createElement("img")
    img.src = taskEntry.querySelector("div.task-initimg > img").src
    $tooltip.append(img)
    $tooltip.append(`<div class="top-right"><button>Use as Input</button></div>`)
    $tooltip.find("button").on("click", (e) => {
        e.stopPropagation()
        onUseAsInputClick(null, img)
    })
}

let startX, startY
function onTaskEntryDragOver(event) {
    imagePreview.querySelectorAll(".imageTaskContainer").forEach((itc) => {
        if (itc != event.target.closest(".imageTaskContainer")) {
            itc.classList.remove("dropTargetBefore", "dropTargetAfter")
        }
    })
    if (event.target.closest(".imageTaskContainer")) {
        if (startX && startY) {
            if (event.target.closest(".imageTaskContainer").offsetTop > startY) {
                event.target.closest(".imageTaskContainer").classList.add("dropTargetAfter")
            } else if (event.target.closest(".imageTaskContainer").offsetTop < startY) {
                event.target.closest(".imageTaskContainer").classList.add("dropTargetBefore")
            } else if (event.target.closest(".imageTaskContainer").offsetLeft > startX) {
                event.target.closest(".imageTaskContainer").classList.add("dropTargetAfter")
            } else if (event.target.closest(".imageTaskContainer").offsetLeft < startX) {
                event.target.closest(".imageTaskContainer").classList.add("dropTargetBefore")
            }
        }
    }
}

function generateConfig({ label, value, visible, cssKey }) {
    if (!visible) return null
    return `<div class="taskConfigContainer task${cssKey}Container"><b>${label}:</b> <span class="task${cssKey}">${value}`
}

function getVisibleConfig(config, task) {
    const mergedTaskConfig = { ...config.taskConfig, ...config.pluginTaskConfig }
    return Object.keys(mergedTaskConfig)
        .map((key) => {
            const value = mergedTaskConfig?.[key]?.value?.(task) ?? task.reqBody[key]
            const visible = mergedTaskConfig?.[key]?.visible?.(task) ?? value !== undefined ?? true
            const label = mergedTaskConfig?.[key]?.label ?? mergedTaskConfig?.[key]
            const cssKey = config.getCSSKey(key)
            return { label, visible, value, cssKey }
        })
        .map((obj) => generateConfig(obj))
        .filter((obj) => obj)
}

function createTaskConfig(task) {
    return getVisibleConfig(taskConfigSetup, task).join("</span>,&nbsp;</div>")
}

function createTask(task) {
    let taskConfig = ""

    if (task.reqBody.init_image !== undefined) {
        let h = 80
        let w = ((task.reqBody.width * h) / task.reqBody.height) >> 0
        taskConfig += `<div class="task-initimg init-img-preview" style="float:left;"><img style="width:${w}px;height:${h}px;" src="${task.reqBody.init_image}"><div class="task-fs-initimage"></div></div>`
    }
    if (task.reqBody.control_image !== undefined) {
        let h = 80
        let w = ((task.reqBody.width * h) / task.reqBody.height) >> 0
        taskConfig += `<div class="task-initimg controlnet-img-preview" style="float:left;"><img style="width:${w}px;height:${h}px;" src="${task.reqBody.control_image}"><div class="task-fs-initimage"></div></div>`
    }

    taskConfig += `<div class="taskConfigData">${createTaskConfig(task)}</span></div></div>`

    let taskEntry = document.createElement("div")
    taskEntry.id = `imageTaskContainer-${Date.now()}`
    taskEntry.className = "imageTaskContainer"
    taskEntry.innerHTML = ` <div class="header-content panel collapsible active">
                                <i class="drag-handle fa-solid fa-grip"></i>
                                <div class="taskStatusLabel">Enqueued</div>
                                <button class="secondaryButton stopTask"><i class="fa-solid fa-xmark"></i> Cancel</button>
                                <button class="tertiaryButton useSettings"><i class="fa-solid fa-redo"></i> Use these settings</button>
                                <div class="preview-prompt"></div>
                                <div class="taskConfig">${taskConfig}</div>
                                <div class="outputMsg"></div>
                                <div class="progress-bar active"><div></div></div>
                            </div>
                            <div class="collapsible-content">
                                <div class="img-preview">
                            </div>`

    if (task.reqBody.init_image !== undefined || task.reqBody.control_image !== undefined) {
        createInitImageHover(taskEntry, task)
    }

    if (task.reqBody.control_image !== undefined && task.reqBody.control_filter_to_apply !== undefined) {
        let req = {
            image: task.reqBody.control_image,
            filter: task.reqBody.control_filter_to_apply,
            model_paths: {},
            filter_params: {},
        }
        req["model_paths"][task.reqBody.control_filter_to_apply] = task.reqBody.control_filter_to_apply

        task["previewTaskReq"] = req
    }

    createCollapsibles(taskEntry)

    let draghandle = taskEntry.querySelector(".drag-handle")
    draghandle.addEventListener("mousedown", (e) => {
        taskEntry.setAttribute("draggable", true)
    })
    // Add a debounce delay to allow mobile to bouble tap.
    draghandle.addEventListener(
        "mouseup",
        debounce((e) => {
            taskEntry.setAttribute("draggable", false)
        }, 2000)
    )
    draghandle.addEventListener("click", (e) => {
        e.preventDefault() // Don't allow the results to be collapsed...
    })
    taskEntry.addEventListener("dragend", (e) => {
        taskEntry.setAttribute("draggable", false)
        imagePreview.querySelectorAll(".imageTaskContainer").forEach((itc) => {
            itc.classList.remove("dropTargetBefore", "dropTargetAfter")
        })
        imagePreview.removeEventListener("dragover", onTaskEntryDragOver)
    })
    taskEntry.addEventListener("dragstart", function(e) {
        imagePreview.addEventListener("dragover", onTaskEntryDragOver)
        e.dataTransfer.setData("text/plain", taskEntry.id)
        startX = e.target.closest(".imageTaskContainer").offsetLeft
        startY = e.target.closest(".imageTaskContainer").offsetTop
    })

    task["taskConfig"] = taskEntry.querySelector(".taskConfig")
    task["taskStatusLabel"] = taskEntry.querySelector(".taskStatusLabel")
    task["outputContainer"] = taskEntry.querySelector(".img-preview")
    task["outputMsg"] = taskEntry.querySelector(".outputMsg")
    task["previewPrompt"] = taskEntry.querySelector(".preview-prompt")
    task["progressBar"] = taskEntry.querySelector(".progress-bar")
    task["stopTask"] = taskEntry.querySelector(".stopTask")

    task["stopTask"].addEventListener("click", (e) => {
        e.stopPropagation()

        if (task["isProcessing"]) {
            shiftOrConfirm(e, "Stop this task?", async function(e) {
                if (task.batchesDone <= 0 || !task.isProcessing) {
                    removeTask(taskEntry)
                }
                abortTask(task)
            })
        } else {
            removeTask(taskEntry)
        }
    })

    task["useSettings"] = taskEntry.querySelector(".useSettings")
    task["useSettings"].addEventListener("click", function(e) {
        e.stopPropagation()
        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
    })

    task.isProcessing = true
    taskEntry = imagePreviewContent.insertBefore(taskEntry, supportBanner.nextSibling)
    htmlTaskMap.set(taskEntry, task)

    task.previewPrompt.innerText = task.reqBody.prompt
    if (task.previewPrompt.innerText.trim() === "") {
        task.previewPrompt.innerHTML = "&nbsp;" // allows the results to be collapsed
    }
    return taskEntry.id
}

function getCurrentUserRequest() {
    const numOutputsTotal = parseInt(numOutputsTotalField.value)
    let numOutputsParallel = parseInt(numOutputsParallelField.value)
    const seed = randomSeedField.checked ? Math.floor(Math.random() * (2 ** 32 - 1)) : parseInt(seedField.value)

    // if (
    //     testDiffusers.checked &&
    //     document.getElementById("toggle-tensorrt-install").innerHTML == "Uninstall" &&
    //     document.querySelector("#convert_to_tensorrt").checked
    // ) {
    //     // TRT enabled

    //     numOutputsParallel = 1 // force 1 parallel
    // }

    // clamp to multiple of 8
    let width = parseInt(widthField.value)
    let height = parseInt(heightField.value)
    width = width - (width % IMAGE_STEP_SIZE)
    height = height - (height % IMAGE_STEP_SIZE)

    const newTask = {
        batchesDone: 0,
        numOutputsTotal: numOutputsTotal,
        batchCount: Math.ceil(numOutputsTotal / numOutputsParallel),
        seed,
        reqBody: {
            seed,
            used_random_seed: randomSeedField.checked,
            negative_prompt: negativePromptField.value.trim(),
            num_outputs: numOutputsParallel,
            num_inference_steps: parseInt(numInferenceStepsField.value),
            guidance_scale: parseFloat(guidanceScaleField.value),
            width: width,
            height: height,
            // allow_nsfw: allowNSFWField.checked,
            vram_usage_level: vramUsageLevelField.value,
            sampler_name: samplerField.value,
            //render_device: undefined, // Set device affinity. Prefer this device, but wont activate.
            use_stable_diffusion_model: stableDiffusionModelField.value,
            clip_skip: clipSkipField.checked,
            use_vae_model: vaeModelField.value,
            stream_progress_updates: true,
            stream_image_progress: numOutputsTotal > 50 ? false : streamImageProgressField.checked,
            show_only_filtered_image: showOnlyFilteredImageField.checked,
            block_nsfw: blockNSFWField.checked,
            output_format: outputFormatField.value,
            output_quality: parseInt(outputQualityField.value),
            output_lossless: outputLosslessField.checked,
            metadata_output_format: metadataOutputFormatField.value,
            original_prompt: promptField.value,
            active_tags: activeTags.map((x) => x.name),
            inactive_tags: activeTags.filter((tag) => tag.inactive === true).map((x) => x.name),
        },
    }
    if (IMAGE_REGEX.test(initImagePreview.src)) {
        newTask.reqBody.init_image = initImagePreview.src
        newTask.reqBody.prompt_strength = parseFloat(promptStrengthField.value)
        // if (IMAGE_REGEX.test(maskImagePreview.src)) {
        //     newTask.reqBody.mask = maskImagePreview.src
        // }
        if (maskSetting.checked) {
            newTask.reqBody.mask = imageInpainter.getImg()
            newTask.reqBody.strict_mask_border = strictMaskBorderField.checked
        }
        newTask.reqBody.preserve_init_image_color_profile = applyColorCorrectionField.checked
        if (!testDiffusers.checked) {
            newTask.reqBody.sampler_name = "ddim"
        }
    }
    if (saveToDiskField.checked && diskPathField.value.trim() !== "") {
        newTask.reqBody.save_to_disk_path = diskPathField.value.trim()
    }
    if (useFaceCorrectionField.checked) {
        newTask.reqBody.use_face_correction = gfpganModelField.value

        if (gfpganModelField.value.includes("codeformer")) {
            newTask.reqBody.codeformer_upscale_faces = document.querySelector("#codeformer_upscale_faces").checked
            newTask.reqBody.codeformer_fidelity = 1 - parseFloat(codeformerFidelityField.value)
        }
    }
    if (useUpscalingField.checked) {
        newTask.reqBody.use_upscale = upscaleModelField.value
        newTask.reqBody.upscale_amount = upscaleAmountField.value
        if (upscaleModelField.value === "latent_upscaler") {
            newTask.reqBody.upscale_amount = "2"
            newTask.reqBody.latent_upscaler_steps = latentUpscalerStepsField.value
        }
    }
    if (hypernetworkModelField.value) {
        newTask.reqBody.use_hypernetwork_model = hypernetworkModelField.value
        newTask.reqBody.hypernetwork_strength = parseFloat(hypernetworkStrengthField.value)
    }
    if (testDiffusers.checked) {
        let loraModelData = loraModelField.value
        let modelNames = loraModelData["modelNames"]
        let modelStrengths = loraModelData["modelWeights"]

        if (modelNames.length > 0) {
            modelNames = modelNames.length == 1 ? modelNames[0] : modelNames
            modelStrengths = modelStrengths.length == 1 ? modelStrengths[0] : modelStrengths

            newTask.reqBody.use_lora_model = modelNames
            newTask.reqBody.lora_alpha = modelStrengths
        }

        if (tilingField.value !== "none") {
            newTask.reqBody.tiling = tilingField.value
        }
    }
    if (testDiffusers.checked && document.getElementById("toggle-tensorrt-install").innerHTML == "Uninstall") {
        // TRT is installed
        newTask.reqBody.convert_to_tensorrt = document.querySelector("#convert_to_tensorrt").checked
        let trtBuildConfig = {
            batch_size_range: [
                parseInt(document.querySelector("#trt-build-min-batch").value),
                parseInt(document.querySelector("#trt-build-max-batch").value),
            ],
            dimensions_range: [],
        }

        let sizes = [512, 768, 1024, 1280, 1536]
        sizes.forEach((i) => {
            let el = document.querySelector("#trt-build-res-" + i)
            if (el.checked) {
                trtBuildConfig["dimensions_range"].push([i, i + 256])
            }
        })
        newTask.reqBody.trt_build_config = trtBuildConfig
    }
    if (controlnetModelField.value !== "" && IMAGE_REGEX.test(controlImagePreview.src)) {
        newTask.reqBody.use_controlnet_model = controlnetModelField.value
        newTask.reqBody.control_image = controlImagePreview.src
        if (controlImageFilterField.value !== "") {
            newTask.reqBody.control_filter_to_apply = controlImageFilterField.value
        }
    }

    return newTask
}

function setEmbeddings(task) {
    let prompt = task.reqBody.prompt
    let negativePrompt = task.reqBody.negative_prompt
    let overallPrompt = (prompt + " " + negativePrompt).toLowerCase()
    overallPrompt = overallPrompt.replaceAll(/[^a-z0-9\-_\.]/g, " ") // only allow alpha-numeric, dots and hyphens
    overallPrompt = overallPrompt.split(" ")

    let embeddingsTree = modelsOptions["embeddings"]
    let embeddings = []
    function extract(entries, basePath = "") {
        entries.forEach((e) => {
            if (Array.isArray(e)) {
                let path = basePath === "" ? basePath + e[0] : basePath + "/" + e[0]
                extract(e[1], path)
            } else {
                let path = basePath === "" ? basePath + e : basePath + "/" + e
                embeddings.push([e.toLowerCase().replace(" ", "_"), path])
            }
        })
    }
    extract(embeddingsTree)

    let embeddingPaths = []

    embeddings.forEach((e) => {
        let token = e[0]
        let path = e[1]

        if (overallPrompt.includes(token)) {
            embeddingPaths.push(path)
        }
    })

    if (embeddingPaths.length > 0) {
        task.reqBody.use_embeddings_model = embeddingPaths
    }
}

function getPrompts(prompts) {
    if (typeof prompts === "undefined") {
        prompts = promptField.value
    }
    if (prompts.trim() === "" && activeTags.length === 0) {
        return [""]
    }

    let promptsToMake = []
    if (prompts.trim() !== "") {
        prompts = prompts.split("\n")
        prompts = prompts.map((prompt) => prompt.trim())
        prompts = prompts.filter((prompt) => prompt !== "")

        promptsToMake = applyPermuteOperator(prompts)
        promptsToMake = applySetOperator(promptsToMake)
    }
    const newTags = activeTags.filter((tag) => tag.inactive === undefined || tag.inactive === false)
    if (newTags.length > 0) {
        const promptTags = newTags.map((x) => x.name).join(", ")
        if (promptsToMake.length > 0) {
            promptsToMake = promptsToMake.map((prompt) => `${prompt}, ${promptTags}`)
        } else {
            promptsToMake.push(promptTags)
        }
    }

    promptsToMake = applyPermuteOperator(promptsToMake)
    promptsToMake = applySetOperator(promptsToMake)

    PLUGINS["GET_PROMPTS_HOOK"].forEach((fn) => {
        promptsToMake = fn(promptsToMake)
    })

    return promptsToMake
}

function getPromptsNumber(prompts) {
    if (typeof prompts === "undefined") {
        prompts = promptField.value
    }
    if (prompts.trim() === "" && activeTags.length === 0) {
        return [""]
    }

    let promptsToMake = []
    let numberOfPrompts = 0
    if (prompts.trim() !== "") {
        // this needs to stay sort of the same, as the prompts have to be passed through to the other functions
        prompts = prompts.split("\n")
        prompts = prompts.map((prompt) => prompt.trim())
        prompts = prompts.filter((prompt) => prompt !== "")

        // estimate number of prompts
        let estimatedNumberOfPrompts = 0
        prompts.forEach((prompt) => {
            estimatedNumberOfPrompts +=
                (prompt.match(/{[^}]*}/g) || [])
                    .map((e) => (e.match(/,/g) || []).length + 1)
                    .reduce((p, a) => p * a, 1) *
                2 ** (prompt.match(/\|/g) || []).length
        })

        if (estimatedNumberOfPrompts >= 10000) {
            return 10000
        }

        promptsToMake = applySetOperator(prompts) // switched those around as Set grows in a linear fashion and permute in 2^n, and one has to be computed for the other to be calculated
        numberOfPrompts = applyPermuteOperatorNumber(promptsToMake)
    }
    const newTags = activeTags.filter((tag) => tag.inactive === undefined || tag.inactive === false)
    if (newTags.length > 0) {
        const promptTags = newTags.map((x) => x.name).join(", ")
        if (numberOfPrompts > 0) {
            // promptsToMake = promptsToMake.map((prompt) => `${prompt}, ${promptTags}`)
            // nothing changes, as all prompts just get modified
        } else {
            // promptsToMake.push(promptTags)
            numberOfPrompts = 1
        }
    }

    // Why is this applied twice? It does not do anything here, as everything should have already been done earlier
    // promptsToMake = applyPermuteOperator(promptsToMake)
    // promptsToMake = applySetOperator(promptsToMake)

    return numberOfPrompts
}

function applySetOperator(prompts) {
    let promptsToMake = []
    let braceExpander = new BraceExpander()
    prompts.forEach((prompt) => {
        let expandedPrompts = braceExpander.expand(prompt)
        promptsToMake = promptsToMake.concat(expandedPrompts)
    })

    return promptsToMake
}

function applyPermuteOperator(prompts) {
    // prompts is array of input, trimmed, filtered and split by \n
    let promptsToMake = []
    prompts.forEach((prompt) => {
        let promptMatrix = prompt.split("|")
        prompt = promptMatrix.shift().trim()
        promptsToMake.push(prompt)

        promptMatrix = promptMatrix.map((p) => p.trim())
        promptMatrix = promptMatrix.filter((p) => p !== "")

        if (promptMatrix.length > 0) {
            let promptPermutations = permutePrompts(prompt, promptMatrix)
            promptsToMake = promptsToMake.concat(promptPermutations)
        }
    })

    return promptsToMake
}

// returns how many prompts would have to be made with the given prompts
function applyPermuteOperatorNumber(prompts) {
    // prompts is array of input, trimmed, filtered and split by \n
    let numberOfPrompts = 0
    prompts.forEach((prompt) => {
        let promptCounter = 1
        let promptMatrix = prompt.split("|")
        promptMatrix.shift()

        promptMatrix = promptMatrix.map((p) => p.trim())
        promptMatrix = promptMatrix.filter((p) => p !== "")

        if (promptMatrix.length > 0) {
            promptCounter *= permuteNumber(promptMatrix)
        }
        numberOfPrompts += promptCounter
    })

    return numberOfPrompts
}

function permutePrompts(promptBase, promptMatrix) {
    let prompts = []
    let permutations = permute(promptMatrix)
    permutations.forEach((perm) => {
        let prompt = promptBase

        if (perm.length > 0) {
            let promptAddition = perm.join(", ")
            if (promptAddition.trim() === "") {
                return
            }

            prompt += ", " + promptAddition
        }

        prompts.push(prompt)
    })

    return prompts
}

// create a file name with embedded prompt and metadata
// for easier cateloging and comparison
function createFileName(prompt, seed, steps, guidance, outputFormat) {
    // Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, "_")
    underscoreName = underscoreName.substring(0, 70)

    // name and the top level metadata
    let fileName = `${underscoreName}_S${seed}_St${steps}_G${guidance}.${outputFormat}`

    return fileName
}

function updateInitialText() {
    if (document.querySelector(".imageTaskContainer") === null) {
        if (undoBuffer.length > 0) {
            initialText.prepend(undoButton)
        }
        previewTools.classList.add("displayNone")
        initialText.classList.remove("displayNone")
        supportBanner.classList.add("displayNone")
    } else {
        initialText.classList.add("displayNone")
        previewTools.classList.remove("displayNone")
        document.querySelector("div.display-settings").prepend(undoButton)

        const countBeforeBanner = localStorage.getItem("countBeforeBanner") || 1
        if (countBeforeBanner <= 0) {
            supportBanner.classList.remove("displayNone")
        }
    }
}

function removeTask(taskToRemove) {
    undoableRemove(taskToRemove)
    updateInitialText()
}

clearAllPreviewsBtn.addEventListener("click", (e) => {
    shiftOrConfirm(e, "Clear all the results and tasks in this window?", async function() {
        await stopAllTasks()

        let taskEntries = document.querySelectorAll(".imageTaskContainer")
        taskEntries.forEach(removeTask)
    })
})

/* Download images popup */
showDownloadDialogBtn.addEventListener("click", (e) => {
    saveAllImagesDialog.showModal()
})
saveAllImagesCloseBtn.addEventListener("click", (e) => {
    saveAllImagesDialog.close()
})
modalDialogCloseOnBackdropClick(saveAllImagesDialog)
makeDialogDraggable(saveAllImagesDialog)

saveAllZipToggle.addEventListener("change", (e) => {
    if (saveAllZipToggle.checked) {
        saveAllFoldersOption.classList.remove("displayNone")
    } else {
        saveAllFoldersOption.classList.add("displayNone")
    }
})

// convert base64 to raw binary data held in a string
function dataURItoBlob(dataURI) {
    var byteString = atob(dataURI.split(",")[1])

    // separate out the mime component
    var mimeString = dataURI
        .split(",")[0]
        .split(":")[1]
        .split(";")[0]

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length)

    // create a view into the buffer
    var ia = new Uint8Array(ab)

    // set the bytes of the buffer to the correct values
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i)
    }

    // write the ArrayBuffer to a blob, and you're done
    return new Blob([ab], { type: mimeString })
}

function downloadAllImages() {
    let i = 0

    let optZIP = saveAllZipToggle.checked
    let optTree = optZIP && saveAllTreeToggle.checked
    let optJSON = saveAllJSONToggle.checked

    let zip = new JSZip()
    let folder = zip

    document.querySelectorAll(".imageTaskContainer").forEach((container) => {
        if (optTree) {
            let name =
                ++i +
                "-" +
                container
                    .querySelector(".preview-prompt")
                    .textContent.replace(/[^a-zA-Z0-9]/g, "_")
                    .substring(0, 25)
            folder = zip.folder(name)
        }
        container.querySelectorAll(".imgContainer img").forEach((img) => {
            let imgItem = img.closest(".imgItem")

            if (imgItem.style.display === "none") {
                return
            }

            let req = imageRequest[img.dataset["imagecounter"]]
            if (optZIP) {
                let suffix = img.dataset["imagecounter"] + "." + req["output_format"]
                folder.file(getDownloadFilename(img, suffix), dataURItoBlob(img.src))
                if (optJSON) {
                    suffix = img.dataset["imagecounter"] + ".json"
                    folder.file(getDownloadFilename(img, suffix), JSON.stringify(req, null, 2))
                }
            } else {
                setTimeout(() => {
                    imgItem.querySelector(".download-img").click()
                }, i * 200)
                i = i + 1
                if (optJSON) {
                    setTimeout(() => {
                        imgItem.querySelector(".download-json").click()
                    }, i * 200)
                    i = i + 1
                }
            }
        })
    })
    if (optZIP) {
        let now = Date.now()
            .toString(36)
            .toUpperCase()
        zip.generateAsync({ type: "blob" }).then(function(blob) {
            saveAs(blob, `EasyDiffusion-Images-${now}.zip`)
        })
    }
}

saveAllImagesBtn.addEventListener("click", (e) => {
    downloadAllImages()
})

stopImageBtn.addEventListener("click", (e) => {
    shiftOrConfirm(e, "Stop all the tasks?", async function(e) {
        await stopAllTasks()
    })
})

widthField.addEventListener("change", onDimensionChange)
heightField.addEventListener("change", onDimensionChange)

function renameMakeImageButton() {
    let totalImages =
        Math.max(parseInt(numOutputsTotalField.value), parseInt(numOutputsParallelField.value)) * getPromptsNumber()
    let imageLabel = "Image"
    if (totalImages > 1) {
        imageLabel = totalImages + " Images"
    }
    if (SD.activeTasks.size == 0) {
        if (totalImages >= 10000) makeImageBtn.innerText = "Make 10000+ images"
        else makeImageBtn.innerText = "Make " + imageLabel
    } else {
        if (totalImages >= 10000) makeImageBtn.innerText = "Enqueue 10000+ images"
        else makeImageBtn.innerText = "Enqueue Next " + imageLabel
    }
}
numOutputsTotalField.addEventListener("change", renameMakeImageButton)
numOutputsTotalField.addEventListener("keyup", debounce(renameMakeImageButton, 300))
numOutputsParallelField.addEventListener("change", renameMakeImageButton)
numOutputsParallelField.addEventListener("keyup", debounce(renameMakeImageButton, 300))

function onDimensionChange() {
    let widthValue = parseInt(widthField.value)
    let heightValue = parseInt(heightField.value)
    if (!initImagePreviewContainer.classList.contains("has-image")) {
        imageEditor.setImage(null, widthValue, heightValue)
    } else {
        imageInpainter.setImage(initImagePreview.src, widthValue, heightValue)
    }
    if (widthValue < 512 && heightValue < 512) {
        smallImageWarning.classList.remove("displayNone")
    } else {
        smallImageWarning.classList.add("displayNone")
    }
}

diskPathField.disabled = !saveToDiskField.checked
metadataOutputFormatField.disabled = !saveToDiskField.checked

gfpganModelField.disabled = !useFaceCorrectionField.checked
useFaceCorrectionField.addEventListener("change", function(e) {
    gfpganModelField.disabled = !this.checked

    onFixFaceModelChange()
})

function onFixFaceModelChange() {
    let codeformerSettings = document.querySelector("#codeformer_settings")
    if (gfpganModelField.value === "codeformer" && !gfpganModelField.disabled) {
        codeformerSettings.classList.remove("displayNone")
        codeformerSettings.classList.add("expandedSettingRow")
    } else {
        codeformerSettings.classList.add("displayNone")
        codeformerSettings.classList.remove("expandedSettingRow")
    }
}
gfpganModelField.addEventListener("change", onFixFaceModelChange)
onFixFaceModelChange()

function onControlnetModelChange() {
    let configBox = document.querySelector("#controlnet_config")
    if (IMAGE_REGEX.test(controlImagePreview.src)) {
        configBox.classList.remove("displayNone")
        controlImageContainer.classList.remove("displayNone")
    } else {
        configBox.classList.add("displayNone")
        controlImageContainer.classList.add("displayNone")
    }
}
controlImagePreview.addEventListener("load", onControlnetModelChange)
controlImagePreview.addEventListener("unload", onControlnetModelChange)
onControlnetModelChange()

function onControlImageFilterChange() {
    let filterId = controlImageFilterField.value
    if (filterId.includes("openpose")) {
        controlnetModelField.value = "control_v11p_sd15_openpose"
    } else if (filterId === "canny") {
        controlnetModelField.value = "control_v11p_sd15_canny"
    } else if (filterId === "mlsd") {
        controlnetModelField.value = "control_v11p_sd15_mlsd"
    } else if (filterId === "mlsd") {
        controlnetModelField.value = "control_v11p_sd15_mlsd"
    } else if (filterId.includes("scribble")) {
        controlnetModelField.value = "control_v11p_sd15_scribble"
    } else if (filterId.includes("softedge")) {
        controlnetModelField.value = "control_v11p_sd15_softedge"
    } else if (filterId === "normal_bae") {
        controlnetModelField.value = "control_v11p_sd15_normalbae"
    } else if (filterId.includes("depth")) {
        controlnetModelField.value = "control_v11f1p_sd15_depth"
    } else if (filterId === "lineart_anime") {
        controlnetModelField.value = "control_v11p_sd15s2_lineart_anime"
    } else if (filterId.includes("lineart")) {
        controlnetModelField.value = "control_v11p_sd15_lineart"
    } else if (filterId === "shuffle") {
        controlnetModelField.value = "control_v11e_sd15_shuffle"
    } else if (filterId === "segment") {
        controlnetModelField.value = "control_v11p_sd15_seg"
    }
}
controlImageFilterField.addEventListener("change", onControlImageFilterChange)
onControlImageFilterChange()

upscaleModelField.disabled = !useUpscalingField.checked
upscaleAmountField.disabled = !useUpscalingField.checked
useUpscalingField.addEventListener("change", function(e) {
    upscaleModelField.disabled = !this.checked
    upscaleAmountField.disabled = !this.checked

    onUpscaleModelChange()
})

function onUpscaleModelChange() {
    let upscale4x = document.querySelector("#upscale_amount_4x")
    if (upscaleModelField.value === "latent_upscaler" && !upscaleModelField.disabled) {
        upscale4x.disabled = true
        upscaleAmountField.value = "2"
        latentUpscalerSettings.classList.remove("displayNone")
        latentUpscalerSettings.classList.add("expandedSettingRow")
    } else {
        upscale4x.disabled = false
        latentUpscalerSettings.classList.add("displayNone")
        latentUpscalerSettings.classList.remove("expandedSettingRow")
    }
}
upscaleModelField.addEventListener("change", onUpscaleModelChange)
onUpscaleModelChange()

makeImageBtn.addEventListener("click", makeImage)

document.onkeydown = function(e) {
    if (e.ctrlKey && e.code === "Enter") {
        makeImage()
        e.preventDefault()
    }
}

/********************* CodeFormer Fidelity **************************/
function updateCodeformerFidelity() {
    codeformerFidelityField.value = codeformerFidelitySlider.value / 10
    codeformerFidelityField.dispatchEvent(new Event("change"))
}

function updateCodeformerFidelitySlider() {
    if (codeformerFidelityField.value < 0) {
        codeformerFidelityField.value = 0
    } else if (codeformerFidelityField.value > 1) {
        codeformerFidelityField.value = 1
    }

    codeformerFidelitySlider.value = codeformerFidelityField.value * 10
    codeformerFidelitySlider.dispatchEvent(new Event("change"))
}

codeformerFidelitySlider.addEventListener("input", updateCodeformerFidelity)
codeformerFidelityField.addEventListener("input", updateCodeformerFidelitySlider)
updateCodeformerFidelity()

/********************* Latent Upscaler Steps **************************/
function updateLatentUpscalerSteps() {
    latentUpscalerStepsField.value = latentUpscalerStepsSlider.value
    latentUpscalerStepsField.dispatchEvent(new Event("change"))
}

function updateLatentUpscalerStepsSlider() {
    if (latentUpscalerStepsField.value < 1) {
        latentUpscalerStepsField.value = 1
    } else if (latentUpscalerStepsField.value > 50) {
        latentUpscalerStepsField.value = 50
    }

    latentUpscalerStepsSlider.value = latentUpscalerStepsField.value
    latentUpscalerStepsSlider.dispatchEvent(new Event("change"))
}

latentUpscalerStepsSlider.addEventListener("input", updateLatentUpscalerSteps)
latentUpscalerStepsField.addEventListener("input", updateLatentUpscalerStepsSlider)
updateLatentUpscalerSteps()

/********************* Guidance **************************/
function updateGuidanceScale() {
    guidanceScaleField.value = guidanceScaleSlider.value / 10
    guidanceScaleField.dispatchEvent(new Event("change"))
}

function updateGuidanceScaleSlider() {
    if (guidanceScaleField.value < 0) {
        guidanceScaleField.value = 0
    } else if (guidanceScaleField.value > 50) {
        guidanceScaleField.value = 50
    }

    guidanceScaleSlider.value = guidanceScaleField.value * 10
    guidanceScaleSlider.dispatchEvent(new Event("change"))
}

guidanceScaleSlider.addEventListener("input", updateGuidanceScale)
guidanceScaleField.addEventListener("input", updateGuidanceScaleSlider)
updateGuidanceScale()

/********************* Prompt Strength *******************/
function updatePromptStrength() {
    promptStrengthField.value = promptStrengthSlider.value / 100
    promptStrengthField.dispatchEvent(new Event("change"))
}

function updatePromptStrengthSlider() {
    if (promptStrengthField.value < 0) {
        promptStrengthField.value = 0
    } else if (promptStrengthField.value > 0.99) {
        promptStrengthField.value = 0.99
    }

    promptStrengthSlider.value = promptStrengthField.value * 100
    promptStrengthSlider.dispatchEvent(new Event("change"))
}

promptStrengthSlider.addEventListener("input", updatePromptStrength)
promptStrengthField.addEventListener("input", updatePromptStrengthSlider)
updatePromptStrength()

/********************* Hypernetwork Strength **********************/
function updateHypernetworkStrength() {
    hypernetworkStrengthField.value = hypernetworkStrengthSlider.value / 100
    hypernetworkStrengthField.dispatchEvent(new Event("change"))
}

function updateHypernetworkStrengthSlider() {
    if (hypernetworkStrengthField.value < 0) {
        hypernetworkStrengthField.value = 0
    } else if (hypernetworkStrengthField.value > 0.99) {
        hypernetworkStrengthField.value = 0.99
    }

    hypernetworkStrengthSlider.value = hypernetworkStrengthField.value * 100
    hypernetworkStrengthSlider.dispatchEvent(new Event("change"))
}

hypernetworkStrengthSlider.addEventListener("input", updateHypernetworkStrength)
hypernetworkStrengthField.addEventListener("input", updateHypernetworkStrengthSlider)
updateHypernetworkStrength()

function updateHypernetworkStrengthContainer() {
    document.querySelector("#hypernetwork_strength_container").style.display =
        hypernetworkModelField.value === "" ? "none" : ""
}
hypernetworkModelField.addEventListener("change", updateHypernetworkStrengthContainer)
updateHypernetworkStrengthContainer()

/********************* JPEG/WEBP Quality **********************/
function updateOutputQuality() {
    outputQualityField.value = 0 | outputQualitySlider.value
    outputQualityField.dispatchEvent(new Event("change"))
}

function updateOutputQualitySlider() {
    if (outputQualityField.value < 10) {
        outputQualityField.value = 10
    } else if (outputQualityField.value > 95) {
        outputQualityField.value = 95
    }

    outputQualitySlider.value = 0 | outputQualityField.value
    outputQualitySlider.dispatchEvent(new Event("change"))
}

outputQualitySlider.addEventListener("input", updateOutputQuality)
outputQualityField.addEventListener("input", debounce(updateOutputQualitySlider, 1500))
updateOutputQuality()

function updateOutputQualityVisibility() {
    if (outputFormatField.value === "webp") {
        outputLosslessContainer.classList.remove("displayNone")
        if (outputLosslessField.checked) {
            outputQualityRow.classList.add("displayNone")
        } else {
            outputQualityRow.classList.remove("displayNone")
        }
    } else if (outputFormatField.value === "png") {
        outputQualityRow.classList.add("displayNone")
        outputLosslessContainer.classList.add("displayNone")
    } else {
        outputQualityRow.classList.remove("displayNone")
        outputLosslessContainer.classList.add("displayNone")
    }
}

outputFormatField.addEventListener("change", updateOutputQualityVisibility)
outputLosslessField.addEventListener("change", updateOutputQualityVisibility)
/********************* Zoom Slider **********************/
thumbnailSizeField.addEventListener("change", () => {
    ;(function(s) {
        for (var j = 0; j < document.styleSheets.length; j++) {
            let cssSheet = document.styleSheets[j]
            for (var i = 0; i < cssSheet.cssRules.length; i++) {
                var rule = cssSheet.cssRules[i]
                if (rule.selectorText == "div.img-preview img") {
                    rule.style["max-height"] = s + "vh"
                    rule.style["max-width"] = s + "vw"
                    return
                }
            }
        }
    })(thumbnailSizeField.value)
})

function onAutoScrollUpdate() {
    if (autoScroll.checked) {
        autoscrollBtn.classList.add("pressed")
    } else {
        autoscrollBtn.classList.remove("pressed")
    }
    autoscrollBtn.querySelector(".state").innerHTML = autoScroll.checked ? "ON" : "OFF"
}
autoscrollBtn.addEventListener("click", function() {
    autoScroll.checked = !autoScroll.checked
    autoScroll.dispatchEvent(new Event("change"))
    onAutoScrollUpdate()
})
autoScroll.addEventListener("change", onAutoScrollUpdate)

function checkRandomSeed() {
    if (randomSeedField.checked) {
        seedField.disabled = true
        //seedField.value = "0" // This causes the seed to be lost if the user changes their mind after toggling the checkbox
    } else {
        seedField.disabled = false
    }
}
randomSeedField.addEventListener("input", checkRandomSeed)
checkRandomSeed()

// warning: the core plugin `image-editor-improvements.js:172` replaces loadImg2ImgFromFile() with a custom version
function loadImg2ImgFromFile() {
    if (initImageSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = initImageSelector.files[0]

    reader.addEventListener("load", function(event) {
        initImagePreview.src = reader.result
    })

    if (file) {
        reader.readAsDataURL(file)
    }
}
initImageSelector.addEventListener("change", loadImg2ImgFromFile)
loadImg2ImgFromFile()

function img2imgLoad() {
    promptStrengthContainer.style.display = "table-row"
    if (!testDiffusers.checked) {
        samplerSelectionContainer.style.display = "none"
    }
    initImagePreviewContainer.classList.add("has-image")
    colorCorrectionSetting.style.display = ""
    strictMaskBorderSetting.style.display = maskSetting.checked ? "" : "none"

    initImageSizeBox.textContent = initImagePreview.naturalWidth + " x " + initImagePreview.naturalHeight
    imageEditor.setImage(this.src, initImagePreview.naturalWidth, initImagePreview.naturalHeight)
    imageInpainter.setImage(this.src, parseInt(widthField.value), parseInt(heightField.value))
}

function img2imgUnload() {
    initImageSelector.value = null
    initImagePreview.src = ""
    maskSetting.checked = false

    promptStrengthContainer.style.display = "none"
    if (!testDiffusers.checked) {
        samplerSelectionContainer.style.display = ""
    }
    initImagePreviewContainer.classList.remove("has-image")
    colorCorrectionSetting.style.display = "none"
    strictMaskBorderSetting.style.display = "none"
    imageEditor.setImage(null, parseInt(widthField.value), parseInt(heightField.value))
}
initImagePreview.addEventListener("load", img2imgLoad)
initImageClearBtn.addEventListener("click", img2imgUnload)

maskSetting.addEventListener("click", function() {
    onDimensionChange()
})
maskSetting.addEventListener("change", function() {
    strictMaskBorderSetting.style.display = this.checked ? "" : "none"
})

promptsFromFileBtn.addEventListener("click", function() {
    promptsFromFileSelector.click()
})

function loadControlnetImageFromFile() {
    if (controlImageSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = controlImageSelector.files[0]

    reader.addEventListener("load", function(event) {
        controlImagePreview.src = reader.result
    })

    if (file) {
        reader.readAsDataURL(file)
    }
}
controlImageSelector.addEventListener("change", loadControlnetImageFromFile)

function controlImageLoad() {
    let w = controlImagePreview.naturalWidth
    let h = controlImagePreview.naturalHeight
    w = w - (w % IMAGE_STEP_SIZE)
    h = h - (h % IMAGE_STEP_SIZE)

    addImageSizeOption(w)
    addImageSizeOption(h)

    widthField.value = w
    heightField.value = h
    widthField.dispatchEvent(new Event("change"))
    heightField.dispatchEvent(new Event("change"))
}
controlImagePreview.addEventListener("load", controlImageLoad)

function controlImageUnload() {
    controlImageSelector.value = null
    controlImagePreview.src = ""
    controlImagePreview.dispatchEvent(new Event("unload"))
}
controlImageClearBtn.addEventListener("click", controlImageUnload)

promptsFromFileSelector.addEventListener("change", async function() {
    if (promptsFromFileSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = promptsFromFileSelector.files[0]

    reader.addEventListener("load", async function() {
        await parseContent(reader.result)
    })

    if (file) {
        reader.readAsText(file)
    }
})

/* setup popup handlers */
document.querySelectorAll(".popup").forEach((popup) => {
    popup.addEventListener("click", (event) => {
        if (event.target == popup) {
            popup.classList.remove("active")
        }
    })
    var closeButton = popup.querySelector(".close-button")
    if (closeButton) {
        closeButton.addEventListener("click", () => {
            popup.classList.remove("active")
        })
    }
})

var tabElements = []
function selectTab(tab_id) {
    let tabInfo = tabElements.find((t) => t.tab.id == tab_id)
    if (!tabInfo.tab.classList.contains("active")) {
        tabElements.forEach((info) => {
            if (info.tab.classList.contains("active") && info.tab.parentNode === tabInfo.tab.parentNode) {
                info.tab.classList.toggle("active")
                info.content.classList.toggle("active")
            }
        })
        tabInfo.tab.classList.toggle("active")
        tabInfo.content.classList.toggle("active")
    }
    document.dispatchEvent(new CustomEvent("tabClick", { detail: tabInfo }))
}
function linkTabContents(tab) {
    var name = tab.id.replace("tab-", "")
    var content = document.getElementById(`tab-content-${name}`)
    tabElements.push({
        name: name,
        tab: tab,
        content: content,
    })

    tab.addEventListener("click", (event) => selectTab(tab.id))
}
function isTabActive(tab) {
    return tab.classList.contains("active")
}

function splashScreen(force = false) {
    const splashVersion = splashScreenPopup.dataset["version"]
    const lastSplash = localStorage.getItem("lastSplashScreenVersion") || 0
    if (testDiffusers.checked) {
        if (force || lastSplash < splashVersion) {
            splashScreenPopup.classList.add("active")
            localStorage.setItem("lastSplashScreenVersion", splashVersion)
        }
    }
}

document.getElementById("logo_img").addEventListener("click", (e) => {
    splashScreen(true)
})

promptField.addEventListener("input", debounce(renameMakeImageButton, 1000))

function onPing(event) {
    tunnelUpdate(event)
    packagesUpdate(event)
}

function tunnelUpdate(event) {
    if ("cloudflare" in event) {
        document.getElementById("cloudflare-off").classList.add("displayNone")
        document.getElementById("cloudflare-on").classList.remove("displayNone")
        cloudflareAddressField.value = event.cloudflare
        document.getElementById("toggle-cloudflare-tunnel").innerHTML = "Stop"
    } else {
        document.getElementById("cloudflare-on").classList.add("displayNone")
        document.getElementById("cloudflare-off").classList.remove("displayNone")
        document.getElementById("toggle-cloudflare-tunnel").innerHTML = "Start"
    }
}

function packagesUpdate(event) {
    let trtBtn = document.getElementById("toggle-tensorrt-install")
    let trtInstalled = "packages_installed" in event && "tensorrt" in event["packages_installed"]

    if ("packages_installing" in event && event["packages_installing"].includes("tensorrt")) {
        trtBtn.innerHTML = "Installing.."
        trtBtn.disabled = true
    } else {
        trtBtn.innerHTML = trtInstalled ? "Uninstall" : "Install"
        trtBtn.disabled = false
    }

    if (document.getElementById("toggle-tensorrt-install").innerHTML == "Uninstall") {
        document.querySelector("#enable_trt_config").classList.remove("displayNone")
        document.querySelector("#trt-build-config").classList.remove("displayNone")
    }
}

document.getElementById("toggle-cloudflare-tunnel").addEventListener("click", async function() {
    let command = "stop"
    if (document.getElementById("toggle-cloudflare-tunnel").innerHTML == "Start") {
        command = "start"
    }
    showToast(`Cloudflare tunnel ${command} initiated. Please wait.`)

    let res = await fetch("/tunnel/cloudflare/" + command, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
    })
    res = await res.json()

    console.log(`Cloudflare tunnel ${command} result:`, res)
})

document.getElementById("toggle-tensorrt-install").addEventListener("click", function(e) {
    if (this.disabled === true) {
        return
    }

    let command = this.innerHTML.toLowerCase()
    let self = this

    shiftOrConfirm(
        e,
        "Are you sure you want to " + command + " TensorRT?",
        async function() {
            showToast(`TensorRT ${command} started. Please wait.`)

            self.disabled = true

            if (command === "install") {
                self.innerHTML = "Installing.."
            } else if (command === "uninstall") {
                self.innerHTML = "Uninstalling.."
            }

            if (command === "installing..") {
                alert("Already installing TensorRT!")
                return
            }
            if (command !== "install" && command !== "uninstall") {
                return
            }

            let res = await fetch("/package/tensorrt", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    command: command,
                }),
            })
            res = await res.json()

            self.disabled = false

            if (res.status === "OK") {
                alert("TensorRT " + command + "ed successfully!")
                self.innerHTML = command === "install" ? "Uninstall" : "Install"
            } else if (res.status_code === 500) {
                alert("TensorselfRT failed to " + command + ": " + res.detail)
                self.innerHTML = command === "install" ? "Install" : "Uninstall"
            }

            console.log(`Package ${command} result:`, res)
        },
        false
    )
})

/* Embeddings */

addEmbeddingsThumb.addEventListener("click", (e) => addEmbeddingsThumbInput.click())
addEmbeddingsThumbInput.addEventListener("change", loadThumbnailImageFromFile)

function loadThumbnailImageFromFile() {
    if (addEmbeddingsThumbInput.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = addEmbeddingsThumbInput.files[0]

    reader.addEventListener("load", function(event) {
        let img = document.createElement("img")
        img.src = reader.result
        onUseAsThumbnailClick(
            {
                use_embeddings_model: getAllModelNames("embeddings").sort((a, b) =>
                    a.localeCompare(b, undefined, { sensitivity: "base" })
                ),
            },
            img
        )
    })

    if (file) {
        reader.readAsDataURL(file)
    }
}

function updateEmbeddingsList(filter = "") {
    function html(model, iconMap = {}, prefix = "", filter = "") {
        filter = filter.toLowerCase()
        let toplevel = document.createElement("div")
        let folders = document.createElement("div")

        let profileName = profileNameField.value
        model?.forEach((m) => {
            if (typeof m == "string") {
                let token = m.toLowerCase()
                if (token.search(filter) != -1) {
                    let button
                    let img = "/media/images/noimg.png"
                    if (token in iconMap) {
                        img = `/bucket/${profileName}/${iconMap[token]}`
                    }
                    button = createModifierCard(m, [img, img], true)
                    // }
                    button.dataset["embedding"] = m
                    button.addEventListener("click", onButtonClick)
                    toplevel.appendChild(button)
                }
            } else {
                let subdir = html(m[1], iconMap, prefix + m[0] + "/", filter)
                if (typeof subdir == "object") {
                    let div1 = document.createElement("div")
                    let div2 = document.createElement("div")
                    div1.classList.add("collapsible-content")
                    div1.classList.add("embedding-category")
                    div1.appendChild(subdir)
                    div2.replaceChildren(htmlToElement(`<h4 class="collapsible">${prefix}${m[0]}</h4>`), div1)
                    folders.appendChild(div2)
                }
            }
        })

        if (toplevel.children.length == 0 && folders.children.length == 0) {
            // Empty folder
            return ""
        }

        let result = document.createElement("div")
        result.replaceChildren(toplevel, htmlToElement('<br style="clear: both;">'), folders)
        return result
    }

    function onButtonClick(e) {
        let text = e.target.closest("[data-embedding]").dataset["embedding"]
        const insertIntoNegative = e.shiftKey || positiveEmbeddingText.classList.contains("displayNone")

        if (embeddingsModeField.value == "insert") {
            if (insertIntoNegative) {
                insertAtCursor(negativePromptField, text)
            } else {
                insertAtCursor(promptField, text)
            }
        } else {
            let pad = ""
            if (insertIntoNegative) {
                if (!negativePromptField.value.endsWith(" ")) {
                    pad = " "
                }
                negativePromptField.value += pad + text
            } else {
                if (!promptField.value.endsWith(" ")) {
                    pad = " "
                }
                promptField.value += pad + text
            }
        }
    }

    // Usually the rendering of the Embeddings HTML takes less than a second. In case it takes longer, show a spinner
    embeddingsList.innerHTML = `
        <div class="spinner-container">
          <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div>
          <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div>
          <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div>
          <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div> <div class="spinner-block"></div>
        </div>
    `

    let loraTokens = []
    let profileName = profileNameField.value
    let iconMap = {}

    Bucket.getList(`${profileName}/embeddings/`)
        .then((icons) => {
            iconMap = Object.assign(
                {},
                ...icons.map((x) => ({
                    [x
                        .toLowerCase()
                        .split(".")
                        .slice(0, -1)
                        .join(".")]: `embeddings/${x}`,
                }))
            )

            return Bucket.getList(`${profileName}/lora/`)
        })
        .then(async function (icons) {
            for (let lora of loraModelField.value.modelNames) {
                let keywords = await getLoraKeywords(lora)
                loraTokens = loraTokens.concat(keywords)
                let loraname = lora.split("/").pop()

                if (icons.includes(`${loraname}.png`)) {
                    keywords.forEach((kw) => {
                        iconMap[kw.toLowerCase()] = `lora/${loraname}.png`
                        
                    })
                }
            }

            let tokenList = [...modelsOptions.embeddings]
            if (loraTokens.length != 0) {
                tokenList.unshift(['LORA Keywords', loraTokens])
            }
            embeddingsList.replaceChildren(html(tokenList, iconMap, "", filter))
            createCollapsibles(embeddingsList)
            if (filter != "") {
                embeddingsExpandAll()
            }
            resizeModifierCards(embeddingsCardSizeSelector.value)
        })
}

function showEmbeddingDialog() {
    updateEmbeddingsList()
    embeddingsSearchBox.value = ""
    embeddingsDialog.showModal()
}

embeddingsButton.addEventListener("click", () => {
    positiveEmbeddingText.classList.remove("displayNone")
    negativeEmbeddingText.classList.add("displayNone")
    showEmbeddingDialog()
})

negativeEmbeddingsButton.addEventListener("click", () => {
    positiveEmbeddingText.classList.add("displayNone")
    negativeEmbeddingText.classList.remove("displayNone")
    showEmbeddingDialog()
})

embeddingsDialogCloseBtn.addEventListener("click", (e) => {
    embeddingsDialog.close()
})

embeddingsSearchBox.addEventListener("input", (e) => {
    updateEmbeddingsList(embeddingsSearchBox.value)
})

embeddingsCardSizeSelector.addEventListener("change", (e) => {
    resizeModifierCards(embeddingsCardSizeSelector.value)
})

modalDialogCloseOnBackdropClick(embeddingsDialog)
makeDialogDraggable(embeddingsDialog)

const collapseText = "Collapse Categories"
const expandText = "Expand Categories"

const collapseIconClasses = ["fa-solid", "fa-square-minus"]
const expandIconClasses = ["fa-solid", "fa-square-plus"]

function embeddingsCollapseAll() {
    const btnElem = embeddingsCollapsiblesBtn

    const iconElem = btnElem.querySelector(".embeddings-action-icon")
    const textElem = btnElem.querySelector(".embeddings-action-text")
    collapseAll("#embeddings-list .collapsible")

    collapsiblesBtnState = false

    collapseIconClasses.forEach((c) => iconElem.classList.remove(c))
    expandIconClasses.forEach((c) => iconElem.classList.add(c))

    textElem.innerText = expandText
}

function embeddingsExpandAll() {
    const btnElem = embeddingsCollapsiblesBtn

    const iconElem = btnElem.querySelector(".embeddings-action-icon")
    const textElem = btnElem.querySelector(".embeddings-action-text")
    expandAll("#embeddings-list .collapsible")

    collapsiblesBtnState = true

    expandIconClasses.forEach((c) => iconElem.classList.remove(c))
    collapseIconClasses.forEach((c) => iconElem.classList.add(c))

    textElem.innerText = collapseText
}

embeddingsCollapsiblesBtn.addEventListener("click", (e) => {
    if (collapsiblesBtnState) {
        embeddingsCollapseAll()
    } else {
        embeddingsExpandAll()
    }
})

/* Pause function */
document.querySelectorAll(".tab").forEach(linkTabContents)

window.addEventListener("beforeunload", function(e) {
    const msg = "Unsaved pictures will be lost!"

    let elementList = document.getElementsByClassName("imageTaskContainer")
    if (elementList.length != 0) {
        e.preventDefault()
        ;(e || window.event).returnValue = msg
        return msg
    } else {
        return true
    }
})

document.addEventListener("collapsibleClick", function(e) {
    let header = e.detail
    if (header === document.querySelector("#negative_prompt_handle")) {
        if (header.classList.contains("active")) {
            negativeEmbeddingsButton.classList.remove("displayNone")
        } else {
            negativeEmbeddingsButton.classList.add("displayNone")
        }
    }
})

createCollapsibles()
prettifyInputs(document)

// set the textbox as focused on start
promptField.focus()
promptField.selectionStart = promptField.value.length

////////////////////////////// Image Size Widget //////////////////////////////////////////

function roundToMultiple(number, n) {
    if (n == "") {
        n = 1
    }
    return Math.round(number / n) * n
}

function addImageSizeOption(size) {
    let sizes = Object.values(widthField.options).map((o) => o.value)
    if (!sizes.includes(String(size))) {
        sizes.push(String(size))
        sizes.sort((a, b) => Number(a) - Number(b))

        let option = document.createElement("option")
        option.value = size
        option.text = `${size}`

        widthField.add(option, sizes.indexOf(String(size)))
        heightField.add(option.cloneNode(true), sizes.indexOf(String(size)))
    }
}

function setImageWidthHeight(w, h) {
    let step = customWidthField.step
    w = roundToMultiple(w, step)
    h = roundToMultiple(h, step)

    addImageSizeOption(w)
    addImageSizeOption(h)

    widthField.value = w
    heightField.value = h
    widthField.dispatchEvent(new Event("change"))
    heightField.dispatchEvent(new Event("change"))
}

function enlargeImageSize(factor) {
    let step = customWidthField.step

    let w = roundToMultiple(widthField.value * factor, step)
    let h = roundToMultiple(heightField.value * factor, step)
    customWidthField.value = w
    customHeightField.value = h
}

let recentResolutionsValues = []

;(function() {
    ///// Init resolutions dropdown

    function makeResolutionButtons(listElement, resolutionList) {
        listElement.innerHTML = ""
        resolutionList.forEach((el) => {
            let button = createElement("button", { style: "width: 8em;" }, "tertiaryButton", `${el.w}×${el.h}`)
            button.addEventListener("click", () => {
                customWidthField.value = el.w
                customHeightField.value = el.h
                hidePopup()
            })
            listElement.appendChild(button)
            listElement.appendChild(document.createElement("br"))
        })
    }

    enlargeButtons.querySelectorAll("button").forEach((button) =>
        button.addEventListener("click", (e) => {
            enlargeImageSize(parseFloat(button.dataset["factor"]))
            hidePopup()
        })
    )

    customWidthField.addEventListener("change", () => {
        let w = customWidthField.value
        customWidthField.value = roundToMultiple(w, customWidthField.step)
        if (w != customWidthField.value) {
            showToast(`Rounded width to the closest multiple of ${customWidthField.step}.`)
        }
    })

    customHeightField.addEventListener("change", () => {
        let h = customHeightField.value
        customHeightField.value = roundToMultiple(h, customHeightField.step)
        if (h != customHeightField.value) {
            showToast(`Rounded height to the closest multiple of ${customHeightField.step}.`)
        }
    })

    makeImageBtn.addEventListener("click", () => {
        let w = widthField.value
        let h = heightField.value

        recentResolutionsValues = recentResolutionsValues.filter((el) => el.w != w || el.h != h)
        recentResolutionsValues.unshift({ w: w, h: h })
        recentResolutionsValues = recentResolutionsValues.slice(0, 8)

        localStorage.recentResolutionsValues = JSON.stringify(recentResolutionsValues)
        makeResolutionButtons(recentResolutionList, recentResolutionsValues)
    })

    const defaultResolutionsValues = [
        { w: 512, h: 512 },
        { w: 448, h: 640 },
        { w: 512, h: 768 },
        { w: 768, h: 512 },
        { w: 1024, h: 768 },
        { w: 768, h: 1024 },
        { w: 1024, h: 1024 },
        { w: 1920, h: 1080 },
    ]
    let _jsonstring = localStorage.recentResolutionsValues
    if (_jsonstring == undefined) {
        recentResolutionsValues = defaultResolutionsValues
        localStorage.recentResolutionsValues = JSON.stringify(recentResolutionsValues)
    } else {
        recentResolutionsValues = JSON.parse(localStorage.recentResolutionsValues)
    }

    makeResolutionButtons(recentResolutionList, recentResolutionsValues)
    makeResolutionButtons(commonResolutionList, defaultResolutionsValues)

    recentResolutionsValues.forEach((val) => {
        addImageSizeOption(val.w)
        addImageSizeOption(val.h)
    })

    function processClick(e) {
        if (!recentResolutionsPopup.contains(e.target)) {
            hidePopup()
        }
    }

    function showPopup() {
        customWidthField.value = widthField.value
        customHeightField.value = heightField.value
        recentResolutionsPopup.classList.remove("displayNone")
        resizeSlider.value = 1
        resizeSlider.dataset["w"] = widthField.value
        resizeSlider.dataset["h"] = heightField.value
        document.addEventListener("click", processClick)
    }

    function hidePopup() {
        recentResolutionsPopup.classList.add("displayNone")
        setImageWidthHeight(customWidthField.value, customHeightField.value)
        document.removeEventListener("click", processClick)
    }

    recentResolutionsButton.addEventListener("click", (event) => {
        if (recentResolutionsPopup.classList.contains("displayNone")) {
            showPopup()
            event.stopPropagation()
        } else {
            hidePopup()
        }
    })

    resizeSlider.addEventListener("input", (e) => {
        let w = parseInt(resizeSlider.dataset["w"])
        let h = parseInt(resizeSlider.dataset["h"])
        let factor = parseFloat(resizeSlider.value)
        let step = customWidthField.step

        customWidthField.value = roundToMultiple(w * factor * factor, step)
        customHeightField.value = roundToMultiple(h * factor * factor, step)
    })

    resizeSlider.addEventListener("change", (e) => {
        hidePopup()
    })

    swapWidthHeightButton.addEventListener("click", (event) => {
        let temp = widthField.value
        widthField.value = heightField.value
        heightField.value = temp
    })
})()

document.addEventListener("before_task_start", (e) => {
    let task = e.detail.task

    // Update the seed *before* starting the processing so it's retained if user stops the task
    if (randomSeedField.checked) {
        seedField.value = task.seed
    }
})

document.addEventListener("after_task_start", (e) => {
    renderButtons.style.display = "flex"
    renameMakeImageButton()
    updateInitialText()
})

document.addEventListener("on_task_step", (e) => {
    showImages(e.detail.reqBody, e.detail.stepUpdate, e.detail.outputContainer, true)
})

document.addEventListener("on_render_task_success", (e) => {
    showImages(e.detail.reqBody, e.detail.stepUpdate, e.detail.outputContainer, false)
})

document.addEventListener("on_render_task_fail", (e) => {
    let task = e.detail.task
    let stepUpdate = e.detail.stepUpdate

    const outputMsg = task["outputMsg"]
    let msg = ""
    if ("detail" in stepUpdate && typeof stepUpdate.detail === "string" && stepUpdate.detail.length > 0) {
        msg = stepUpdate.detail
        if (msg.toLowerCase().includes("out of memory")) {
            msg += `<br/><br/>
                    <b>Suggestions</b>:
                    <br/>
                    1. If you have set an initial image, please try reducing its dimension to ${MAX_INIT_IMAGE_DIMENSION}x${MAX_INIT_IMAGE_DIMENSION} or smaller.<br/>
                    2. Try picking a lower level in the '<em>GPU Memory Usage</em>' setting (in the '<em>Settings</em>' tab).<br/>
                    3. Try generating a smaller image.<br/>`
        } else if (msg.includes("DefaultCPUAllocator: not enough memory")) {
            msg += `<br/><br/>
                    Reason: Your computer is running out of system RAM!
                    <br/><br/>
                    <b>Suggestions</b>:
                    <br/>
                    1. Try closing unnecessary programs and browser tabs.<br/>
                    2. If that doesn't help, please increase your computer's virtual memory by following these steps for
                        <a href="https://www.ibm.com/docs/en/opw/8.2.0?topic=tuning-optional-increasing-paging-file-size-windows-computers" target="_blank">Windows</a> or
                        <a href="https://linuxhint.com/increase-swap-space-linux/" target="_blank">Linux</a>.<br/>
                    3. Try restarting your computer.<br/>`
        } else if (msg.includes("RuntimeError: output with shape [320, 320] doesn't match the broadcast shape")) {
            msg += `<br/><br/>
                    <b>Reason</b>: You tried to use a LORA that was trained for a different Stable Diffusion model version!
                    <br/><br/>
                    <b>Suggestions</b>:
                    <br/>
                    Try to use a different model or a different LORA.`
        } else if (msg.includes("'ModuleList' object has no attribute '1'")) {
            msg += `<br/><br/>
                    <b>Reason</b>: SDXL models need a yaml config file.
                    <br/><br/>
                    <b>Suggestions</b>:
                    <br/>
                    <ol>
                    <li>Download the <a href="https://gist.githubusercontent.com/JeLuF/5dc56e7a3a6988265c423f464d3cbdd3/raw/4ba4c39b1c7329877ad7a39c8c8a077ea4b53d11/dreamshaperXL10_alpha2Xl10.yaml" target="_blank">config file</a></li>
                    <li>Save it in the same directory as the SDXL model file</li>
                    <li>Rename the config file so that it matches the filename of the model, with the extension of the model file replaced by <tt>yaml</tt>. 
                        For example, if the model file is called <tt>FantasySDXL_v2.safetensors</tt>, the config file must be called <tt>FantasySDXL_v2.yaml</tt>.
                    </ol>`
        }
    } else {
        msg = `Unexpected Read Error:<br/><pre>StepUpdate: ${JSON.stringify(stepUpdate, undefined, 4)}</pre>`
    }
    logError(msg, stepUpdate, outputMsg)
})

document.addEventListener("on_all_tasks_complete", (e) => {
    renderButtons.style.display = "none"
    renameMakeImageButton()

    if (isSoundEnabled()) {
        playSound()
    }
})
