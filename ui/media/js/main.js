"use strict" // Opt in to a restricted variant of JavaScript
const MAX_INIT_IMAGE_DIMENSION = 768
const MIN_GPUS_TO_SHOW_SELECTION = 2

const IMAGE_REGEX = new RegExp('data:image/[A-Za-z]+;base64')
const htmlTaskMap = new WeakMap()

const taskConfigSetup = {
    taskConfig: {
        seed: { value: ({ seed }) => seed, label: 'Seed' },
        dimensions: { value: ({ reqBody }) => `${reqBody?.width}x${reqBody?.height}`, label: 'Dimensions' },
        sampler_name: 'Sampler',
        num_inference_steps: 'Inference Steps',
        guidance_scale: 'Guidance Scale',
        use_stable_diffusion_model: 'Model',
        use_vae_model: { label: 'VAE', visible: ({ reqBody }) => reqBody?.use_vae_model !== undefined && reqBody?.use_vae_model.trim() !== ''},
        negative_prompt: { label: 'Negative Prompt', visible: ({ reqBody }) => reqBody?.negative_prompt !== undefined && reqBody?.negative_prompt.trim() !== ''},
        prompt_strength: 'Prompt Strength',
        use_face_correction: 'Fix Faces',
        upscale: { value: ({ reqBody }) => `${reqBody?.use_upscale} (${reqBody?.upscale_amount || 4}x)`, label: 'Upscale', visible: ({ reqBody }) => !!reqBody?.use_upscale },
        use_hypernetwork_model: 'Hypernetwork',
        hypernetwork_strength: { label: 'Hypernetwork Strength', visible: ({ reqBody }) => !!reqBody?.use_hypernetwork_model },
        use_lora_model: 'Lora Model',
        preserve_init_image_color_profile: 'Preserve Color Profile',
    },
    pluginTaskConfig: {},
    getCSSKey: (key) => key.split('_').map((s) => s.charAt(0).toUpperCase() + s.slice(1)).join('')
}

let imageCounter = 0
let imageRequest = []

let promptField = document.querySelector('#prompt')
let promptsFromFileSelector = document.querySelector('#prompt_from_file')
let promptsFromFileBtn = document.querySelector('#promptsFromFileBtn')
let negativePromptField = document.querySelector('#negative_prompt')
let numOutputsTotalField = document.querySelector('#num_outputs_total')
let numOutputsParallelField = document.querySelector('#num_outputs_parallel')
let numInferenceStepsField = document.querySelector('#num_inference_steps')
let guidanceScaleSlider = document.querySelector('#guidance_scale_slider')
let guidanceScaleField = document.querySelector('#guidance_scale')
let outputQualitySlider = document.querySelector('#output_quality_slider')
let outputQualityField = document.querySelector('#output_quality')
let outputQualityRow = document.querySelector('#output_quality_row')
let randomSeedField = document.querySelector("#random_seed")
let seedField = document.querySelector('#seed')
let widthField = document.querySelector('#width')
let heightField = document.querySelector('#height')
let smallImageWarning = document.querySelector('#small_image_warning')
let initImageSelector = document.querySelector("#init_image")
let initImagePreview = document.querySelector("#init_image_preview")
let initImageSizeBox = document.querySelector("#init_image_size_box")
let maskImageSelector = document.querySelector("#mask")
let maskImagePreview = document.querySelector("#mask_preview")
let applyColorCorrectionField = document.querySelector('#apply_color_correction')
let colorCorrectionSetting = document.querySelector('#apply_color_correction_setting')
let promptStrengthSlider = document.querySelector('#prompt_strength_slider')
let promptStrengthField = document.querySelector('#prompt_strength')
let samplerField = document.querySelector('#sampler_name')
let samplerSelectionContainer = document.querySelector("#samplerSelection")
let useFaceCorrectionField = document.querySelector("#use_face_correction")
let gfpganModelField = new ModelDropdown(document.querySelector("#gfpgan_model"), 'gfpgan')
let useUpscalingField = document.querySelector("#use_upscale")
let upscaleModelField = document.querySelector("#upscale_model")
let upscaleAmountField = document.querySelector("#upscale_amount")
let stableDiffusionModelField = new ModelDropdown(document.querySelector('#stable_diffusion_model'), 'stable-diffusion')
let vaeModelField = new ModelDropdown(document.querySelector('#vae_model'), 'vae', 'None')
let hypernetworkModelField = new ModelDropdown(document.querySelector('#hypernetwork_model'), 'hypernetwork', 'None')
let hypernetworkStrengthSlider = document.querySelector('#hypernetwork_strength_slider')
let hypernetworkStrengthField = document.querySelector('#hypernetwork_strength')
let loraModelField = new ModelDropdown(document.querySelector('#lora_model'), 'lora', 'None')
let loraAlphaSlider = document.querySelector('#lora_alpha_slider')
let loraAlphaField = document.querySelector('#lora_alpha')
let outputFormatField = document.querySelector('#output_format')
let outputLosslessField = document.querySelector('#output_lossless')
let outputLosslessContainer = document.querySelector('#output_lossless_container')
let blockNSFWField = document.querySelector('#block_nsfw')
let showOnlyFilteredImageField = document.querySelector("#show_only_filtered_image")
let updateBranchLabel = document.querySelector("#updateBranchLabel")
let streamImageProgressField = document.querySelector("#stream_image_progress")
let thumbnailSizeField = document.querySelector("#thumbnail_size-input")
let autoscrollBtn = document.querySelector("#auto_scroll_btn")
let autoScroll = document.querySelector("#auto_scroll")

let makeImageBtn = document.querySelector('#makeImage')
let stopImageBtn = document.querySelector('#stopImage')
let pauseBtn = document.querySelector('#pause')
let resumeBtn = document.querySelector('#resume')
let renderButtons = document.querySelector('#render-buttons')

let imagesContainer = document.querySelector('#current-images')
let initImagePreviewContainer = document.querySelector('#init_image_preview_container')
let initImageClearBtn = document.querySelector('.init_image_clear')
let promptStrengthContainer = document.querySelector('#prompt_strength_container')

let initialText = document.querySelector("#initial-text")
let previewTools = document.querySelector("#preview-tools")
let clearAllPreviewsBtn = document.querySelector("#clear-all-previews")
let showDownloadPopupBtn = document.querySelector("#show-download-popup")
let saveAllImagesPopup = document.querySelector("#download-images-popup")
let saveAllImagesBtn = document.querySelector("#save-all-images")
let saveAllZipToggle = document.querySelector("#zip_toggle")
let saveAllTreeToggle = document.querySelector("#tree_toggle")
let saveAllJSONToggle = document.querySelector("#json_toggle")
let saveAllFoldersOption = document.querySelector("#download-add-folders")

let maskSetting = document.querySelector('#enable_mask')

const processOrder = document.querySelector('#process_order_toggle')

let imagePreview = document.querySelector("#preview")
let imagePreviewContent = document.querySelector("#preview-content")

let undoButton = document.querySelector("#undo")
let undoBuffer = []
const UNDO_LIMIT = 20

imagePreview.addEventListener('drop', function(ev) {
    const data = ev.dataTransfer?.getData("text/plain");
    if (!data) {
        return
    }
    const movedTask = document.getElementById(data)
    if (!movedTask) {
        return
    }
    ev.preventDefault()
    let moveTarget = ev.target
    while (moveTarget && typeof moveTarget === 'object' && moveTarget.parentNode !== imagePreviewContent) {
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



let showConfigToggle = document.querySelector('#configToggleBtn')
// let configBox = document.querySelector('#config')
// let outputMsg = document.querySelector('#outputMsg')

let soundToggle = document.querySelector('#sound_toggle')

let serverStatusColor = document.querySelector('#server-status-color')
let serverStatusMsg = document.querySelector('#server-status-msg')

function getLocalStorageBoolItem(key, fallback) {
    let item = localStorage.getItem(key)
    if (item === null) {
        return fallback
    }

    return (item === 'true' ? true : false)
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

function setStatus(statusType, msg, msgType) {
}

function setServerStatus(event) {
    switch(event.type) {
        case 'online':
            serverStatusColor.style.color = 'var(--status-green)'
            serverStatusMsg.style.color = 'var(--status-green)'
            serverStatusMsg.innerText = 'Stable Diffusion is ' + event.message
            break
        case 'busy':
            serverStatusColor.style.color = 'var(--status-orange)'
            serverStatusMsg.style.color = 'var(--status-orange)'
            serverStatusMsg.innerText = 'Stable Diffusion is ' + event.message
            break
        case 'error':
            serverStatusColor.style.color = 'var(--status-red)'
            serverStatusMsg.style.color = 'var(--status-red)'
            serverStatusMsg.innerText = 'Stable Diffusion has stopped'
            break
    }
    if (SD.serverState.devices) {
        setDeviceInfo(SD.serverState.devices)
    }
}

// shiftOrConfirm(e, prompt, fn)
//   e      : MouseEvent
//   prompt : Text to be shown as prompt. Should be a question to which "yes" is a good answer.
//   fn     : function to be called if the user confirms the dialog or has the shift key pressed
//
// If the user had the shift key pressed while clicking, the function fn will be executed.
// If the setting "confirm_dangerous_actions" in the system settings is disabled, the function 
// fn will be executed.
// Otherwise, a confirmation dialog is shown. If the user confirms, the function fn will also
// be executed.
function shiftOrConfirm(e, prompt, fn) {
    e.stopPropagation()
    if (e.shiftKey || !confirmDangerousActionsField.checked) {
         fn(e)
    } else {
        $.confirm({
            theme: 'modern',
            title: prompt,
            useBootstrap: false,
            animateFromElement: false,
            content: '<small>Tip: To skip this dialog, use shift-click or disable the "Confirm dangerous actions" setting in the Settings tab.</small>',
            buttons: {
                yes: () => { fn(e) },
                cancel: () => {}
            }
        }); 
    }
}

function logMsg(msg, level, outputMsg) {
    if (outputMsg.hasChildNodes()) {
        outputMsg.appendChild(document.createElement('br'))
    }
    if (level === 'error') {
        outputMsg.innerHTML += '<span style="color: red">Error: ' + msg + '</span>'
    } else if (level === 'warn') {
        outputMsg.innerHTML += '<span style="color: orange">Warning: ' + msg + '</span>'
    } else {
        outputMsg.innerText += msg
    }
    console.log(level, msg)
}

function logError(msg, res, outputMsg) {
    logMsg(msg, 'error', outputMsg)

    console.log('request error', res)
    setStatus('request', 'error', 'error')
}

function playSound() {
    const audio = new Audio('/media/ding.mp3')
    audio.volume = 0.2
    var promise = audio.play()
    if (promise !== undefined) {
        promise.then(_ => {}).catch(error => {
            console.warn("browser blocked autoplay")
        })
    }
}

function undoableRemove(element, doubleUndo=false) {
    let data = { 'element': element, 'parent': element.parentNode, 'prev': element.previousSibling, 'next': element.nextSibling, 'doubleUndo': doubleUndo }
    undoBuffer.push(data)
    if (undoBuffer.length > UNDO_LIMIT) {
        // Remove item from memory and also remove it from the data structures
        let item = undoBuffer.shift()
        htmlTaskMap.delete(item.element)
        item.element.querySelectorAll('[data-imagecounter]').forEach( (img) => { delete imageRequest[img.dataset['imagecounter']] })
    }
    element.remove()
    if (undoBuffer.length != 0) {
        undoButton.classList.remove('displayNone')
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
        undoButton.classList.add('displayNone')
    }
    updateInitialText()
}

undoButton.addEventListener('click', () =>  { undoRemove() })

function showImages(reqBody, res, outputContainer, livePreview) {
    let imageItemElements = outputContainer.querySelectorAll('.imgItem')
    if(typeof res != 'object') return
    res.output.reverse()
    res.output.forEach((result, index) => {
        const imageData = result?.data || result?.path + '?t=' + Date.now(),
            imageSeed = result?.seed,
            imagePrompt = reqBody.prompt,
            imageInferenceSteps = reqBody.num_inference_steps,
            imageGuidanceScale = reqBody.guidance_scale,
            imageWidth = reqBody.width,
            imageHeight = reqBody.height;

        if (!imageData.includes('/')) {
            // res contained no data for the image, stop execution
            setStatus('request', 'invalid image', 'error')
            return
        }

        let imageItemElem = (index < imageItemElements.length ? imageItemElements[index] : null)
        if(!imageItemElem) {
            imageItemElem = document.createElement('div')
            imageItemElem.className = 'imgItem'
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
                </div>
            `
            outputContainer.appendChild(imageItemElem)
            const imageRemoveBtn = imageItemElem.querySelector('.imgPreviewItemClearBtn')
            let parentTaskContainer = imageRemoveBtn.closest('.imageTaskContainer')
            imageRemoveBtn.addEventListener('click', (e) => {
                undoableRemove(imageItemElem)
                let allHidden = true;
                let children = parentTaskContainer.querySelectorAll('.imgItem');
                for(let x = 0; x < children.length; x++) {
                    let child = children[x];
                    if(child.style.display != "none") {
                        allHidden = false;
                    }
                }
                if(allHidden === true) {
                    const req = htmlTaskMap.get(parentTaskContainer)
                    if(!req.isProcessing || req.batchesDone == req.batchCount) { undoableRemove(parentTaskContainer, true) }
                }
            })
        }
        const imageElem = imageItemElem.querySelector('img')
        imageElem.src = imageData
        imageElem.width = parseInt(imageWidth)
        imageElem.height = parseInt(imageHeight)
        imageElem.setAttribute('data-prompt', imagePrompt)
        imageElem.setAttribute('data-steps', imageInferenceSteps)
        imageElem.setAttribute('data-guidance', imageGuidanceScale)

        imageElem.addEventListener('load', function() {
            imageItemElem.querySelector('.img_bottom_label').innerText = `${this.naturalWidth} x ${this.naturalHeight}`
        })

        const imageInfo = imageItemElem.querySelector('.imgItemInfo')
        imageInfo.style.visibility = (livePreview ? 'hidden' : 'visible')

        if ('seed' in result && !imageElem.hasAttribute('data-seed')) {
            const imageExpandBtn = imageItemElem.querySelector('.imgExpandBtn')
            imageExpandBtn.addEventListener('click', function() {
                imageModal(imageElem.src)
            })

            const req = Object.assign({}, reqBody, {
                seed: result?.seed || reqBody.seed
            })
            imageElem.setAttribute('data-seed', req.seed)
            imageElem.setAttribute('data-imagecounter', ++imageCounter)
            imageRequest[imageCounter] = req
            const imageSeedLabel = imageItemElem.querySelector('.imgSeedLabel')
            imageSeedLabel.innerText = 'Seed: ' + req.seed

            let buttons = [
                { text: 'Use as Input', on_click: onUseAsInputClick },
                [
                    { html: '<i class="fa-solid fa-download"></i> Download Image', on_click: onDownloadImageClick, class: "download-img" },
                    { html: '<i class="fa-solid fa-download"></i> JSON', on_click: onDownloadJSONClick, class: "download-json" }
                ],
                { text: 'Make Similar Images', on_click: onMakeSimilarClick },
                { text: 'Draw another 25 steps', on_click: onContinueDrawingClick },
                [
                    { text: 'Upscale', on_click: onUpscaleClick, filter: (req, img) => !req.use_upscale },
                    { text: 'Fix Faces', on_click: onFixFacesClick, filter: (req, img) => !req.use_face_correction }
                ]
            ]

            // include the plugins
            buttons = buttons.concat(PLUGINS['IMAGE_INFO_BUTTONS'])

            const imgItemInfo = imageItemElem.querySelector('.imgItemInfo')
            const img = imageItemElem.querySelector('img')
            const createButton = function(btnInfo) {
                if (Array.isArray(btnInfo)) {
                    const wrapper = document.createElement('div');
                    btnInfo
                        .map(createButton)
                        .forEach(buttonElement => wrapper.appendChild(buttonElement))
                    return wrapper
                }

                const isLabel = btnInfo.type === 'label'

                const newButton = document.createElement(isLabel ? 'span' : 'button')
                newButton.classList.add('tasksBtns')

                if (btnInfo.html) {
                    const html = typeof btnInfo.html === 'function' ? btnInfo.html() : btnInfo.html
                    if (html instanceof HTMLElement) {
                        newButton.appendChild(html)
                    } else {
                        newButton.innerHTML = html
                    }
                } else {
                    newButton.innerText = typeof btnInfo.text === 'function' ? btnInfo.text() : btnInfo.text
                }

                if (btnInfo.on_click || !isLabel) {
                    newButton.addEventListener('click', function(event) {
                        btnInfo.on_click(req, img, event)
                    })
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
            buttons.forEach(btn => {
                if (Array.isArray(btn)) {
                    btn = btn.filter(btnInfo => !btnInfo.filter || btnInfo.filter(req, img) === true)
                    if (btn.length === 0) {
                        return
                    }
                } else if (btn.filter && btn.filter(req, img) === false) {
                    return
                }

                try {
                    imgItemInfo.appendChild(createButton(btn))
                } catch (err) {
                    console.error('Error creating image info button from plugin: ', btn, err)
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

function getDownloadFilename(img, suffix) {
    const imageSeed = img.getAttribute('data-seed')
    const imagePrompt = img.getAttribute('data-prompt')
    const imageInferenceSteps = img.getAttribute('data-steps')
    const imageGuidanceScale = img.getAttribute('data-guidance')
    
    return createFileName(imagePrompt, imageSeed, imageInferenceSteps, imageGuidanceScale, suffix)
}

function onDownloadJSONClick(req, img) {
    const name = getDownloadFilename(img, 'json')
    const blob = new Blob([JSON.stringify(req, null, 2)], { type: 'text/plain' })
    saveAs(blob, name)
}

function onDownloadImageClick(req, img) {
    const name = getDownloadFilename(img, req['output_format'])
    const blob = dataURItoBlob(img.src)
    saveAs(blob, name)
}

function modifyCurrentRequest(...reqDiff) {
    const newTaskRequest = getCurrentUserRequest()

    newTaskRequest.reqBody = Object.assign(newTaskRequest.reqBody, ...reqDiff, {
        use_cpu: useCPUField.checked
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
        seed: Math.floor(Math.random() * 10000000)
    })

    newTaskRequest.numOutputsTotal = 5
    newTaskRequest.batchCount = 5

    delete newTaskRequest.reqBody.mask

    createTask(newTaskRequest)
}

function enqueueImageVariationTask(req, img, reqDiff) {
    const imageSeed = img.getAttribute('data-seed')

    const newRequestBody = {
        num_outputs: 1, // this can be user-configurable in the future
        seed: imageSeed
    }

    // If the user is editing pictures, stop modifyCurrentRequest from importing
    // new values by setting the missing properties to undefined
    if (!('init_image' in req) && !('init_image' in reqDiff)) {
        newRequestBody.init_image = undefined
        newRequestBody.mask = undefined
    } else if (!('mask' in req) && !('mask' in reqDiff)) {
        newRequestBody.mask = undefined
    }

    const newTaskRequest = modifyCurrentRequest(req, reqDiff, newRequestBody)
    newTaskRequest.numOutputsTotal = 1 // this can be user-configurable in the future
    newTaskRequest.batchCount = 1

    createTask(newTaskRequest)
}

function onUpscaleClick(req, img) {
    enqueueImageVariationTask(req, img, {
        use_upscale: upscaleModelField.value
    })
}

function onFixFacesClick(req, img) {
    enqueueImageVariationTask(req, img, {
        use_face_correction: gfpganModelField.value
    })
}

function onContinueDrawingClick(req, img) {
    enqueueImageVariationTask(req, img, {
        num_inference_steps: parseInt(req.num_inference_steps) + 25
    })
}

function getUncompletedTaskEntries() {
    const taskEntries = Array.from(
        document.querySelectorAll('#preview .imageTaskContainer .taskStatusLabel')
        ).filter((taskLabel) => taskLabel.style.display !== 'none'
        ).map(function(taskLabel) {
            let imageTaskContainer = taskLabel.parentNode
            while(!imageTaskContainer.classList.contains('imageTaskContainer') && imageTaskContainer.parentNode) {
                imageTaskContainer = imageTaskContainer.parentNode
            }
            return imageTaskContainer
        })
    if (!processOrder.checked) {
        taskEntries.reverse()
    }
    return taskEntries
}

function makeImage() {
    if (typeof performance == "object" && performance.mark) {
        performance.mark('click-makeImage')
    }

    if (!SD.isServerAvailable()) {
        alert('The server is not available.')
        return
    }
    if (!randomSeedField.checked && seedField.value == '') {
        alert('The "Seed" field must not be empty.')
        return
    }
    if (numInferenceStepsField.value == '') {
        alert('The "Inference Steps" field must not be empty.')
        return
    }
    if (numOutputsTotalField.value == '' || numOutputsTotalField.value == 0) {
        numOutputsTotalField.value = 1
    }
    if (numOutputsParallelField.value == '' || numOutputsParallelField.value == 0) {
        numOutputsParallelField.value = 1
    }
    if (guidanceScaleField.value == '') {
        guidanceScaleField.value = guidanceScaleSlider.value / 10
    }
    const taskTemplate = getCurrentUserRequest()
    const newTaskRequests = getPrompts().map((prompt) => Object.assign({}, taskTemplate, {
        reqBody: Object.assign({ prompt: prompt }, taskTemplate.reqBody)
    }))
    newTaskRequests.forEach(createTask)

    updateInitialText()
}

async function onIdle() {
    const serverCapacity = SD.serverCapacity
    if (pauseClient===true) {
        await resumeClient()
    }

    for (const taskEntry of getUncompletedTaskEntries()) {
        if (SD.activeTasks.size >= serverCapacity) {
            break
        }
        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            const taskStatusLabel = taskEntry.querySelector('.taskStatusLabel')
            taskStatusLabel.style.display = 'none'
            continue
        }
        await onTaskStart(task)
    }
}

function getTaskUpdater(task, reqBody, outputContainer) {
    const outputMsg = task['outputMsg']
    const progressBar = task['progressBar']
    const progressBarInner = progressBar.querySelector("div")

    const batchCount = task.batchCount
    let lastStatus = undefined
    return async function(event) {
        if (this.status !== lastStatus) {
            lastStatus = this.status
            switch(this.status) {
                case SD.TaskStatus.pending:
                    task['taskStatusLabel'].innerText = "Pending"
                    task['taskStatusLabel'].classList.add('waitingTaskLabel')
                    break
                case SD.TaskStatus.waiting:
                    task['taskStatusLabel'].innerText = "Waiting"
                    task['taskStatusLabel'].classList.add('waitingTaskLabel')
                    task['taskStatusLabel'].classList.remove('activeTaskLabel')
                    break
                case SD.TaskStatus.processing:
                case SD.TaskStatus.completed:
                    task['taskStatusLabel'].innerText = "Processing"
                    task['taskStatusLabel'].classList.add('activeTaskLabel')
                    task['taskStatusLabel'].classList.remove('waitingTaskLabel')
                    break
                case SD.TaskStatus.stopped:
                    break
                case SD.TaskStatus.failed:
                    if (!SD.isServerAvailable()) {
                        logError("Stable Diffusion is still starting up, please wait. If this goes on beyond a few minutes, Stable Diffusion has probably crashed. Please check the error message in the command-line window.", event, outputMsg)
                    } else if (typeof event?.response === 'object') {
                        let msg = 'Stable Diffusion had an error reading the response:<br/><pre>'
                        if (this.exception) {
                            msg += `Error: ${this.exception.message}<br/>`
                        }
                        try { // 'Response': body stream already read
                            msg += 'Read: ' + await event.response.text()
                        } catch(e) {
                            msg += 'Unexpected end of stream. '
                        }
                        const bufferString = event.reader.bufferedString
                        if (bufferString) {
                            msg += 'Buffered data: ' + bufferString
                        }
                        msg += '</pre>'
                        logError(msg, event, outputMsg)
                    } else {
                        let msg = `Unexpected Read Error:<br/><pre>Error:${this.exception}<br/>EventInfo: ${JSON.stringify(event, undefined, 4)}</pre>`
                        logError(msg, event, outputMsg)
                    }
                    break
            }
        }
        if ('update' in event) {
            const stepUpdate = event.update
            if (!('step' in stepUpdate)) {
                return
            }
            // task.instances can be a mix of different tasks with uneven number of steps (Render Vs Filter Tasks)
            const overallStepCount = task.instances.reduce(
                (sum, instance) => sum + (instance.isPending ? Math.max(0, instance.step || stepUpdate.step) / (instance.total_steps || stepUpdate.total_steps) : 1),
                0 // Initial value
            ) * stepUpdate.total_steps // Scale to current number of steps.
            const totalSteps = task.instances.reduce(
                (sum, instance) => sum + (instance.total_steps || stepUpdate.total_steps),
                stepUpdate.total_steps * (batchCount - task.batchesDone) // Initial value at (unstarted task count * Nbr of steps)
            )
            const percent = Math.min(100, 100 * (overallStepCount / totalSteps)).toFixed(0)

            const timeTaken = stepUpdate.step_time // sec
            const stepsRemaining = Math.max(0, totalSteps - overallStepCount)
            const timeRemaining = (timeTaken < 0 ? '' : millisecondsToStr(stepsRemaining * timeTaken * 1000))
            outputMsg.innerHTML = `Batch ${task.batchesDone} of ${batchCount}. Generating image(s): ${percent}%. Time remaining (approx): ${timeRemaining}`
            outputMsg.style.display = 'block'
            progressBarInner.style.width = `${percent}%`

            if (stepUpdate.output) {
                showImages(reqBody, stepUpdate, outputContainer, true)
            }
        }
    }
}

function abortTask(task) {
    if (!task.isProcessing) {
        return false
    }
    task.isProcessing = false
    task.progressBar.classList.remove("active")
    task['taskStatusLabel'].style.display = 'none'
    task['stopTask'].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    if (!task.instances?.some((r) => r.isPending)) {
        return
    }
    task.instances.forEach((instance) => {
        try {
            instance.abort()
        } catch (e) {
            console.error(e)
        }
    })
}

function onTaskErrorHandler(task, reqBody, instance, reason) {
    if (!task.isProcessing) {
        return
    }
    console.log('Render request %o, Instance: %o, Error: %s', reqBody, instance, reason)
    abortTask(task)
    const outputMsg = task['outputMsg']
    logError('Stable Diffusion had an error. Please check the logs in the command-line window. <br/><br/>' + reason + '<br/><pre>' + reason.stack + '</pre>', task, outputMsg)
    setStatus('request', 'error', 'error')
}

function onTaskCompleted(task, reqBody, instance, outputContainer, stepUpdate) {
    if (typeof stepUpdate === 'object') {
        if (stepUpdate.status === 'succeeded') {
            showImages(reqBody, stepUpdate, outputContainer, false)
        } else {
            task.isProcessing = false
            const outputMsg = task['outputMsg']
            let msg = ''
            if ('detail' in stepUpdate && typeof stepUpdate.detail === 'string' && stepUpdate.detail.length > 0) {
                msg = stepUpdate.detail
                if (msg.toLowerCase().includes('out of memory')) {
                    msg += `<br/><br/>
                            <b>Suggestions</b>:
                            <br/>
                            1. If you have set an initial image, please try reducing its dimension to ${MAX_INIT_IMAGE_DIMENSION}x${MAX_INIT_IMAGE_DIMENSION} or smaller.<br/>
                            2. Try picking a lower level in the '<em>GPU Memory Usage</em>' setting (in the '<em>Settings</em>' tab).<br/>
                            3. Try generating a smaller image.<br/>`
                }
            } else {
                msg = `Unexpected Read Error:<br/><pre>StepUpdate: ${JSON.stringify(stepUpdate, undefined, 4)}</pre>`
            }
            logError(msg, stepUpdate, outputMsg)
        }
    }
    if (task.isProcessing && task.batchesDone < task.batchCount) {
        task['taskStatusLabel'].innerText = "Pending"
        task['taskStatusLabel'].classList.add('waitingTaskLabel')
        task['taskStatusLabel'].classList.remove('activeTaskLabel')
        return
    }
    if ('instances' in task && task.instances.some((ins) => ins != instance && ins.isPending)) {
        return
    }

    task.isProcessing = false
    task['stopTask'].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    task['taskStatusLabel'].style.display = 'none'

    let time = millisecondsToStr( Date.now() - task.startTime )

    if (task.batchesDone == task.batchCount) {
        if (!task.outputMsg.innerText.toLowerCase().includes('error')) {
            task.outputMsg.innerText = `Processed ${task.numOutputsTotal} images in ${time}`
        }
        task.progressBar.style.height = "0px"
        task.progressBar.style.border = "0px solid var(--background-color3)"
        task.progressBar.classList.remove("active")
        setStatus('request', 'done', 'success')
    } else {
        task.outputMsg.innerText += `. Task ended after ${time}`
    }

    if (randomSeedField.checked) {
        seedField.value = task.seed
    }

    if (SD.activeTasks.size > 0) {
        return
    }
    const uncompletedTasks = getUncompletedTaskEntries()
    if (uncompletedTasks && uncompletedTasks.length > 0) {
        return
    }

    if (pauseClient) { 
        resumeBtn.click() 
    }
    renderButtons.style.display = 'none'
    renameMakeImageButton()

    if (isSoundEnabled()) {
        playSound()
    }
}


async function onTaskStart(task) {
    if (!task.isProcessing || task.batchesDone >= task.batchCount) {
        return
    }

    if (typeof task.startTime !== 'number') {
        task.startTime = Date.now()
    }
    if (!('instances' in task)) {
        task['instances'] = []
    }

    task['stopTask'].innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop'
    task['taskStatusLabel'].innerText = "Starting"
    task['taskStatusLabel'].classList.add('waitingTaskLabel')

    let newTaskReqBody = task.reqBody
    if (task.batchCount > 1) {
        // Each output render batch needs it's own task reqBody instance to avoid altering the other runs after they are completed.
        newTaskReqBody = Object.assign({}, task.reqBody)
        if (task.batchesDone == task.batchCount-1) { 
            // Last batch of the task
            // If the number of parallel jobs is no factor of the total number of images, the last batch must create less than "parallel jobs count" images
            // E.g. with numOutputsTotal = 6 and num_outputs = 5, the last batch shall only generate 1 image.
            newTaskReqBody.num_outputs = task.numOutputsTotal - task.reqBody.num_outputs * (task.batchCount-1)
        }
    }

    const startSeed = task.seed || newTaskReqBody.seed
    const genSeeds = Boolean(typeof newTaskReqBody.seed !== 'number' || (newTaskReqBody.seed === task.seed && task.numOutputsTotal > 1))
    if (genSeeds) {
        newTaskReqBody.seed = parseInt(startSeed) + (task.batchesDone * task.reqBody.num_outputs)
    }

    // Update the seed *before* starting the processing so it's retained if user stops the task
    if (randomSeedField.checked) {
        seedField.value = task.seed
    }

    const outputContainer = document.createElement('div')
    outputContainer.className = 'img-batch'
    task.outputContainer.insertBefore(outputContainer, task.outputContainer.firstChild)

    const eventInfo = {reqBody:newTaskReqBody}
    const callbacksPromises = PLUGINS['TASK_CREATE'].map((hook) => {
        if (typeof hook !== 'function') {
            console.error('The provided TASK_CREATE hook is not a function. Hook: %o', hook)
            return Promise.reject(new Error('hook is not a function.'))
        }
        try {
            return Promise.resolve(hook.call(task, eventInfo))
        } catch (err) {
            console.error(err)
            return Promise.reject(err)
        }
    })
    await Promise.allSettled(callbacksPromises)
    let instance = eventInfo.instance
    if (!instance) {
        const factory = PLUGINS.OUTPUTS_FORMATS.get(eventInfo.reqBody?.output_format || newTaskReqBody.output_format)
        if (factory) {
            instance = await Promise.resolve(factory(eventInfo.reqBody || newTaskReqBody))
        }
        if (!instance) {
            console.error(`${factory ? "Factory " + String(factory) : 'No factory defined'} for output format ${eventInfo.reqBody?.output_format || newTaskReqBody.output_format}. Instance is ${instance || 'undefined'}. Using default renderer.`)
            instance = new SD.RenderTask(eventInfo.reqBody || newTaskReqBody)
        }
    }

    task['instances'].push(instance)
    task.batchesDone++

    instance.enqueue(getTaskUpdater(task, newTaskReqBody, outputContainer)).then(
        (renderResult) => {
            onTaskCompleted(task, newTaskReqBody, instance, outputContainer, renderResult)
        },
        (reason) => {
            onTaskErrorHandler(task, newTaskReqBody, instance, reason)
        }
    )

    setStatus('request', 'fetching..')
    renderButtons.style.display = 'flex'
    renameMakeImageButton()
    updateInitialText()
}

/* Hover effect for the init image in the task list */
function createInitImageHover(taskEntry) {
    var $tooltip = $( taskEntry.querySelector('.task-fs-initimage') )
    var img = document.createElement('img')
    img.src = taskEntry.querySelector('div.task-initimg > img').src
    $tooltip.append(img)
    $tooltip.append(`<div class="top-right"><button>Use as Input</button></div>`)
    $tooltip.find('button').on('click', (e) => {
        e.stopPropagation()
        onUseAsInputClick(null,img) 
    })
}

let startX, startY;
function onTaskEntryDragOver(event) {
    imagePreview.querySelectorAll(".imageTaskContainer").forEach(itc => {
        if(itc != event.target.closest(".imageTaskContainer")){
            itc.classList.remove('dropTargetBefore','dropTargetAfter');
        }
    });
    if(event.target.closest(".imageTaskContainer")){
        if(startX && startY){
            if(event.target.closest(".imageTaskContainer").offsetTop > startY){
                event.target.closest(".imageTaskContainer").classList.add('dropTargetAfter');
            }else if(event.target.closest(".imageTaskContainer").offsetTop < startY){
                event.target.closest(".imageTaskContainer").classList.add('dropTargetBefore');
            }else if (event.target.closest(".imageTaskContainer").offsetLeft > startX){
                event.target.closest(".imageTaskContainer").classList.add('dropTargetAfter');
            }else if (event.target.closest(".imageTaskContainer").offsetLeft < startX){
                event.target.closest(".imageTaskContainer").classList.add('dropTargetBefore');
            }
        }
    }
}

function generateConfig({ label, value, visible, cssKey }) {
    if (!visible) return null;
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
        .filter(obj => obj)
}

function createTaskConfig(task) {
    return getVisibleConfig(taskConfigSetup, task).join('</span>,&nbsp;</div>')
}

function createTask(task) {
    let taskConfig = ''

    if (task.reqBody.init_image !== undefined) {
        let h = 80
        let w = task.reqBody.width * h / task.reqBody.height >>0
        taskConfig += `<div class="task-initimg" style="float:left;"><img style="width:${w}px;height:${h}px;" src="${task.reqBody.init_image}"><div class="task-fs-initimage"></div></div>`
    }

    taskConfig += `<div class="taskConfigData">${createTaskConfig(task)}</span></div></div>`;

    let taskEntry = document.createElement('div')
    taskEntry.id = `imageTaskContainer-${Date.now()}`
    taskEntry.className = 'imageTaskContainer'
    taskEntry.innerHTML = ` <div class="header-content panel collapsible active">
                                <i class="drag-handle fa-solid fa-grip"></i>
                                <div class="taskStatusLabel">Enqueued</div>
                                <button class="secondaryButton stopTask"><i class="fa-solid fa-trash-can"></i> Remove</button>
                                <button class="tertiaryButton useSettings"><i class="fa-solid fa-redo"></i> Use these settings</button>
                                <div class="preview-prompt"></div>
                                <div class="taskConfig">${taskConfig}</div>
                                <div class="outputMsg"></div>
                                <div class="progress-bar active"><div></div></div>
                            </div>
                            <div class="collapsible-content">
                                <div class="img-preview">
                            </div>`

    createCollapsibles(taskEntry)

    let draghandle = taskEntry.querySelector('.drag-handle')
    draghandle.addEventListener('mousedown', (e) => {
        taskEntry.setAttribute('draggable', true)
    })
    // Add a debounce delay to allow mobile to bouble tap.
    draghandle.addEventListener('mouseup', debounce((e) => {
        taskEntry.setAttribute('draggable', false)
    }, 2000))
    draghandle.addEventListener('click', (e) => {
        e.preventDefault() // Don't allow the results to be collapsed...
    })
    taskEntry.addEventListener('dragend', (e) => {
        taskEntry.setAttribute('draggable', false);
        imagePreview.querySelectorAll(".imageTaskContainer").forEach(itc => {
            itc.classList.remove('dropTargetBefore','dropTargetAfter');
        });
        imagePreview.removeEventListener("dragover", onTaskEntryDragOver );
    })
    taskEntry.addEventListener('dragstart', function(e) {
        imagePreview.addEventListener("dragover", onTaskEntryDragOver );
        e.dataTransfer.setData("text/plain", taskEntry.id);
        startX = e.target.closest(".imageTaskContainer").offsetLeft;
        startY = e.target.closest(".imageTaskContainer").offsetTop;
    })

    if (task.reqBody.init_image !== undefined) {
        createInitImageHover(taskEntry)
    }

    task['taskStatusLabel'] = taskEntry.querySelector('.taskStatusLabel')
    task['outputContainer'] = taskEntry.querySelector('.img-preview')
    task['outputMsg'] = taskEntry.querySelector('.outputMsg')
    task['previewPrompt'] = taskEntry.querySelector('.preview-prompt')
    task['progressBar'] = taskEntry.querySelector('.progress-bar')
    task['stopTask'] = taskEntry.querySelector('.stopTask')

    task['stopTask'].addEventListener('click', (e) => {
        e.stopPropagation()

        if (task['isProcessing']) {
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

    task['useSettings'] = taskEntry.querySelector('.useSettings')
    task['useSettings'].addEventListener('click', function(e) {
        e.stopPropagation()
        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
    })

    task.isProcessing = true
    taskEntry = imagePreviewContent.insertBefore(taskEntry, previewTools.nextSibling)
    htmlTaskMap.set(taskEntry, task)

    task.previewPrompt.innerText = task.reqBody.prompt
    if (task.previewPrompt.innerText.trim() === '') {
        task.previewPrompt.innerHTML = '&nbsp;' // allows the results to be collapsed
    }
    return taskEntry.id
}

function getCurrentUserRequest() {
    const numOutputsTotal = parseInt(numOutputsTotalField.value)
    const numOutputsParallel = parseInt(numOutputsParallelField.value)
    const seed = (randomSeedField.checked ? Math.floor(Math.random() * 10000000) : parseInt(seedField.value))

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
            width: parseInt(widthField.value),
            height: parseInt(heightField.value),
            // allow_nsfw: allowNSFWField.checked,
            vram_usage_level: vramUsageLevelField.value,
            sampler_name: samplerField.value,
            //render_device: undefined, // Set device affinity. Prefer this device, but wont activate.
            use_stable_diffusion_model: stableDiffusionModelField.value,
            use_vae_model: vaeModelField.value,
            stream_progress_updates: true,
            stream_image_progress: (numOutputsTotal > 50 ? false : streamImageProgressField.checked),
            show_only_filtered_image: showOnlyFilteredImageField.checked,
            block_nsfw: blockNSFWField.checked,
            output_format: outputFormatField.value,
            output_quality: parseInt(outputQualityField.value),
            output_lossless: outputLosslessField.checked,
            metadata_output_format: metadataOutputFormatField.value,
            original_prompt: promptField.value,
            active_tags: (activeTags.map(x => x.name)),
            inactive_tags: (activeTags.filter(tag => tag.inactive === true).map(x => x.name))
        }
    }
    if (IMAGE_REGEX.test(initImagePreview.src)) {
        newTask.reqBody.init_image = initImagePreview.src
        newTask.reqBody.prompt_strength = parseFloat(promptStrengthField.value)
        // if (IMAGE_REGEX.test(maskImagePreview.src)) {
        //     newTask.reqBody.mask = maskImagePreview.src
        // }
        if (maskSetting.checked) {
            newTask.reqBody.mask = imageInpainter.getImg()
        }
        newTask.reqBody.preserve_init_image_color_profile = applyColorCorrectionField.checked
        if (!testDiffusers.checked) {
            newTask.reqBody.sampler_name = 'ddim'
        }
    }
    if (saveToDiskField.checked && diskPathField.value.trim() !== '') {
        newTask.reqBody.save_to_disk_path = diskPathField.value.trim()
    }
    if (useFaceCorrectionField.checked) {
        newTask.reqBody.use_face_correction = gfpganModelField.value
    }
    if (useUpscalingField.checked) {
        newTask.reqBody.use_upscale = upscaleModelField.value
        newTask.reqBody.upscale_amount = upscaleAmountField.value
    }
    if (hypernetworkModelField.value) {
        newTask.reqBody.use_hypernetwork_model = hypernetworkModelField.value
        newTask.reqBody.hypernetwork_strength = parseFloat(hypernetworkStrengthField.value)
    }
    if (testDiffusers.checked) {
        newTask.reqBody.use_lora_model = loraModelField.value
    }
    return newTask
}

function getPrompts(prompts) {
    if (typeof prompts === 'undefined') {
        prompts = promptField.value
    }
    if (prompts.trim() === '' && activeTags.length === 0) {
        return ['']
    }

    let promptsToMake = []
    if (prompts.trim() !== '') {
        prompts = prompts.split('\n')
        prompts = prompts.map(prompt => prompt.trim())
        prompts = prompts.filter(prompt => prompt !== '')
    
        promptsToMake = applyPermuteOperator(prompts)
        promptsToMake = applySetOperator(promptsToMake)
    }
    const newTags = activeTags.filter(tag => tag.inactive === undefined || tag.inactive === false)
    if (newTags.length > 0) {
        const promptTags = newTags.map(x => x.name).join(", ")
        if (promptsToMake.length > 0) {
            promptsToMake = promptsToMake.map((prompt) => `${prompt}, ${promptTags}`)
        }
        else
        {
            promptsToMake.push(promptTags)
        }
    }

    promptsToMake = applyPermuteOperator(promptsToMake)
    promptsToMake = applySetOperator(promptsToMake)

    PLUGINS['GET_PROMPTS_HOOK'].forEach(fn => { promptsToMake = fn(promptsToMake) })

    return promptsToMake
}

function applySetOperator(prompts) {
    let promptsToMake = []
    let braceExpander = new BraceExpander()
    prompts.forEach(prompt => {
        let expandedPrompts = braceExpander.expand(prompt)
        promptsToMake = promptsToMake.concat(expandedPrompts)
    })

    return promptsToMake
}

function applyPermuteOperator(prompts) {
    let promptsToMake = []
    prompts.forEach(prompt => {
        let promptMatrix = prompt.split('|')
        prompt = promptMatrix.shift().trim()
        promptsToMake.push(prompt)

        promptMatrix = promptMatrix.map(p => p.trim())
        promptMatrix = promptMatrix.filter(p => p !== '')

        if (promptMatrix.length > 0) {
            let promptPermutations = permutePrompts(prompt, promptMatrix)
            promptsToMake = promptsToMake.concat(promptPermutations)
        }
    })

    return promptsToMake
}

function permutePrompts(promptBase, promptMatrix) {
    let prompts = []
    let permutations = permute(promptMatrix)
    permutations.forEach(perm => {
        let prompt = promptBase

        if (perm.length > 0) {
            let promptAddition = perm.join(', ')
            if (promptAddition.trim() === '') {
                return
            }

            prompt += ', ' + promptAddition
        }

        prompts.push(prompt)
    })

    return prompts
}

// create a file name with embedded prompt and metadata
// for easier cateloging and comparison
function createFileName(prompt, seed, steps, guidance, outputFormat) {

    // Most important information is the prompt
    let underscoreName = prompt.replace(/[^a-zA-Z0-9]/g, '_')
    underscoreName = underscoreName.substring(0, 70)

    // name and the top level metadata
    let fileName = `${underscoreName}_S${seed}_St${steps}_G${guidance}.${outputFormat}`

    return fileName
}

async function stopAllTasks() {
    getUncompletedTaskEntries().forEach((taskEntry) => {
        const taskStatusLabel = taskEntry.querySelector('.taskStatusLabel')
        if (taskStatusLabel) {
            taskStatusLabel.style.display = 'none'
        }
        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            return
        }
        abortTask(task)
    })
}

function updateInitialText() {
    if (document.querySelector('.imageTaskContainer') === null) {
        if (undoBuffer.length == 0) {
            previewTools.classList.add('displayNone')
        }
        initialText.classList.remove('displayNone')
    } else {
        initialText.classList.add('displayNone')
        previewTools.classList.remove('displayNone')
    }
}

function removeTask(taskToRemove) {
    undoableRemove(taskToRemove)
    updateInitialText()
}

clearAllPreviewsBtn.addEventListener('click', (e) => { shiftOrConfirm(e, "Clear all the results and tasks in this window?", async function() {
    await stopAllTasks()

    let taskEntries = document.querySelectorAll('.imageTaskContainer')
    taskEntries.forEach(removeTask)
})})

/* Download images popup */
showDownloadPopupBtn.addEventListener("click", (e) => { saveAllImagesPopup.classList.add("active") })

saveAllZipToggle.addEventListener('change', (e) => {
    if (saveAllZipToggle.checked) {
        saveAllFoldersOption.classList.remove('displayNone')
    } else {
        saveAllFoldersOption.classList.add('displayNone')
    }
})

// convert base64 to raw binary data held in a string
function dataURItoBlob(dataURI) {
    var byteString = atob(dataURI.split(',')[1])

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0]

    // write the bytes of the string to an ArrayBuffer
    var ab = new ArrayBuffer(byteString.length)

    // create a view into the buffer
    var ia = new Uint8Array(ab)

    // set the bytes of the buffer to the correct values
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i)
    }

    // write the ArrayBuffer to a blob, and you're done
    return new Blob([ab], {type: mimeString})
}

function downloadAllImages() {
    let i = 0

    let optZIP  = saveAllZipToggle.checked
    let optTree = optZIP && saveAllTreeToggle.checked
    let optJSON = saveAllJSONToggle.checked
    
    let zip = new JSZip()
    let folder = zip

    document.querySelectorAll(".imageTaskContainer").forEach(container => {
        if (optTree) {
            let name = ++i + '-' + container.querySelector('.preview-prompt').textContent.replace(/[^a-zA-Z0-9]/g, '_').substring(0,25)
            folder = zip.folder(name)
        }
        container.querySelectorAll(".imgContainer img").forEach(img => {
            let imgItem = img.closest('.imgItem')

            if (imgItem.style.display === 'none') {
                return
            }

            let req = imageRequest[img.dataset['imagecounter']]
            if (optZIP) {
                let suffix = img.dataset['imagecounter'] + '.' + req['output_format']
                folder.file(getDownloadFilename(img, suffix), dataURItoBlob(img.src))
                if (optJSON) {
                    suffix = img.dataset['imagecounter'] + '.json'
                    folder.file(getDownloadFilename(img, suffix), JSON.stringify(req, null, 2))
                }
            } else {
                setTimeout(() => {imgItem.querySelector('.download-img').click()}, i*200)
                i = i+1
                if (optJSON) {
                    setTimeout(() => {imgItem.querySelector('.download-json').click()}, i*200)
                    i = i+1
                }
            }
        })
    })
    if (optZIP) {
        let now = Date.now().toString(36).toUpperCase()
        zip.generateAsync({type:"blob"}).then(function (blob) { 
            saveAs(blob, `EasyDiffusion-Images-${now}.zip`);
        })
    }

}    

saveAllImagesBtn.addEventListener('click', (e) => { downloadAllImages() })

stopImageBtn.addEventListener('click', (e) => { shiftOrConfirm(e, "Stop all the tasks?", async function(e) {
    await stopAllTasks()
})})

widthField.addEventListener('change', onDimensionChange)
heightField.addEventListener('change', onDimensionChange)

function renameMakeImageButton() {
    let totalImages = Math.max(parseInt(numOutputsTotalField.value), parseInt(numOutputsParallelField.value)) * getPrompts().length
    let imageLabel = 'Image'
    if (totalImages > 1) {
        imageLabel = totalImages + ' Images'
    }
    if (SD.activeTasks.size == 0) {
        makeImageBtn.innerText = 'Make ' + imageLabel
    } else {
        makeImageBtn.innerText = 'Enqueue Next ' + imageLabel
    }
}
numOutputsTotalField.addEventListener('change', renameMakeImageButton)
numOutputsTotalField.addEventListener('keyup', debounce(renameMakeImageButton, 300))
numOutputsParallelField.addEventListener('change', renameMakeImageButton)
numOutputsParallelField.addEventListener('keyup', debounce(renameMakeImageButton, 300))

function onDimensionChange() {
    let widthValue = parseInt(widthField.value)
    let heightValue = parseInt(heightField.value)
    if (!initImagePreviewContainer.classList.contains("has-image")) {
        imageEditor.setImage(null, widthValue, heightValue)
    }
    else {
        imageInpainter.setImage(initImagePreview.src, widthValue, heightValue)
    }
    if ( widthValue < 512 && heightValue < 512 ) {
        smallImageWarning.classList.remove('displayNone')
    } else {
        smallImageWarning.classList.add('displayNone')
    }
}

diskPathField.disabled = !saveToDiskField.checked
metadataOutputFormatField.disabled = !saveToDiskField.checked

gfpganModelField.disabled = !useFaceCorrectionField.checked
useFaceCorrectionField.addEventListener('change', function(e) {
    gfpganModelField.disabled = !this.checked
})

upscaleModelField.disabled = !useUpscalingField.checked
upscaleAmountField.disabled = !useUpscalingField.checked
useUpscalingField.addEventListener('change', function(e) {
    upscaleModelField.disabled = !this.checked
    upscaleAmountField.disabled = !this.checked
})

makeImageBtn.addEventListener('click', makeImage)

document.onkeydown = function(e) {
    if (e.ctrlKey && e.code === 'Enter') {
        makeImage()
        e.preventDefault()
    }
}

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

guidanceScaleSlider.addEventListener('input', updateGuidanceScale)
guidanceScaleField.addEventListener('input', updateGuidanceScaleSlider)
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

promptStrengthSlider.addEventListener('input', updatePromptStrength)
promptStrengthField.addEventListener('input', updatePromptStrengthSlider)
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

hypernetworkStrengthSlider.addEventListener('input', updateHypernetworkStrength)
hypernetworkStrengthField.addEventListener('input', updateHypernetworkStrengthSlider)
updateHypernetworkStrength()

function updateHypernetworkStrengthContainer() {
    document.querySelector("#hypernetwork_strength_container").style.display = (hypernetworkModelField.value === "" ? 'none' : '')
}
hypernetworkModelField.addEventListener('change', updateHypernetworkStrengthContainer)
updateHypernetworkStrengthContainer()

/********************* LoRA alpha **********************/
function updateLoraAlpha() {
    loraAlphaField.value = loraAlphaSlider.value / 100
    loraAlphaField.dispatchEvent(new Event("change"))
}

function updateLoraAlphaSlider() {
    if (loraAlphaField.value < 0) {
        loraAlphaField.value = 0
    } else if (loraAlphaField.value > 0.99) {
        loraAlphaField.value = 0.99
    }

    loraAlphaSlider.value = loraAlphaField.value * 100
    loraAlphaSlider.dispatchEvent(new Event("change"))
}

loraAlphaSlider.addEventListener('input', updateLoraAlpha)
loraAlphaField.addEventListener('input', updateLoraAlphaSlider)
updateLoraAlpha()

// function updateLoraAlphaContainer() {
//     document.querySelector("#lora_alpha_container").style.display = (loraModelField.value === "" ? 'none' : '')
// }
// loraModelField.addEventListener('change', updateLoraAlphaContainer)
// updateLoraAlphaContainer()
document.querySelector("#lora_alpha_container").style.display = 'none'

/********************* JPEG/WEBP Quality **********************/
function updateOutputQuality() {
    outputQualityField.value =  0 | outputQualitySlider.value
    outputQualityField.dispatchEvent(new Event("change"))
}

function updateOutputQualitySlider() {
    if (outputQualityField.value < 10) {
        outputQualityField.value = 10
    } else if (outputQualityField.value > 95) {
        outputQualityField.value = 95
    }

    outputQualitySlider.value =  0 | outputQualityField.value
    outputQualitySlider.dispatchEvent(new Event("change"))
}

outputQualitySlider.addEventListener('input', updateOutputQuality)
outputQualityField.addEventListener('input', debounce(updateOutputQualitySlider, 1500))
updateOutputQuality()

function updateOutputQualityVisibility() {
    if (outputFormatField.value === 'webp') {
        outputLosslessContainer.classList.remove('displayNone')
        if (outputLosslessField.checked) {
            outputQualityRow.classList.add('displayNone')
        } else {
            outputQualityRow.classList.remove('displayNone')
        }
    }
    else if (outputFormatField.value === 'png') {
        outputQualityRow.classList.add('displayNone')
        outputLosslessContainer.classList.add('displayNone')
    } else {
        outputQualityRow.classList.remove('displayNone')
        outputLosslessContainer.classList.add('displayNone')
    }
}

outputFormatField.addEventListener('change', updateOutputQualityVisibility)
outputLosslessField.addEventListener('change', updateOutputQualityVisibility)
/********************* Zoom Slider **********************/
thumbnailSizeField.addEventListener('change', () => {
    (function (s) {
        for (var j =0; j < document.styleSheets.length; j++) {
            let cssSheet = document.styleSheets[j]
            for (var i = 0; i < cssSheet.cssRules.length; i++) {
                var rule = cssSheet.cssRules[i];
                if (rule.selectorText == "div.img-preview img") {
                  rule.style['max-height'] = s+'vh';
                  rule.style['max-width'] = s+'vw';
                  return;
                }
            }
        }
    })(thumbnailSizeField.value)
})

function onAutoScrollUpdate() {
    if (autoScroll.checked) {
        autoscrollBtn.classList.add('pressed')
    } else {
        autoscrollBtn.classList.remove('pressed')
    }
    autoscrollBtn.querySelector(".state").innerHTML = (autoScroll.checked ? "ON" : "OFF")
}
autoscrollBtn.addEventListener('click', function() {
    autoScroll.checked = !autoScroll.checked
    autoScroll.dispatchEvent(new Event("change"))
    onAutoScrollUpdate()
})
autoScroll.addEventListener('change', onAutoScrollUpdate)

function checkRandomSeed() {
    if (randomSeedField.checked) {
        seedField.disabled = true
        //seedField.value = "0" // This causes the seed to be lost if the user changes their mind after toggling the checkbox
    } else {
        seedField.disabled = false
    }
}
randomSeedField.addEventListener('input', checkRandomSeed)
checkRandomSeed()

function loadImg2ImgFromFile() {
    if (initImageSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = initImageSelector.files[0]

    reader.addEventListener('load', function(event) {
        initImagePreview.src = reader.result
    })

    if (file) {
        reader.readAsDataURL(file)
    }
}
initImageSelector.addEventListener('change', loadImg2ImgFromFile)
loadImg2ImgFromFile()

function img2imgLoad() {
    promptStrengthContainer.style.display = 'table-row'
    if (!testDiffusers.checked) {
        samplerSelectionContainer.style.display = "none"
    }
    initImagePreviewContainer.classList.add("has-image")
    colorCorrectionSetting.style.display = ''

    initImageSizeBox.textContent = initImagePreview.naturalWidth + " x " + initImagePreview.naturalHeight
    imageEditor.setImage(this.src, initImagePreview.naturalWidth, initImagePreview.naturalHeight)
    imageInpainter.setImage(this.src, parseInt(widthField.value), parseInt(heightField.value))
}

function img2imgUnload() {
    initImageSelector.value = null
    initImagePreview.src = ''
    maskSetting.checked = false

    promptStrengthContainer.style.display = "none"
    if (!testDiffusers.checked) {
        samplerSelectionContainer.style.display = ""
    }
    initImagePreviewContainer.classList.remove("has-image")
    colorCorrectionSetting.style.display = 'none'
    imageEditor.setImage(null, parseInt(widthField.value), parseInt(heightField.value))

}
initImagePreview.addEventListener('load', img2imgLoad)
initImageClearBtn.addEventListener('click', img2imgUnload)

maskSetting.addEventListener('click', function() {
    onDimensionChange()
})

promptsFromFileBtn.addEventListener('click', function() {
    promptsFromFileSelector.click()
})

promptsFromFileSelector.addEventListener('change', async function() {
    if (promptsFromFileSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = promptsFromFileSelector.files[0]

    reader.addEventListener('load', async function() {
        await parseContent(reader.result)
    })

    if (file) {
        reader.readAsText(file)
    }
})

/* setup popup handlers */
document.querySelectorAll('.popup').forEach(popup => {
    popup.addEventListener('click', event => {
        if (event.target == popup) {
            popup.classList.remove("active")
        }
    })
    var closeButton = popup.querySelector(".close-button")
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            popup.classList.remove("active")
        })
    }
})

var tabElements = []
function selectTab(tab_id) {
    let tabInfo = tabElements.find(t => t.tab.id == tab_id)
    if (!tabInfo.tab.classList.contains("active")) {
        tabElements.forEach(info => {
            if (info.tab.classList.contains("active") && info.tab.parentNode === tabInfo.tab.parentNode) {
                info.tab.classList.toggle("active")
                info.content.classList.toggle("active")
            }
        })
        tabInfo.tab.classList.toggle("active")
        tabInfo.content.classList.toggle("active")
    }
    document.dispatchEvent(new CustomEvent('tabClick', { detail: tabInfo }))
}
function linkTabContents(tab) {
    var name = tab.id.replace("tab-", "")
    var content = document.getElementById(`tab-content-${name}`)
    tabElements.push({
        name: name,
        tab: tab,
        content: content
    })

    tab.addEventListener("click", event => selectTab(tab.id))
}
function isTabActive(tab) {
    return tab.classList.contains("active")
}

let pauseClient = false

function resumeClient() {
    if (pauseClient) {
        document.body.classList.remove('wait-pause')
        document.body.classList.add('pause')
    }
    return new Promise(resolve => {
        let playbuttonclick = function () {
            resumeBtn.removeEventListener("click", playbuttonclick);
            resolve("resolved");
        }
        resumeBtn.addEventListener("click", playbuttonclick)
    })
}

promptField.addEventListener("input", debounce( renameMakeImageButton, 1000) )


pauseBtn.addEventListener("click", function () {
    pauseClient = true
    pauseBtn.style.display="none"
    resumeBtn.style.display = "inline"
    document.body.classList.add('wait-pause')
})

resumeBtn.addEventListener("click", function () {
    pauseClient = false
    resumeBtn.style.display = "none"
    pauseBtn.style.display = "inline"
    document.body.classList.remove('pause')
    document.body.classList.remove('wait-pause')
})

/* Pause function */
document.querySelectorAll(".tab").forEach(linkTabContents)

window.addEventListener("beforeunload", function(e) {
    const msg = "Unsaved pictures will be lost!";

    let elementList = document.getElementsByClassName("imageTaskContainer");
    if (elementList.length != 0) {
        e.preventDefault();
        (e || window.event).returnValue = msg;
        return msg;
    } else {
        return true;
    }
});

createCollapsibles()
prettifyInputs(document);

// set the textbox as focused on start
promptField.focus()
promptField.selectionStart = promptField.value.length
