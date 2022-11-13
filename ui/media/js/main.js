"use strict" // Opt in to a restricted variant of JavaScript
const MAX_INIT_IMAGE_DIMENSION = 768
const MIN_GPUS_TO_SHOW_SELECTION = 2

const IMAGE_REGEX = new RegExp('data:image/[A-Za-z]+;base64')
const htmlTaskMap = new WeakMap()

let promptField = document.querySelector('#prompt')
let promptsFromFileSelector = document.querySelector('#prompt_from_file')
let promptsFromFileBtn = document.querySelector('#promptsFromFileBtn')
let negativePromptField = document.querySelector('#negative_prompt')
let numOutputsTotalField = document.querySelector('#num_outputs_total')
let numOutputsParallelField = document.querySelector('#num_outputs_parallel')
let numInferenceStepsField = document.querySelector('#num_inference_steps')
let guidanceScaleSlider = document.querySelector('#guidance_scale_slider')
let guidanceScaleField = document.querySelector('#guidance_scale')
let randomSeedField = document.querySelector("#random_seed")
let seedField = document.querySelector('#seed')
let widthField = document.querySelector('#width')
let heightField = document.querySelector('#height')
let initImageSelector = document.querySelector("#init_image")
let initImagePreview = document.querySelector("#init_image_preview")
let initImageSizeBox = document.querySelector("#init_image_size_box")
let maskImageSelector = document.querySelector("#mask")
let maskImagePreview = document.querySelector("#mask_preview")
let turboField = document.querySelector('#turbo')
let useCPUField = document.querySelector('#use_cpu')
let useGPUsField = document.querySelector('#use_gpus')
let useFullPrecisionField = document.querySelector('#use_full_precision')
let saveToDiskField = document.querySelector('#save_to_disk')
let diskPathField = document.querySelector('#diskPath')
// let allowNSFWField = document.querySelector("#allow_nsfw")
let useBetaChannelField = document.querySelector("#use_beta_channel")
let promptStrengthSlider = document.querySelector('#prompt_strength_slider')
let promptStrengthField = document.querySelector('#prompt_strength')
let samplerField = document.querySelector('#sampler')
let samplerSelectionContainer = document.querySelector("#samplerSelection")
let useFaceCorrectionField = document.querySelector("#use_face_correction")
let useUpscalingField = document.querySelector("#use_upscale")
let upscaleModelField = document.querySelector("#upscale_model")
let stableDiffusionModelField = document.querySelector('#stable_diffusion_model')
let vaeModelField = document.querySelector('#vae_model')
let outputFormatField = document.querySelector('#output_format')
let showOnlyFilteredImageField = document.querySelector("#show_only_filtered_image")
let updateBranchLabel = document.querySelector("#updateBranchLabel")
let streamImageProgressField = document.querySelector("#stream_image_progress")

let makeImageBtn = document.querySelector('#makeImage')
let stopImageBtn = document.querySelector('#stopImage')

let imagesContainer = document.querySelector('#current-images')
let initImagePreviewContainer = document.querySelector('#init_image_preview_container')
let initImageClearBtn = document.querySelector('.init_image_clear')
let promptStrengthContainer = document.querySelector('#prompt_strength_container')

let initialText = document.querySelector("#initial-text")
let previewTools = document.querySelector("#preview-tools")
let clearAllPreviewsBtn = document.querySelector("#clear-all-previews")

// let maskSetting = document.querySelector('#editor-inputs-mask_setting')
// let maskImagePreviewContainer = document.querySelector('#mask_preview_container')
// let maskImageClearBtn = document.querySelector('#mask_clear')
let maskSetting = document.querySelector('#enable_mask')

const processOrder = document.querySelector('#process_order_toggle')

let imagePreview = document.querySelector("#preview")
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
    while (moveTarget && typeof moveTarget === 'object' && moveTarget.parentNode !== imagePreview) {
        moveTarget = moveTarget.parentNode
    }
    if (moveTarget === initialText || moveTarget === previewTools) {
        moveTarget = null
    }
    if (moveTarget === movedTask) {
        return
    }
    if (moveTarget) {
        const childs = Array.from(imagePreview.children)
        if (moveTarget.nextSibling && childs.indexOf(movedTask) < childs.indexOf(moveTarget)) {
            // Move after the target if lower than current position.
            moveTarget = moveTarget.nextSibling
        }
    }
    const newNode = imagePreview.insertBefore(movedTask, moveTarget || previewTools.nextSibling)
    if (newNode === movedTask) {
        return
    }
    imagePreview.removeChild(movedTask)
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

document.querySelector('.drawing-board-control-navigation-back').innerHTML = '<i class="fa-solid fa-rotate-left"></i>'
document.querySelector('.drawing-board-control-navigation-forward').innerHTML = '<i class="fa-solid fa-rotate-right"></i>'

let maskResetButton = document.querySelector('.drawing-board-control-navigation-reset')
maskResetButton.innerHTML = 'Clear'
maskResetButton.style.fontWeight = 'normal'
maskResetButton.style.fontSize = '10pt'

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

function setServerStatus(msgType, msg) {
    switch(msgType) {
        case 'online':
            serverStatusColor.style.color = 'green'
            serverStatusMsg.style.color = 'green'
            serverStatusMsg.innerText = 'Stable Diffusion is ' + msg
            break
        case 'busy':
            serverStatusColor.style.color = 'rgb(200, 139, 0)'
            serverStatusMsg.style.color = 'rgb(200, 139, 0)'
            serverStatusMsg.innerText = 'Stable Diffusion is ' + msg
            break
        case 'error':
            serverStatusColor.style.color = 'red'
            serverStatusMsg.style.color = 'red'
            serverStatusMsg.innerText = 'Stable Diffusion has stopped'
            break
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
                        <span class="imgSeedLabel"></span>
                    </div>
                </div>
            `
            outputContainer.appendChild(imageItemElem)
        }
        const imageElem = imageItemElem.querySelector('img')
        imageElem.src = imageData
        imageElem.width = parseInt(imageWidth)
        imageElem.height = parseInt(imageHeight)
        imageElem.setAttribute('data-prompt', imagePrompt)
        imageElem.setAttribute('data-steps', imageInferenceSteps)
        imageElem.setAttribute('data-guidance', imageGuidanceScale)


        const imageInfo = imageItemElem.querySelector('.imgItemInfo')
        imageInfo.style.visibility = (livePreview ? 'hidden' : 'visible')

        if ('seed' in result && !imageElem.hasAttribute('data-seed')) {
            const req = Object.assign({}, reqBody, {
                seed: result?.seed || reqBody.seed
            })
            imageElem.setAttribute('data-seed', req.seed)
            const imageSeedLabel = imageItemElem.querySelector('.imgSeedLabel')
            imageSeedLabel.innerText = 'Seed: ' + req.seed

            let buttons = [
                { text: 'Use as Input', on_click: onUseAsInputClick },
                { text: 'Download', on_click: onDownloadImageClick },
                { text: 'Make Similar Images', on_click: onMakeSimilarClick },
                { text: 'Draw another 25 steps', on_click: onContinueDrawingClick },
                { text: 'Upscale', on_click: onUpscaleClick, filter: (req, img) => !req.use_upscale },
                { text: 'Fix Faces', on_click: onFixFacesClick, filter: (req, img) => !req.use_face_correction }
            ]

            // include the plugins
            buttons = buttons.concat(PLUGINS['IMAGE_INFO_BUTTONS'])

            const imgItemInfo = imageItemElem.querySelector('.imgItemInfo')
            const img = imageItemElem.querySelector('img')
            const createButton = function(btnInfo) {
                const newButton = document.createElement('button')
                newButton.classList.add('tasksBtns')
                newButton.innerText = btnInfo.text
                newButton.addEventListener('click', function() {
                    btnInfo.on_click(req, img)
                })
                imgItemInfo.appendChild(newButton)
            }
            buttons.forEach(btn => {
                if (btn.filter && btn.filter(req, img) === false) {
                    return
                }

                createButton(btn)
            })
        }
    })
}

function onUseAsInputClick(req, img) {
    const imgData = img.src

    initImageSelector.value = null
    initImagePreview.src = imgData

    initImagePreviewContainer.style.display = 'block'
    inpaintingEditorContainer.style.display = 'none'
    promptStrengthContainer.style.display = 'table-row'
    maskSetting.checked = false
    samplerSelectionContainer.style.display = 'none'
}

function onDownloadImageClick(req, img) {
    const imgData = img.src
    const imageSeed = img.getAttribute('data-seed')
    const imagePrompt = img.getAttribute('data-prompt')
    const imageInferenceSteps = img.getAttribute('data-steps')
    const imageGuidanceScale = img.getAttribute('data-guidance')

    const imgDownload = document.createElement('a')
    imgDownload.download = createFileName(imagePrompt, imageSeed, imageInferenceSteps, imageGuidanceScale, req['output_format'])
    imgDownload.href = imgData
    imgDownload.click()
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

    const newTaskRequest = modifyCurrentRequest(req, reqDiff, {
        num_outputs: 1, // this can be user-configurable in the future
        seed: imageSeed
    })

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
        use_face_correction: 'GFPGANv1.3'
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
    if (!SD.isServerAvailable()) {
        alert('The server is not available.')
        return
    }
    const taskTemplate = getCurrentUserRequest()
    const newTaskRequests = []
    getPrompts().forEach((prompt) => newTaskRequests.push(Object.assign({}, taskTemplate, {
        reqBody: Object.assign({ prompt: prompt }, taskTemplate.reqBody)
    })))
    newTaskRequests.forEach(createTask)

    initialText.style.display = 'none'
}
function onIdle() {
    for (const taskEntry of getUncompletedTaskEntries()) {
        if (SD.activeTasks.size >= 1) {
            continue
        }
        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            const taskStatusLabel = taskEntry.querySelector('.taskStatusLabel')
            taskStatusLabel.style.display = 'none'
            continue
        }
        onTaskStart(task)
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
                (sum, instance) => sum + (instance.isPending ? Math.max(0, instance.step || stepUpdate.step) / (instance.total_steps || stepUpdate.total_steps) : 1)
                , 0 // Initial value
            ) * stepUpdate.total_steps // Scale to current number of steps.
            const totalSteps = task.instances.reduce(
                (sum, instance) => sum + (instance.total_steps || stepUpdate.total_steps)
                , stepUpdate.total_steps * (batchCount - task.batchesDone) // Initial value at (unstarted task count * Nbr of steps)
            )
            const percent = Math.min(100, 100 * (overallStepCount / totalSteps)).toFixed(0)

            const timeTaken = stepUpdate.step_time // sec
            const stepsRemaining = Math.max(0, totalSteps - overallStepCount)
            const timeRemaining = (timeTaken < 0 ? '' : millisecondsToStr(stepsRemaining * timeTaken * 1000))
            outputMsg.innerHTML = `Batch ${task.batchesDone} of ${batchCount}. Generating image(s): ${percent}%. Time remaining (approx): ${timeRemaining}`
            outputMsg.style.display = 'block'

            progressBarInner.style.width = `${percent}%`
            if (percent == 100) {
                task.progressBar.style.height = "0px"
                task.progressBar.style.border = "0px solid var(--background-color3)"
                task.progressBar.classList.remove("active")
            }

            if (stepUpdate.output) {
                showImages(reqBody, stepUpdate, outputContainer, true)
            }
        }
    }
}

function getTaskErrorHandler(task, reqBody, instance, reason) {
    if (!task.isProcessing) {
        return
    }
    task.isProcessing = false
    console.log('Render request %o, Instance: %o, Error: %s', reqBody, instance, reason)
    const outputMsg = task['outputMsg']
    logError('Stable Diffusion had an error. Please check the logs in the command-line window. <br/><br/>' + reason + '<br/><pre>' + reason.stack + '</pre>', task, outputMsg)
    setStatus('request', 'error', 'error')
}

function onTaskCompleted(task, reqBody, instance, outputContainer, stepUpdate) {
    if (typeof stepUpdate !== 'object') {
        return
    }
    if (stepUpdate.status !== 'succeeded') {
        const outputMsg = task['outputMsg']
        let msg = ''
        if ('detail' in stepUpdate && typeof stepUpdate.detail === 'string' && stepUpdate.detail.length > 0) {
            msg = stepUpdate.detail
            if (msg.toLowerCase().includes('out of memory')) {
                msg += `<br/><br/>
                        <b>Suggestions</b>:
                        <br/>
                        1. If you have set an initial image, please try reducing its dimension to ${MAX_INIT_IMAGE_DIMENSION}x${MAX_INIT_IMAGE_DIMENSION} or smaller.<br/>
                        2. Try disabling the '<em>Turbo mode</em>' under '<em>Advanced Settings</em>'.<br/>
                        3. Try generating a smaller image.<br/>`
            }
        } else {
            msg = `Unexpected Read Error:<br/><pre>StepUpdate: ${JSON.stringify(stepUpdate, undefined, 4)}</pre>`
        }
        logError(msg, stepUpdate, outputMsg)
        return false
    }
    showImages(reqBody, stepUpdate, outputContainer, false)
    if (task.isProcessing && task.batchesDone < task.batchCount) {
        task['taskStatusLabel'].innerText = "Pending"
        task['taskStatusLabel'].classList.add('waitingTaskLabel')
        task['taskStatusLabel'].classList.remove('activeTaskLabel')
        return
    }
    if ('instances' in task && task.instances.some((ins) => ins != instance && ins.isPending)) {
        return
    }

    setStatus('request', 'done', 'success')

    task.isProcessing = false
    task['stopTask'].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    task['taskStatusLabel'].style.display = 'none'

    let time = Date.now() - task.startTime
    time /= 1000

    if (task.batchesDone == task.batchCount) {
        task.outputMsg.innerText = `Processed ${task.numOutputsTotal} images in ${time} seconds`
        task.progressBar.style.height = "0px"
        task.progressBar.style.border = "0px solid var(--background-color3)"
        task.progressBar.classList.remove("active")
        setStatus('request', 'done', 'success')
    } else {
        task.outputMsg.innerText += `Task ended after ${time} seconds`
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

    stopImageBtn.style.display = 'none'
    renameMakeImageButton()

    if (isSoundEnabled()) {
        playSound()
    }
}

function onTaskStart(task) {
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
    }

    const startSeed = task.seed || newTaskReqBody.seed
    const genSeeds = Boolean(typeof newTaskReqBody.seed !== 'number' || (newTaskReqBody.seed === task.seed && task.numOutputsTotal > 1))
    if (genSeeds) {
        newTaskReqBody.seed = parseInt(startSeed) + (task.batchesDone * newTaskReqBody.num_outputs)
    }

    const outputContainer = document.createElement('div')
    outputContainer.className = 'img-batch'
    task.outputContainer.insertBefore(outputContainer, task.outputContainer.firstChild)

    const instance = new SD.RenderTask(newTaskReqBody)
    task['instances'].push(instance)
    task.batchesDone++

    instance.enqueue(getTaskUpdater(task, newTaskReqBody, outputContainer)).then(
        (renderResult) => {
            onTaskCompleted(task, newTaskReqBody, instance, outputContainer, renderResult)
        }
        , (reason) => {
            getTaskErrorHandler(task, newTaskReqBody, instance, reason)
        }
    )

    setStatus('request', 'fetching..')
    stopImageBtn.style.display = 'block'
    renameMakeImageButton()
    previewTools.style.display = 'block'
}

function createTask(task) {
    let taskConfig = `Seed: ${task.seed}, Sampler: ${task.reqBody.sampler}, Inference Steps: ${task.reqBody.num_inference_steps}, Guidance Scale: ${task.reqBody.guidance_scale}, Model: ${task.reqBody.use_stable_diffusion_model}`
    if (task.reqBody.use_vae_model.trim() !== '') {
        taskConfig += `, VAE: ${task.reqBody.use_vae_model}`
    }
    if (task.reqBody.negative_prompt.trim() !== '') {
        taskConfig += `, Negative Prompt: ${task.reqBody.negative_prompt}`
    }
    if (task.reqBody.init_image !== undefined) {
        taskConfig += `, Prompt Strength: ${task.reqBody.prompt_strength}`
    }
    if (task.reqBody.use_face_correction) {
        taskConfig += `, Fix Faces: ${task.reqBody.use_face_correction}`
    }
    if (task.reqBody.use_upscale) {
        taskConfig += `, Upscale: ${task.reqBody.use_upscale}`
    }

    let taskEntry = document.createElement('div')
    taskEntry.id = `imageTaskContainer-${Date.now()}`
    taskEntry.className = 'imageTaskContainer'
    taskEntry.innerHTML = ` <div class="header-content panel collapsible active">
                                <div class="taskStatusLabel">Enqueued</div>
                                <button class="secondaryButton stopTask"><i class="fa-solid fa-trash-can"></i> Remove</button>
                                <div class="preview-prompt collapsible active"></div>
                                <div class="taskConfig">${taskConfig}</div>
                                <div class="outputMsg"></div>
                                <div class="progress-bar active"><div></div></div>
                            </div>
                            <div class="collapsible-content">
                                <div class="img-preview">
                            </div>`

    createCollapsibles(taskEntry)

    task['taskStatusLabel'] = taskEntry.querySelector('.taskStatusLabel')
    task['outputContainer'] = taskEntry.querySelector('.img-preview')
    task['outputMsg'] = taskEntry.querySelector('.outputMsg')
    task['previewPrompt'] = taskEntry.querySelector('.preview-prompt')
    task['progressBar'] = taskEntry.querySelector('.progress-bar')
    task['stopTask'] = taskEntry.querySelector('.stopTask')

    task['stopTask'].addEventListener('click', async function(e) {
        e.stopPropagation()
        if (task.batchesDone <= 0 || !task.isProcessing) {
            taskEntry.remove()
        }
        if (task.isProcessing) {
            task.isProcessing = false
            task.progressBar.classList.remove("active")
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
    })

    task.isProcessing = true
    taskEntry = imagePreview.insertBefore(taskEntry, previewTools.nextSibling)
    taskEntry.draggable = true
    htmlTaskMap.set(taskEntry, task)
    taskEntry.addEventListener('dragstart', function(ev) {
        ev.dataTransfer.setData("text/plain", ev.target.id);
    })

    task.previewPrompt.innerText = task.reqBody.prompt
    if (task.previewPrompt.innerText.trim() === '') {
        task.previewPrompt.innerHTML = '&nbsp;' // allows the results to be collapsed
    }
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
            negative_prompt: negativePromptField.value.trim(),
            num_outputs: numOutputsParallel,
            num_inference_steps: parseInt(numInferenceStepsField.value),
            guidance_scale: parseFloat(guidanceScaleField.value),
            width: parseInt(widthField.value),
            height: parseInt(heightField.value),
            // allow_nsfw: allowNSFWField.checked,
            turbo: turboField.checked,
            render_device: getCurrentRenderDeviceSelection(),
            use_full_precision: useFullPrecisionField.checked,
            use_stable_diffusion_model: stableDiffusionModelField.value,
            use_vae_model: vaeModelField.value,
            stream_progress_updates: true,
            stream_image_progress: (numOutputsTotal > 50 ? false : streamImageProgressField.checked),
            show_only_filtered_image: showOnlyFilteredImageField.checked,
            output_format: outputFormatField.value
        }
    }
    if (IMAGE_REGEX.test(initImagePreview.src)) {
        newTask.reqBody.init_image = initImagePreview.src
        newTask.reqBody.prompt_strength = parseFloat(promptStrengthField.value)

        // if (IMAGE_REGEX.test(maskImagePreview.src)) {
        //     newTask.reqBody.mask = maskImagePreview.src
        // }
        if (maskSetting.checked) {
            newTask.reqBody.mask = inpaintingEditor.getImg()
        }
        newTask.reqBody.sampler = 'ddim'
    } else {
        newTask.reqBody.sampler = samplerField.value
    }
    if (saveToDiskField.checked && diskPathField.value.trim() !== '') {
        newTask.reqBody.save_to_disk_path = diskPathField.value.trim()
    }
    if (useFaceCorrectionField.checked) {
        newTask.reqBody.use_face_correction = 'GFPGANv1.3'
    }
    if (useUpscalingField.checked) {
        newTask.reqBody.use_upscale = upscaleModelField.value
    }
    return newTask
}

function getCurrentRenderDeviceSelection() {
    if (useCPUField.checked) {
        return 'cpu'
    }

    let selectedGPUs = $(useGPUsField).val()
    if (selectedGPUs.length == 0) {
        selectedGPUs = ['auto']
    }
    return selectedGPUs.join(',')
}

function getPrompts() {
    let prompts = promptField.value
    if (prompts.trim() === '') {
        return ['']
    }

    prompts = prompts.split('\n')
    prompts = prompts.map(prompt => prompt.trim())
    prompts = prompts.filter(prompt => prompt !== '')

    let promptsToMake = applySetOperator(prompts)
    promptsToMake = applyPermuteOperator(promptsToMake)

    if (activeTags.length <= 0) {
        return promptsToMake
    }

    const promptTags = activeTags.map(x => x.name).join(", ")
    return promptsToMake.map((prompt) => `${prompt}, ${promptTags}`)
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
    underscoreName = underscoreName.substring(0, 100)
    //const steps = numInferenceStepsField.value
    //const guidance =  guidanceScaleField.value

    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${steps}_Guidance-${guidance}`

    // add the tags
    // let tags = []
    // let tagString = ''
    // document.querySelectorAll(modifyTagsSelector).forEach(function(tag) {
    //     tags.push(tag.innerHTML)
    // })

    // join the tags with a pipe
    // if (activeTags.length > 0) {
    //     tagString = '_Tags-'
    //     tagString += tags.join('|')
    // }

    // // append empty or populated tags
    // fileName += `${tagString}`

    // add the file extension
    fileName += '.' + (outputFormat === 'png' ? 'png' : 'jpeg')

    return fileName
}

async function stopAllTasks() {
    getUncompletedTaskEntries().forEach((taskEntry) => {
        const taskStatusLabel = taskEntry.querySelector('.taskStatusLabel')
        taskStatusLabel.style.display = 'none'

        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            return
        }

        task.isProcessing = false
        task.instances.forEach((instance) => {
            try {
                instance.abort()
            } catch (e) {
                console.error(e)
            }
        })
    })
}

clearAllPreviewsBtn.addEventListener('click', async function() {
    await stopAllTasks()

    let taskEntries = document.querySelectorAll('.imageTaskContainer')
    taskEntries.forEach(task => {
        task.remove()
    })

    previewTools.style.display = 'none'
    initialText.style.display = 'block'
})

stopImageBtn.addEventListener('click', async function() {
    await stopAllTasks()
})

widthField.addEventListener('change', onDimensionChange)
heightField.addEventListener('change', onDimensionChange)

function renameMakeImageButton() {
    let totalImages = Math.max(parseInt(numOutputsTotalField.value), parseInt(numOutputsParallelField.value))
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
numOutputsParallelField.addEventListener('change', renameMakeImageButton)

function onDimensionChange() {
    if (!maskSetting.checked) {
        return
    }
    let widthValue = parseInt(widthField.value)
    let heightValue = parseInt(heightField.value)

    resizeInpaintingEditor(widthValue, heightValue)
}

diskPathField.disabled = !saveToDiskField.checked
saveToDiskField.addEventListener('change', function(e) {
    diskPathField.disabled = !this.checked
})

upscaleModelField.disabled = !useUpscalingField.checked
useUpscalingField.addEventListener('change', function(e) {
    upscaleModelField.disabled = !this.checked
})

makeImageBtn.addEventListener('click', makeImage)

document.onkeydown = function(e) {
    if (e.ctrlKey && e.code === 'Enter') {
        makeImage()
        e.preventDefault()
    }
}

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

useCPUField.addEventListener('click', function() {
    let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
    if (this.checked) {
        gpuSettingEntry.style.display = 'none'
    } else if (useGPUsField.options.length >= MIN_GPUS_TO_SHOW_SELECTION) {
        gpuSettingEntry.style.display = ''
    }
})

async function changeAppConfig(configDelta) {
    // if (!SD.isServerAvailable()) {
    //     // logError('The server is still starting up..')
    //     alert('The server is still starting up..')
    //     e.preventDefault()
    //     return false
    // }

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

useBetaChannelField.addEventListener('click', async function(e) {
    let updateBranch = (this.checked ? 'beta' : 'main')

    await changeAppConfig({
        'update_branch': updateBranch
    })
})

async function getAppConfig() {
    try {
        let res = await fetch('/get/app_config')
        const config = await res.json()

        if (config.update_branch === 'beta') {
            useBetaChannelField.checked = true
            updateBranchLabel.innerText = "(beta)"
        }

        console.log('get config status response', config)
    } catch (e) {
        console.log('get config status error', e)
    }
}

async function getModels() {
    try {
        const sd_model_setting_key = "stable_diffusion_model"
        const vae_model_setting_key = "vae_model"
        const selectedSDModel = SETTINGS[sd_model_setting_key].value
        const selectedVaeModel = SETTINGS[vae_model_setting_key].value

        const models = await SD.getModels()
        const stableDiffusionOptions = models['stable-diffusion']
        const vaeOptions = models['vae']
        vaeOptions.unshift('') // add a None option

        function createModelOptions(modelField, selectedModel) {
            return function(modelName) {
                const modelOption = document.createElement('option')
                modelOption.value = modelName
                modelOption.innerText = modelName !== '' ? modelName : 'None'

                if (modelName === selectedModel) {
                    modelOption.selected = true
                }

                modelField.appendChild(modelOption)
            }
        }

        stableDiffusionOptions.forEach(createModelOptions(stableDiffusionModelField, selectedSDModel))
        vaeOptions.forEach(createModelOptions(vaeModelField, selectedVaeModel))

        // TODO: set default for model here too
        SETTINGS[sd_model_setting_key].default = stableDiffusionOptions[0]
        if (getSetting(sd_model_setting_key) == '' || SETTINGS[sd_model_setting_key].value == '') {
            setSetting(sd_model_setting_key, stableDiffusionOptions[0])
        }
    } catch (e) {
        console.log('get models error', e)
    }
}

function checkRandomSeed() {
    if (randomSeedField.checked) {
        seedField.disabled = true
        seedField.value = "0"
    } else {
        seedField.disabled = false
    }
}
randomSeedField.addEventListener('input', checkRandomSeed)
checkRandomSeed()

function showInitImagePreview() {
    if (initImageSelector.files.length === 0) {
        initImagePreviewContainer.style.display = 'none'
        // inpaintingEditorContainer.style.display = 'none'
        promptStrengthContainer.style.display = 'none'
        // maskSetting.style.display = 'none'
        return
    }

    let reader = new FileReader()
    let file = initImageSelector.files[0]

    reader.addEventListener('load', function(event) {
        // console.log(file.name, reader.result)
        initImagePreview.src = reader.result
        initImagePreviewContainer.style.display = 'block'
        inpaintingEditorContainer.style.display = 'none'
        promptStrengthContainer.style.display = 'table-row'
        samplerSelectionContainer.style.display = 'none'
        // maskSetting.checked = false
    })

    if (file) {
        reader.readAsDataURL(file)
    }
}
initImageSelector.addEventListener('change', showInitImagePreview)
showInitImagePreview()

initImagePreview.addEventListener('load', function() {
    inpaintingEditorCanvasBackground.style.backgroundImage = "url('" + this.src + "')"
    // maskSetting.style.display = 'block'
    // inpaintingEditorContainer.style.display = 'block'
    initImageSizeBox.textContent = initImagePreview.naturalWidth + " x " + initImagePreview.naturalHeight
    initImageSizeBox.style.display = 'block'
})

initImageClearBtn.addEventListener('click', function() {
    initImageSelector.value = null
    // maskImageSelector.value = null

    initImagePreview.src = ''
    // maskImagePreview.src = ''
    maskSetting.checked = false

    initImagePreviewContainer.style.display = 'none'
    // inpaintingEditorContainer.style.display = 'none'
    // maskImagePreviewContainer.style.display = 'none'

    // maskSetting.style.display = 'none'

    promptStrengthContainer.style.display = 'none'
    samplerSelectionContainer.style.display = 'table-row'
    initImageSizeBox.style.display = 'none'
})

maskSetting.addEventListener('click', function() {
    inpaintingEditorContainer.style.display = (this.checked ? 'block' : 'none')
    onDimensionChange()
})

promptsFromFileBtn.addEventListener('click', function() {
    promptsFromFileSelector.click()
})

promptsFromFileSelector.addEventListener('change', function() {
    if (promptsFromFileSelector.files.length === 0) {
        return
    }

    let reader = new FileReader()
    let file = promptsFromFileSelector.files[0]

    reader.addEventListener('load', function() {
        promptField.value = reader.result
    })

    if (file) {
        reader.readAsText(file)
    }
})

async function getDiskPath() {
    try {
        var diskPath = getSetting("diskPath")
        if (diskPath == '' || diskPath == undefined || diskPath == "undefined") {
            let res = await fetch('/get/output_dir')
            if (res.status === 200) {
                res = await res.json()
                res = res.output_dir

                setSetting("diskPath", res)
            }
        }
    } catch (e) {
        console.log('error fetching output dir path', e)
    }
}

async function getDevices() {
    try {
        const res = await SD.getDevices()
        let allDeviceIds = Object.keys(res['all']).filter(d => d !== 'cpu')
        let activeDeviceIds = Object.keys(res['active']).filter(d => d !== 'cpu')

        if (activeDeviceIds.length === 0) {
            useCPUField.checked = true
        }

        if (allDeviceIds.length < MIN_GPUS_TO_SHOW_SELECTION) {
            let gpuSettingEntry = getParameterSettingsEntry('use_gpus')
            gpuSettingEntry.style.display = 'none'

            if (allDeviceIds.length === 0) {
                useCPUField.checked = true
                useCPUField.disabled = true // no compatible GPUs, so make the CPU mandatory
            }
        }

        useGPUsField.innerHTML = ''

        allDeviceIds.forEach(device => {
            let deviceName = res['all'][device]
            let selected = (activeDeviceIds.includes(device) ? 'selected' : '')
            let deviceOption = `<option value="${device}" ${selected}>${deviceName}</option>`
            useGPUsField.insertAdjacentHTML('beforeend', deviceOption)
        })
    } catch (e) {
        console.log('error fetching devices', e)
    }
}


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

var tabElements = [];
document.querySelectorAll(".tab").forEach(tab => {
    var name = tab.id.replace("tab-", "");
    var content = document.getElementById(`tab-content-${name}`)
    tabElements.push({
        name: name,
        tab: tab,
        content: content
    })

    tab.addEventListener("click", event => {
        if (!tab.classList.contains("active")) {
            tabElements.forEach(tabInfo => {
                if (tabInfo.tab.classList.contains("active")) {
                    tabInfo.tab.classList.toggle("active")
                    tabInfo.content.classList.toggle("active")
                }
            })
            tab.classList.toggle("active")
            content.classList.toggle("active")
        }
    })
})

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
