"use strict" // Opt in to a restricted variant of JavaScript
const HEALTH_PING_INTERVAL = 5 // seconds
const MAX_INIT_IMAGE_DIMENSION = 768
const MIN_GPUS_TO_SHOW_SELECTION = 2

const IMAGE_REGEX = new RegExp('data:image/[A-Za-z]+;base64')

let sessionId = Date.now()

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

let maskSetting = document.querySelector('#enable_mask')

let imagePreview = document.querySelector("#preview")

let serverStatusColor = document.querySelector('#server-status-color')
let serverStatusMsg = document.querySelector('#server-status-msg')


let serverState = {'status': 'Offline', 'time': Date.now()}
let bellPending = false

let taskQueue = []
let currentTask = null

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
function isServerAvailable() {
    if (typeof serverState !== 'object') {
        return false
    }
    switch (serverState.status) {
        case 'LoadingModel':
        case 'Rendering':
        case 'Online':
            return true
        default:
            return false
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

async function healthCheck() {
    try {
        let res = undefined
        if (sessionId) {
            res = await fetch('/ping?session_id=' + sessionId)
        } else {
            res = await fetch('/ping')
        }
        serverState = await res.json()
        if (typeof serverState !== 'object' || typeof serverState.status !== 'string') {
            serverState = {'status': 'Offline', 'time': Date.now()}
            setServerStatus('error', 'offline')
            return
        }
        // Set status
        switch(serverState.status) {
            case 'Init':
                // Wait for init to complete before updating status.
                break
            case 'Online':
                setServerStatus('online', 'ready')
                break
            case 'LoadingModel':
                setServerStatus('busy', 'loading..')
                break
            case 'Rendering':
                setServerStatus('busy', 'rendering..')
                break
            default: // Unavailable
                setServerStatus('error', serverState.status.toLowerCase())
                break
        }
        if (serverState.devices) {
            setDeviceInfo(serverState.devices)
        }
        serverState.time = Date.now()
    } catch (e) {
        console.log(e)
        serverState = {'status': 'Offline', 'time': Date.now()}
        setServerStatus('error', 'offline')
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

    maskSetting.checked = false
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
        use_face_correction: 'GFPGANv1.3'
    })
}

function onContinueDrawingClick(req, img) {
    enqueueImageVariationTask(req, img, {
        num_inference_steps: parseInt(req.num_inference_steps) + 25
    })
}

// makes a single image. don't call this directly, use makeImage() instead
async function doMakeImage(task) {
    if (task.stopped) {
        return
    }

    const RETRY_DELAY_IF_BUFFER_IS_EMPTY = 1000 // ms
    const RETRY_DELAY_IF_SERVER_IS_BUSY = 30 * 1000 // ms, status_code 503, already a task running
    const TASK_START_DELAY_ON_SERVER = 1500 // ms
    const SERVER_STATE_VALIDITY_DURATION = 90 * 1000 // ms

    const reqBody = task.reqBody
    const batchCount = task.batchCount
    const outputContainer = document.createElement('div')

    outputContainer.className = 'img-batch'
    task.outputContainer.insertBefore(outputContainer, task.outputContainer.firstChild)

    const outputMsg = task['outputMsg']
    const previewPrompt = task['previewPrompt']
    const progressBar = task['progressBar']
    const progressBarInner = progressBar.querySelector("div")

    let res = undefined
    try {
        let renderRequest = undefined
        do {
            res = await fetch('/render', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reqBody)
            })
            renderRequest = await res.json()
            // status_code 503, already a task running.
        } while (res.status === 503 && await asyncDelay(RETRY_DELAY_IF_SERVER_IS_BUSY))

        if (typeof renderRequest?.stream !== 'string') {
            console.log('Endpoint response: ', renderRequest)
            throw new Error(renderRequest?.detail || 'Endpoint response does not contains a response stream url.')
        }

        task['taskStatusLabel'].innerText = "Waiting"
        task['taskStatusLabel'].classList.add('waitingTaskLabel')
        task['taskStatusLabel'].classList.remove('activeTaskLabel')

        do { // Wait for server status to update.
            await asyncDelay(250)
            if (!isServerAvailable()) {
                throw new Error('Connexion with server lost.')
            }
        } while (Date.now() < (serverState.time + SERVER_STATE_VALIDITY_DURATION) && serverState.task !== renderRequest.task)

        switch(serverState.session) {
            case 'pending':
            case 'running':
            case 'buffer':
                // Normal expected messages.
                break
            case 'completed':
                console.warn('Server %o render request %o completed unexpectedly', serverState, renderRequest)
                break // Continue anyway to try to read cached result.
            case 'error':
                console.error('Server %o render request %o has failed', serverState, renderRequest)
                break // Still valid, Update UI with error message
            case 'stopped':
                console.log('Server %o render request %o was stopped', serverState, renderRequest)
                return false
            default:
                throw new Error('Unexpected server task state: ' + serverState.session || 'Undefined')
        }

        while (serverState.task === renderRequest.task && serverState.session === 'pending') {
            // Wait for task to start on server.
            await asyncDelay(TASK_START_DELAY_ON_SERVER)
        }

        // Task started!
        res = await fetch(renderRequest.stream, {
            headers: {
                'Content-Type': 'application/json'
            },
        })

        task['taskStatusLabel'].innerText = "Processing"
        task['taskStatusLabel'].classList.add('activeTaskLabel')
        task['taskStatusLabel'].classList.remove('waitingTaskLabel')

        let stepUpdate = undefined
        let reader = res.body.getReader()
        let textDecoder = new TextDecoder()
        let finalJSON = ''
        let readComplete = false
        while (!readComplete || finalJSON.length > 0) {
            let t = Date.now()
            let jsonStr = ''
            if (!readComplete) {
                const {value, done} = await reader.read()
                if (done) {
                    readComplete = true
                }
                if (value) {
                    jsonStr = textDecoder.decode(value)
                }
            }
            stepUpdate = undefined
            try {
                // hack for a middleman buffering all the streaming updates, and unleashing them on the poor browser in one shot.
                // this results in having to parse JSON like {"step": 1}{"step": 2}{"step": 3}{"ste...
                // which is obviously invalid and can happen at any point while rendering.
                // So we need to extract only the next {} section
                if (finalJSON.length > 0) {
                    // Append new data when required
                    if (jsonStr.length > 0) {
                        jsonStr = finalJSON + jsonStr
                    } else {
                        jsonStr = finalJSON
                    }
                    finalJSON = ''
                }
                // Find next delimiter
                let lastChunkIdx = jsonStr.indexOf('}{')
                if (lastChunkIdx !== -1) {
                    finalJSON = jsonStr.substring(0, lastChunkIdx + 1)
                    jsonStr = jsonStr.substring(lastChunkIdx + 1)
                } else {
                    finalJSON = jsonStr
                    jsonStr = ''
                }
                // Try to parse
                stepUpdate = (finalJSON.length > 0 ? JSON.parse(finalJSON) : undefined)
                finalJSON = jsonStr
            } catch (e) {
                if (e instanceof SyntaxError && !readComplete) {
                    finalJSON += jsonStr
                } else {
                    throw e
                }
            }
            if (typeof stepUpdate === 'object' && 'step' in stepUpdate) {
                let batchSize = stepUpdate.total_steps
                let overallStepCount = stepUpdate.step + task.batchesDone * batchSize
                let totalSteps = batchCount * batchSize
                let percent = 100 * (overallStepCount / totalSteps)
                percent = (percent > 100 ? 100 : percent)
                percent = percent.toFixed(0)
                let timeTaken = stepUpdate.step_time // sec

                let stepsRemaining = totalSteps - overallStepCount
                stepsRemaining = (stepsRemaining < 0 ? 0 : stepsRemaining)
                let timeRemaining = (timeTaken === -1 ? '' : stepsRemaining * timeTaken * 1000) // ms

                outputMsg.innerHTML = `Batch ${task.batchesDone+1} of ${batchCount}`
                outputMsg.innerHTML += `. Generating image(s): ${percent}%`

                timeRemaining = (timeTaken !== -1 ? millisecondsToStr(timeRemaining) : '')
                outputMsg.innerHTML += `. Time remaining (approx): ${timeRemaining}`
                outputMsg.style.display = 'block'

                progressBarInner.style.width = `${percent}%`
                if (percent == 100) {
                    task.progressBar.style.height = "0px"
                    task.progressBar.style.border = "0px solid var(--background-color3)"
                    task.progressBar.classList.remove("active")
                }

                if (stepUpdate.output !== undefined) {
                    showImages(reqBody, stepUpdate, outputContainer, true)
                }
            }
            if (stepUpdate?.status) {
                break
            }
            if (readComplete && finalJSON.length <= 0) {
                if (res.status === 200) {
                    await asyncDelay(RETRY_DELAY_IF_BUFFER_IS_EMPTY)
                    res = await fetch(renderRequest.stream, {
                        headers: {
                            'Content-Type': 'application/json'
                        },
                    })
                    reader = res.body.getReader()
                    readComplete = false
                } else {
                    console.log('Stream stopped: ', res)
                }
            }
        }

        if (typeof stepUpdate === 'object' && stepUpdate.status !== 'succeeded') {
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
            logError(msg, res, outputMsg)
            return false
        }
        if (typeof stepUpdate !== 'object' || !res || res.status != 200) {
            if (!isServerAvailable()) {
                logError("Stable Diffusion is still starting up, please wait. If this goes on beyond a few minutes, Stable Diffusion has probably crashed. Please check the error message in the command-line window.", res, outputMsg)
            } else if (typeof res === 'object') {
                let msg = 'Stable Diffusion had an error reading the response: '
                try { // 'Response': body stream already read
                    msg += 'Read: ' + await res.text()
                } catch(e) {
                    msg += 'Unexpected end of stream. '
                }
                if (finalJSON) {
                    msg += 'Buffered data: ' + finalJSON
                }
                logError(msg, res, outputMsg)
            } else {
                let msg = `Unexpected Read Error:<br/><pre>Response: ${res}<br/>StepUpdate: ${typeof stepUpdate === 'object' ? JSON.stringify(stepUpdate, undefined, 4) : stepUpdate}</pre>`
                logError(msg, res, outputMsg)
            }
            return false
        }

        showImages(reqBody, stepUpdate, outputContainer, false)
    } catch (e) {
        console.log('request error', e)
        logError('Stable Diffusion had an error. Please check the logs in the command-line window. <br/><br/>' + e + '<br/><pre>' + e.stack + '</pre>', res, outputMsg)
        setStatus('request', 'error', 'error')
        return false
    }
    return true
}

async function checkTasks() {
    if (taskQueue.length === 0) {
        setStatus('request', 'done', 'success')
        setTimeout(checkTasks, 500)
        stopImageBtn.style.display = 'none'
        renameMakeImageButton()

        currentTask = null

        if (bellPending) {
            if (isSoundEnabled()) {
                playSound()
            }
            bellPending = false
        }

        return
    }

    setStatus('request', 'fetching..')

    stopImageBtn.style.display = 'block'
    renameMakeImageButton()
    bellPending = true

    previewTools.style.display = 'block'

    let task = taskQueue.pop()
    currentTask = task

    let time = Date.now()

    let successCount = 0

    task.isProcessing = true
    task['stopTask'].innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop'
    task['taskStatusLabel'].innerText = "Starting"
    task['taskStatusLabel'].classList.add('waitingTaskLabel')

    const genSeeds = Boolean(typeof task.reqBody.seed !== 'number' || (task.reqBody.seed === task.seed && task.numOutputsTotal > 1))
    const startSeed = task.reqBody.seed || task.seed
    
    // Update the seed *before* starting the processing so it's retained if user stops the task
    if (randomSeedField.checked) {
        seedField.value = task.seed
    }
    
    for (let i = 0; i < task.batchCount; i++) {
        let newTask = task
        if (task.batchCount > 1) {
            // Each output render batch needs it's own task instance to avoid altering the other runs after they are completed.
            newTask = Object.assign({}, task, {
                reqBody: Object.assign({}, task.reqBody)
            })
        }
        if (genSeeds) {
            newTask.reqBody.seed = parseInt(startSeed) + (i * newTask.reqBody.num_outputs)
            newTask.seed = newTask.reqBody.seed
        } else if (newTask.seed !== newTask.reqBody.seed) {
            newTask.seed = newTask.reqBody.seed
        }

        let success = await doMakeImage(newTask)
        task.batchesDone++

        if (!task.isProcessing || !success) {
            break
        }

        if (success) {
            successCount++
        }
    }

    task.isProcessing = false
    task['stopTask'].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    task['taskStatusLabel'].style.display = 'none'

    time = Date.now() - time
    time /= 1000

    if (successCount === task.batchCount) {
        task.outputMsg.innerText = 'Processed ' + task.numOutputsTotal + ' images in ' + time + ' seconds'
        task.progressBar.style.height = "0px"
        task.progressBar.style.border = "0px solid var(--background-color3)"
        task.progressBar.classList.remove("active")
        // setStatus('request', 'done', 'success')
    } else {
        if (task.outputMsg.innerText.toLowerCase().indexOf('error') === -1) {
            task.outputMsg.innerText = 'Task ended after ' + time + ' seconds'
        }
    }

    currentTask = null

    if (typeof requestIdleCallback === 'function') {
        requestIdleCallback(checkTasks, { timeout: 30 * 1000 })
    } else {
        setTimeout(checkTasks, 500)
    }
}
if (typeof requestIdleCallback === 'function') {
    requestIdleCallback(checkTasks, { timeout: 30 * 1000 })
} else {
    setTimeout(checkTasks, 10)
}

function getCurrentUserRequest() {
    const numOutputsTotal = parseInt(numOutputsTotalField.value)
    const numOutputsParallel = parseInt(numOutputsParallelField.value)
    const seed = (randomSeedField.checked ? Math.floor(Math.random() * 10000000) : parseInt(seedField.value))

    const newTask = {
        isProcessing: false,
        stopped: false,
        batchesDone: 0,
        numOutputsTotal: numOutputsTotal,
        batchCount: Math.ceil(numOutputsTotal / numOutputsParallel),
        seed,

        reqBody: {
            session_id: sessionId,
            seed,
            negative_prompt: negativePromptField.value.trim(),
            num_outputs: numOutputsParallel,
            num_inference_steps: numInferenceStepsField.value,
            guidance_scale: guidanceScaleField.value,
            width: widthField.value,
            height: heightField.value,
            // allow_nsfw: allowNSFWField.checked,
            turbo: turboField.checked,
            use_full_precision: useFullPrecisionField.checked,
            use_stable_diffusion_model: stableDiffusionModelField.value,
            use_vae_model: vaeModelField.value,
            stream_progress_updates: true,
            stream_image_progress: (numOutputsTotal > 50 ? false : streamImageProgressField.checked),
            show_only_filtered_image: showOnlyFilteredImageField.checked,
            output_format: outputFormatField.value,
            original_prompt: promptField.value,
            active_tags: (activeTags.map(x => x.name))
        }
    }
    if (IMAGE_REGEX.test(initImagePreview.src)) {
        newTask.reqBody.init_image = initImagePreview.src
        newTask.reqBody.prompt_strength = promptStrengthField.value

        // if (IMAGE_REGEX.test(maskImagePreview.src)) {
        //     newTask.reqBody.mask = maskImagePreview.src
        // }
        if (maskSetting.checked) {
            newTask.reqBody.mask = imageInpainter.getImg()
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

function makeImage() {
    if (!isServerAvailable()) {
        alert('The server is not available.')
    } else if (!randomSeedField.checked && seedField.value == '') {
        alert('The "Seed" field must not be empty.')
    } else if (numOutputsTotalField.value == '') {
        alert('The "Number of Images" field must not be empty.')
    } else if (numOutputsParallelField.value == '') {
        alert('The "Number of parallel Images" field must not be empty.')
    } else if (numInferenceStepsField.value == '') {
        alert('The "Inference Steps" field must not be empty.')
    } else if (guidanceScaleField.value == '') {
        alert('The Guidance Scale field must not be empty.')
    } else {
        const taskTemplate = getCurrentUserRequest()
        const newTaskRequests = []
        getPrompts().forEach((prompt) => newTaskRequests.push(Object.assign({}, taskTemplate, {
            reqBody: Object.assign({ prompt: prompt }, taskTemplate.reqBody)
        })))
        newTaskRequests.forEach(createTask)

        initialText.style.display = 'none'
    }
}

function createTask(task) {
    let taskConfig = `<b>Seed:</b> ${task.seed}, <b>Sampler:</b> ${task.reqBody.sampler}, <b>Inference Steps:</b> ${task.reqBody.num_inference_steps}, <b>Guidance Scale:</b> ${task.reqBody.guidance_scale}, <b>Model:</b> ${task.reqBody.use_stable_diffusion_model}`
    if (task.reqBody.use_vae_model.trim() !== '') {
        taskConfig += `, <b>VAE:</b> ${task.reqBody.use_vae_model}`
    }
    if (task.reqBody.negative_prompt.trim() !== '') {
        taskConfig += `, <b>Negative Prompt:</b> ${task.reqBody.negative_prompt}`
    }
    if (task.reqBody.init_image !== undefined) {
        taskConfig += `, <b>Prompt Strength:</b> ${task.reqBody.prompt_strength}`
    }
    if (task.reqBody.use_face_correction) {
        taskConfig += `, <b>Fix Faces:</b> ${task.reqBody.use_face_correction}`
    }
    if (task.reqBody.use_upscale) {
        taskConfig += `, <b>Upscale:</b> ${task.reqBody.use_upscale}`
    }

    let taskEntry = document.createElement('div')
    taskEntry.className = 'imageTaskContainer'
    taskEntry.innerHTML = ` <div class="header-content panel collapsible active">
                                <div class="taskStatusLabel">Enqueued</div>
                                <button class="secondaryButton stopTask"><i class="fa-solid fa-trash-can"></i> Remove</button>
                                <button class="secondaryButton useSettings"><i class="fa-solid fa-redo"></i> Use these settings</button>
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

    task['stopTask'].addEventListener('click', (e) => {
        let question = (task['isProcessing'] ? "Stop this task?" : "Remove this task?")
        shiftOrConfirm(e, question, async function(e) {
            if (task['isProcessing']) {
                task.isProcessing = false
                task.progressBar.classList.remove("active")
                try {
                    let res = await fetch('/image/stop?session_id=' + sessionId)
                } catch (e) {
                    console.log(e)
                }
            } else {
                let idx = taskQueue.indexOf(task)
                if (idx >= 0) {
                    taskQueue.splice(idx, 1)
                }

                removeTask(taskEntry)
            }
        })
    })

    task['useSettings'] = taskEntry.querySelector('.useSettings')
    task['useSettings'].addEventListener('click', function(e) {
        e.stopPropagation()
        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
    })

    imagePreview.insertBefore(taskEntry, previewTools.nextSibling)

    task.previewPrompt.innerText = task.reqBody.prompt
    if (task.previewPrompt.innerText.trim() === '') {
        task.previewPrompt.innerHTML = '&nbsp;' // allows the results to be collapsed
    }

    taskQueue.unshift(task)
}

function getPrompts() {
    let prompts = promptField.value
    if (prompts.trim() === '') {
        return ['']
    }

    prompts = prompts.split('\n')
    prompts = prompts.map(prompt => prompt.trim())
    prompts = prompts.filter(prompt => prompt !== '')

    const newTags = activeTags.filter(tag => tag.inactive === undefined || tag.inactive === false)
    if (newTags.length > 0) {
        const promptTags = newTags.map(x => x.name).join(", ")
        prompts = prompts.map((prompt) => `${prompt}, ${promptTags}`)
    }

    let promptsToMake = applySetOperator(prompts)
    promptsToMake = applyPermuteOperator(promptsToMake)

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
    taskQueue.forEach(task => {
        task.isProcessing = false
    })
    taskQueue = []

    if (currentTask !== null) {
        currentTask.isProcessing = false
    }

    try {
        let res = await fetch('/image/stop?session_id=' + sessionId)
    } catch (e) {
        console.log(e)
    }
}

function removeTask(taskToRemove) {
    taskToRemove.remove()

    if (document.querySelector('.imageTaskContainer') === null) {
        previewTools.style.display = 'none'
        initialText.style.display = 'block'
    }
}

clearAllPreviewsBtn.addEventListener('click', (e) => { shiftOrConfirm(e, "Clear all the results and tasks in this window?", async function() {
    await stopAllTasks()

    let taskEntries = document.querySelectorAll('.imageTaskContainer')
    taskEntries.forEach(removeTask)
})})

stopImageBtn.addEventListener('click', (e) => { shiftOrConfirm(e, "Stop all the tasks?", async function(e) {
    await stopAllTasks()
})})

widthField.addEventListener('change', onDimensionChange)
heightField.addEventListener('change', onDimensionChange)

function renameMakeImageButton() {
    let totalImages = Math.max(parseInt(numOutputsTotalField.value), parseInt(numOutputsParallelField.value))
    let imageLabel = 'Image'
    if (totalImages > 1) {
        imageLabel = totalImages + ' Images'
    }
    if (taskQueue.length == 0) {
        makeImageBtn.innerText = 'Make ' + imageLabel
    } else {
        makeImageBtn.innerText = 'Enqueue Next ' + imageLabel
    }
}
numOutputsTotalField.addEventListener('change', renameMakeImageButton)
numOutputsParallelField.addEventListener('change', renameMakeImageButton)

function onDimensionChange() {
    let widthValue = parseInt(widthField.value)
    let heightValue = parseInt(heightField.value)
    if (!initImagePreviewContainer.classList.contains("has-image")) {
        imageEditor.setImage(null, widthValue, heightValue)
    }
    else {
        imageInpainter.setImage(initImagePreview.src, widthValue, heightValue)
    }
}

diskPathField.disabled = !saveToDiskField.checked

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

async function getModels() {
    try {
        var sd_model_setting_key = "stable_diffusion_model"
        var vae_model_setting_key = "vae_model"
        var selectedSDModel = SETTINGS[sd_model_setting_key].value
        var selectedVaeModel = SETTINGS[vae_model_setting_key].value
        let res = await fetch('/get/models')
        const models = await res.json()

        console.log('got models response', models)

        if ( "scan-error" in models ) {
            // let previewPane = document.getElementById('tab-content-wrapper')
            let previewPane = document.getElementById('preview')
            previewPane.style.background="red"
            previewPane.style.textAlign="center"
            previewPane.innerHTML = '<H1>🔥Malware alert!🔥</H1><h2>The file <i>' + models['scan-error'] + '</i> in your <tt>models/stable-diffusion</tt> folder is probably malware infected.</h2><h2>Please delete this file from the folder before proceeding!</h2>After deleting the file, reload this page.<br><br><button onClick="window.location.reload();">Reload Page</button>'
            makeImageBtn.disabled = true
        }
        let modelOptions = models['options']
        let stableDiffusionOptions = modelOptions['stable-diffusion']
        let vaeOptions = modelOptions['vae']
        vaeOptions.unshift('') // add a None option

        function createModelOptions(modelField, selectedModel) {
            return function(modelName) {
                let modelOption = document.createElement('option')
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
        //seedField.value = "0" // This causes the seed to be lost if the user changes their mind after toggling the checkbox
    } else {
        seedField.disabled = false
    }
}
randomSeedField.addEventListener('input', checkRandomSeed)
checkRandomSeed()

function showInitImagePreview() {
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
initImageSelector.addEventListener('change', showInitImagePreview)
showInitImagePreview()

initImagePreview.addEventListener('load', function() {
    promptStrengthContainer.style.display = 'table-row'
    initImagePreviewContainer.classList.add("has-image")

    initImageSizeBox.textContent = initImagePreview.naturalWidth + " x " + initImagePreview.naturalHeight
    imageEditor.setImage(this.src, initImagePreview.naturalWidth, initImagePreview.naturalHeight)
    imageInpainter.setImage(this.src, parseInt(widthField.value), parseInt(heightField.value))
})

initImageClearBtn.addEventListener('click', function() {
    initImageSelector.value = null
    initImagePreview.src = ''
    maskSetting.checked = false

    promptStrengthContainer.style.display = 'none'
    initImagePreviewContainer.classList.remove("has-image")
    imageEditor.setImage(null, parseInt(widthField.value), parseInt(heightField.value))
})

maskSetting.addEventListener('click', function() {
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
            if (info.tab.classList.contains("active")) {
                info.tab.classList.toggle("active")
                info.content.classList.toggle("active")
            }
        })
        tabInfo.tab.classList.toggle("active")
        tabInfo.content.classList.toggle("active")
    }
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
