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


document.querySelector('.drawing-board-control-navigation-back').innerHTML = '<i class="fa-solid fa-rotate-left"></i>'
document.querySelector('.drawing-board-control-navigation-forward').innerHTML = '<i class="fa-solid fa-rotate-right"></i>'

let maskResetButton = document.querySelector('.drawing-board-control-navigation-reset')
maskResetButton.innerHTML = 'Clear'
maskResetButton.style.fontWeight = 'normal'
maskResetButton.style.fontSize = '10pt'

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
function setSystemInfo(devices) {
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

    let systemInfo = `
    <table>
        <tr><td><label>Processor:</label></td><td class="value">${cpu}</td></tr>
        <tr><td><label>Compatible Graphics Cards (all):</label></td><td class="value">${allGPUs.join('</br>')}</td></tr>
        <tr><td></td><td>&nbsp;</td></tr>
        <tr><td><label>Used for rendering 🔥:</label></td><td class="value">${activeGPUs.join('</br>')}</td></tr>
    </table>`

    let systemInfoEl = document.querySelector('#system-info')
    systemInfoEl.innerHTML = systemInfo
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
            setSystemInfo(serverState.devices)
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

    if (randomSeedField.checked) {
        seedField.value = task.seed
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

    task['stopTask'].addEventListener('click', async function(e) {
        e.stopPropagation()
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

            taskEntry.remove()
        }
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

    if (activeTags.length > 0) {
	const promptTags = activeTags.map(x => x.name).join(", ")
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
    if (taskQueue.length == 0) {
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
    imageEditor.setImage(initImagePreview.src, initImagePreview.naturalWidth, initImagePreview.naturalHeight)
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
function selectTab(tab_id) {
    let tabInfo = tabElements.find(t => t.tab.id == tab_id);
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
    var name = tab.id.replace("tab-", "");
    var content = document.getElementById(`tab-content-${name}`)
    tabElements.push({
        name: name,
        tab: tab,
        content: content
    })

    tab.addEventListener("click", event => selectTab(tab.id));
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

// stuff for testing image editor
selectTab("tab-editor");
initImagePreview.src = "data:image/jpeg;base64,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G9M5PqatxaYtvKZEljII4XnP14PNQz6skFyq5UsOGUZIH5mlvsUZd7EIpAiTZkVdxC8jPfFUHZnKq2M44xWmVSeZ5o3yc5UEcH2qoLYxuWkSRVAyCo6Z+taIpH//2f/hDqFodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDQuNC4wLUV4aXYyIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6aXB0Y0V4dD0iaHR0cDovL2lwdGMub3JnL3N0ZC9JcHRjNHhtcEV4dC8yMDA4LTAyLTI5LyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0RXZ0PSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VFdmVudCMiIHhtbG5zOnBsdXM9Imh0dHA6Ly9ucy51c2VwbHVzLm9yZy9sZGYveG1wLzEuMC8iIHhtbG5zOkdJTVA9Imh0dHA6Ly93d3cuZ2ltcC5vcmcveG1wLyIgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOkRvY3VtZW50SUQ9ImdpbXA6ZG9jaWQ6Z2ltcDo0NDgxOGJkMy04OTg5LTRlMmMtYjA1YS1iZTJkODFmMWI0MmEiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6NTExOGRkZWMtZGY5ZS00MWQ5LTgzMTMtZWUzNzllOTllNzcwIiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6NmFlNDQ3MjYtZjYwMC00ZjA1LWEyYjEtZDgwNWU1OTNmMWZkIiBHSU1QOkFQST0iMi4wIiBHSU1QOlBsYXRmb3JtPSJXaW5kb3dzIiBHSU1QOlRpbWVTdGFtcD0iMTY2NTc2NjQxNTA1MjU3NiIgR0lNUDpWZXJzaW9uPSIyLjEwLjIyIiBkYzpGb3JtYXQ9ImltYWdlL2pwZWciIHhtcDpDcmVhdG9yVG9vbD0iR0lNUCAyLjEwIj4gPGlwdGNFeHQ6TG9jYXRpb25DcmVhdGVkPiA8cmRmOkJhZy8+IDwvaXB0Y0V4dDpMb2NhdGlvbkNyZWF0ZWQ+IDxpcHRjRXh0OkxvY2F0aW9uU2hvd24+IDxyZGY6QmFnLz4gPC9pcHRjRXh0OkxvY2F0aW9uU2hvd24+IDxpcHRjRXh0OkFydHdvcmtPck9iamVjdD4gPHJkZjpCYWcvPiA8L2lwdGNFeHQ6QXJ0d29ya09yT2JqZWN0PiA8aXB0Y0V4dDpSZWdpc3RyeUlkPiA8cmRmOkJhZy8+IDwvaXB0Y0V4dDpSZWdpc3RyeUlkPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6Y2hhbmdlZD0iLyIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDoxYTc0MDdmMS1jYWI1LTQwZWItODc3Zi1jNmQ0ODRlM2E0MDQiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkdpbXAgMi4xMCAoV2luZG93cykiIHN0RXZ0OndoZW49IjIwMjItMTAtMTRUMDk6NTM6MzUiLz4gPC9yZGY6U2VxPiA8L3htcE1NOkhpc3Rvcnk+IDxwbHVzOkltYWdlU3VwcGxpZXI+IDxyZGY6U2VxLz4gPC9wbHVzOkltYWdlU3VwcGxpZXI+IDxwbHVzOkltYWdlQ3JlYXRvcj4gPHJkZjpTZXEvPiA8L3BsdXM6SW1hZ2VDcmVhdG9yPiA8cGx1czpDb3B5cmlnaHRPd25lcj4gPHJkZjpTZXEvPiA8L3BsdXM6Q29weXJpZ2h0T3duZXI+IDxwbHVzOkxpY2Vuc29yPiA8cmRmOlNlcS8+IDwvcGx1czpMaWNlbnNvcj4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPD94cGFja2V0IGVuZD0idyI/Pv/iArBJQ0NfUFJPRklMRQABAQAAAqBsY21zBDAAAG1udHJSR0IgWFlaIAfmAAoADgAQADEANWFjc3BNU0ZUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtbGNtcwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADWRlc2MAAAEgAAAAQGNwcnQAAAFgAAAANnd0cHQAAAGYAAAAFGNoYWQAAAGsAAAALHJYWVoAAAHYAAAAFGJYWVoAAAHsAAAAFGdYWVoAAAIAAAAAFHJUUkMAAAIUAAAAIGdUUkMAAAIUAAAAIGJUUkMAAAIUAAAAIGNocm0AAAI0AAAAJGRtbmQAAAJYAAAAJGRtZGQAAAJ8AAAAJG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAJAAAABwARwBJAE0AUAAgAGIAdQBpAGwAdAAtAGkAbgAgAHMAUgBHAEJtbHVjAAAAAAAAAAEAAAAMZW5VUwAAABoAAAAcAFAAdQBiAGwAaQBjACAARABvAG0AYQBpAG4AAFhZWiAAAAAAAAD21gABAAAAANMtc2YzMgAAAAAAAQxCAAAF3v//8yUAAAeTAAD9kP//+6H///2iAAAD3AAAwG5YWVogAAAAAAAAb6AAADj1AAADkFhZWiAAAAAAAAAknwAAD4QAALbEWFlaIAAAAAAAAGKXAAC3hwAAGNlwYXJhAAAAAAADAAAAAmZmAADypwAADVkAABPQAAAKW2Nocm0AAAAAAAMAAAAAo9cAAFR8AABMzQAAmZoAACZnAAAPXG1sdWMAAAAAAAAAAQAAAAxlblVTAAAACAAAABwARwBJAE0AUG1sdWMA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";