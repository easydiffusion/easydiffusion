const SOUND_ENABLED_KEY = "soundEnabled"
const SAVE_TO_DISK_KEY = "saveToDisk"
const USE_CPU_KEY = "useCPU"
const USE_FULL_PRECISION_KEY = "useFullPrecision"
const USE_TURBO_MODE_KEY = "useTurboMode"
const DISK_PATH_KEY = "diskPath"
const ADVANCED_PANEL_OPEN_KEY = "advancedPanelOpen"
const MODIFIERS_PANEL_OPEN_KEY = "modifiersPanelOpen"
const USE_FACE_CORRECTION_KEY = "useFaceCorrection"
const USE_UPSCALING_KEY = "useUpscaling"
const SHOW_ONLY_FILTERED_IMAGE_KEY = "showOnlyFilteredImage"
const STREAM_IMAGE_PROGRESS_KEY = "streamImageProgress"
const HEALTH_PING_INTERVAL = 5 // seconds
const MAX_INIT_IMAGE_DIMENSION = 768

const IMAGE_REGEX = new RegExp('data:image/[A-Za-z]+;base64')

let sessionId = new Date().getTime()

let promptField = document.querySelector('#prompt')
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
let maskImageSelector = document.querySelector("#mask")
let maskImagePreview = document.querySelector("#mask_preview")
let turboField = document.querySelector('#turbo')
let useCPUField = document.querySelector('#use_cpu')
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

let editorModifierEntries = document.querySelector('#editor-modifiers-entries')
let editorModifierTagsList = document.querySelector('#editor-inputs-tags-list')
let editorTagsContainer = document.querySelector('#editor-inputs-tags-container')

let imagePreview = document.querySelector("#preview")
let previewImageField = document.querySelector('#preview-image')
previewImageField.onchange = () => changePreviewImages(previewImageField.value);

let modifierCardSizeSlider = document.querySelector('#modifier-card-size-slider')
modifierCardSizeSlider.onchange = () => resizeModifierCards(modifierCardSizeSlider.value);

// let previewPrompt = document.querySelector('#preview-prompt')

let showConfigToggle = document.querySelector('#configToggleBtn')
// let configBox = document.querySelector('#config')
// let outputMsg = document.querySelector('#outputMsg')
// let progressBar = document.querySelector("#progressBar")

let soundToggle = document.querySelector('#sound_toggle')

let serverStatusColor = document.querySelector('#server-status-color')
let serverStatusMsg = document.querySelector('#server-status-msg')

let advancedPanelHandle = document.querySelector("#editor-settings .collapsible")
let modifiersPanelHandle = document.querySelector("#editor-modifiers .collapsible")
let inpaintingEditorContainer = document.querySelector('#inpaintingEditor')
let inpaintingEditor = new DrawingBoard.Board('inpaintingEditor', {
    color: "#ffffff",
    background: false,
    size: 30,
    webStorage: false,
    controls: [{'DrawingMode': {'filler': false}}, 'Size', 'Navigation']
})
let inpaintingEditorCanvasBackground = document.querySelector('.drawing-board-canvas-wrapper')

document.querySelector('.drawing-board-control-navigation-back').innerHTML = '<i class="fa-solid fa-rotate-left"></i>'
document.querySelector('.drawing-board-control-navigation-forward').innerHTML = '<i class="fa-solid fa-rotate-right"></i>'

let maskResetButton = document.querySelector('.drawing-board-control-navigation-reset')
maskResetButton.innerHTML = 'Clear'
maskResetButton.style.fontWeight = 'normal'
maskResetButton.style.fontSize = '10pt'

let serverStatus = 'offline'
let activeTags = []
let modifiers = []
let lastPromptUsed = ''
let bellPending = false

let taskQueue = []
let currentTask = null

const modifierThumbnailPath = 'media/modifier-thumbnails';
const activeCardClass = 'modifier-card-active';

function getLocalStorageItem(key, fallback) {
    let item = localStorage.getItem(key)
    if (item === null) {
        return fallback
    }

    return item
}

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
    return getLocalStorageBoolItem(SOUND_ENABLED_KEY, true)
}

function isFaceCorrectionEnabled() {
    return getLocalStorageBoolItem(USE_FACE_CORRECTION_KEY, false)
}

function isUpscalingEnabled() {
    return getLocalStorageBoolItem(USE_UPSCALING_KEY, false)
}

function isShowOnlyFilteredImageEnabled() {
    return getLocalStorageBoolItem(SHOW_ONLY_FILTERED_IMAGE_KEY, true)
}

function isSaveToDiskEnabled() {
    return getLocalStorageBoolItem(SAVE_TO_DISK_KEY, false)
}

function isUseCPUEnabled() {
    return getLocalStorageBoolItem(USE_CPU_KEY, false)
}

function isUseFullPrecisionEnabled() {
    return getLocalStorageBoolItem(USE_FULL_PRECISION_KEY, false)
}

function isUseTurboModeEnabled() {
    return getLocalStorageBoolItem(USE_TURBO_MODE_KEY, true)
}

function getSavedDiskPath() {
    return getLocalStorageItem(DISK_PATH_KEY, '')
}

function isAdvancedPanelOpenEnabled() {
    return getLocalStorageBoolItem(ADVANCED_PANEL_OPEN_KEY, false)
}

function isModifiersPanelOpenEnabled() {
    return getLocalStorageBoolItem(MODIFIERS_PANEL_OPEN_KEY, false)
}

function isStreamImageProgressEnabled() {
    return getLocalStorageBoolItem(STREAM_IMAGE_PROGRESS_KEY, false)
}

function setStatus(statusType, msg, msgType) {
    if (statusType !== 'server') {
        return;
    }

    if (msgType == 'error') {
        // msg = '<span style="color: red">' + msg + '<span>'
        serverStatusColor.style.color = 'red'
        serverStatusMsg.style.color = 'red'
        serverStatusMsg.innerText = 'Stable Diffusion has stopped'
    } else if (msgType == 'success') {
        // msg = '<span style="color: green">' + msg + '<span>'
        serverStatusColor.style.color = 'green'
        serverStatusMsg.style.color = 'green'
        serverStatusMsg.innerText = 'Stable Diffusion is ready'
        serverStatus = 'online'
    }
}

function logMsg(msg, level, outputMsg) {
    if (level === 'error') {
        outputMsg.innerHTML = '<span style="color: red">Error: ' + msg + '</span>'
    } else if (level === 'warn') {
        outputMsg.innerHTML = '<span style="color: orange">Warning: ' + msg + '</span>'
    } else {
        outputMsg.innerText = msg
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
    audio.play()
}

async function healthCheck() {
    try {
        let res = await fetch('/ping')
        res = await res.json()

        if (res[0] == 'OK') {
            setStatus('server', 'online', 'success')
        } else {
            setStatus('server', 'offline', 'error')
        }
    } catch (e) {
        setStatus('server', 'offline', 'error')
    }
}

function showImages(req, res, outputContainer, livePreview) {
    let imageItemElements = outputContainer.querySelectorAll('.imgItem');

    res.output.forEach((result, index) => {
        if(typeof res != 'object') return;

        const imageData = result?.data || result?.path + '?t=' + new Date().getTime(),
            imageSeed = req.seed,
            imageWidth = req.width,
            imageHeight = req.height;

        if (!imageData.includes('/')) {
            // res contained no data for the image, stop execution

            setStatus('request', 'invalid image', 'error');
            return;
        }

        let imageItemElem = (index < imageItemElements.length ? imageItemElements[index] : null)

        if(!imageItemElem) {
            imageItemElem = document.createElement('div');
            imageItemElem.className = 'imgItem';
            imageItemElem.innerHTML = `
                <div class="imgContainer">
                    <img/>
                    <div class="imgItemInfo">
                        <span class="imgSeedLabel"></span>
                        <button class="imgUseBtn">Use as Input</button>
                        <button class="imgSaveBtn">Download</button>
                    </div>
                </div>
            `;

            const useAsInputBtn = imageItemElem.querySelector('.imgUseBtn'),
                saveImageBtn = imageItemElem.querySelector('.imgSaveBtn');

            useAsInputBtn.addEventListener('click', getUseAsInputHandler(imageItemElem));
            saveImageBtn.addEventListener('click', getSaveImageHandler(imageItemElem));

            outputContainer.appendChild(imageItemElem);
        }

        const imageElem = imageItemElem.querySelector('img'),
            imageSeedLabel = imageItemElem.querySelector('.imgSeedLabel');

        imageElem.src = imageData;
        imageElem.width = parseInt(imageWidth);
        imageElem.height = parseInt(imageHeight);
        imageElem.setAttribute('data-seed', imageSeed)

        const imageInfo = imageItemElem.querySelector('.imgItemInfo')
        imageInfo.style.visibility = (livePreview ? 'hidden' : 'visible')

        imageSeedLabel.innerText = 'Seed: ' + imageSeed;
    });
}

function getUseAsInputHandler(imageItemElem) {
    return function() {
        const imageElem = imageItemElem.querySelector('img')
        const imgData = imageElem.src;
        const imageSeed = imageElem.getAttribute('data-seed')

        initImageSelector.value = null;
        initImagePreview.src = imgData;

        initImagePreviewContainer.style.display = 'block';
        inpaintingEditorContainer.style.display = 'none';
        promptStrengthContainer.style.display = 'block';
        maskSetting.checked = false;

        // maskSetting.style.display = 'block';

        randomSeedField.checked = false;
        seedField.value = imageSeed;
        seedField.disabled = false;
    }
}

function getSaveImageHandler(imageItemElem) {
    return function() {
        const imageElem = imageItemElem.querySelector('img')
        const imgData = imageElem.src;
        const imageSeed = imageElem.getAttribute('data-seed')

        const imgDownload = document.createElement('a');
        imgDownload.download = createFileName(imageSeed);
        imgDownload.href = imgData;
        imgDownload.click();
    }
}

// makes a single image. don't call this directly, use makeImage() instead
async function doMakeImage(task) {
    if (task.stopped) {
        return
    }

    const reqBody = task.reqBody
    const batchCount = task.batchCount
    const outputContainer = task.outputContainer

    const outputMsg = task['outputMsg']
    const previewPrompt = task['previewPrompt']
    const progressBar = task['progressBar']

    let res = ''
    let seed = reqBody['seed']
    let numOutputs = parseInt(reqBody['num_outputs'])

    try {
        res = await fetch('/image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(reqBody)
        })

        let reader = res.body.getReader()
        let textDecoder = new TextDecoder()
        let finalJSON = ''
        let prevTime = -1
        while (true) {
            try {
                let t = new Date().getTime()

                const {value, done} = await reader.read()
                if (done) {
                    break
                }

                let timeTaken = (prevTime === -1 ? -1 : t - prevTime)

                let jsonStr = textDecoder.decode(value)

                try {
                    let stepUpdate = JSON.parse(jsonStr)

                    if (stepUpdate.step === undefined) {
                        finalJSON += jsonStr
                    } else {
                        let batchSize = stepUpdate.total_steps
                        let overallStepCount = stepUpdate.step + task.batchesDone * batchSize
                        let totalSteps = batchCount * batchSize
                        let percent = 100 * (overallStepCount / totalSteps)
                        percent = (percent > 100 ? 100 : percent)
                        percent = percent.toFixed(0)

                        stepsRemaining = totalSteps - overallStepCount
                        stepsRemaining = (stepsRemaining < 0 ? 0 : stepsRemaining)
                        timeRemaining = (timeTaken === -1 ? '' : stepsRemaining * timeTaken) // ms

                        outputMsg.innerHTML = `Batch ${task.batchesDone+1} of ${batchCount}`
                        outputMsg.innerHTML += `. Generating image(s): ${percent}%`

                        timeRemaining = (timeTaken !== -1 ? millisecondsToStr(timeRemaining) : '')

                        outputMsg.innerHTML += `. Time remaining (approx): ${timeRemaining}`
                        outputMsg.style.display = 'block'

                        if (stepUpdate.output !== undefined) {
                            showImages(reqBody, stepUpdate, outputContainer, true);
                        }
                    }
                } catch (e) {
                    finalJSON += jsonStr
                }

                prevTime = t
            } catch (e) {
                logError('Stable Diffusion had an error. Please check the logs in the command-line window.', res, outputMsg)
                res = undefined
                throw e
            }
        }

        if (res.status != 200) {
            if (serverStatus === 'online') {
                logError('Stable Diffusion had an error: ' + await res.text(), res, outputMsg)
            } else {
                logError("Stable Diffusion is still starting up, please wait. If this goes on beyond a few minutes, Stable Diffusion has probably crashed. Please check the error message in the command-line window.", res, outputMsg)
            }
            res = undefined
            progressBar.style.display = 'none'
        } else {
            if (finalJSON !== undefined && finalJSON.indexOf('}{') !== -1) {
                // hack for a middleman buffering all the streaming updates, and unleashing them
                //  on the poor browser in one shot.
                //  this results in having to parse JSON like {"step": 1}{"step": 2}...{"status": "succeeded"..}
                //  which is obviously invalid.
                // So we need to just extract the last {} section, starting from "status" to the end of the response

                let lastChunkIdx = finalJSON.lastIndexOf('}{')
                if (lastChunkIdx !== -1) {
                    let remaining = finalJSON.substring(lastChunkIdx)
                    finalJSON = remaining.substring(1)
                }
            }

            res = JSON.parse(finalJSON)

            if (res.status !== 'succeeded') {
                let msg = ''
                if (res.detail !== undefined) {
                    msg = res.detail

                    if (msg.toLowerCase().includes('out of memory')) {
                        msg += `<br/><br/>
                                <b>Suggestions</b>:
                                <br/>
                                1. If you have set an initial image, please try reducing its dimension to ${MAX_INIT_IMAGE_DIMENSION}x${MAX_INIT_IMAGE_DIMENSION} or smaller.<br/>
                                2. Try disabling the '<em>Turbo mode</em>' under '<em>Advanced Settings</em>'.<br/>
                                3. Try generating a smaller image.<br/>`
                    }
                } else {
                    msg = res
                }
                logError(msg, res, outputMsg)
                res = undefined
            }
        }
    } catch (e) {
        console.log('request error', e)
        logError('Stable Diffusion had an error. Please check the logs in the command-line window. <br/><br/>' + e + '<br/><pre>' + e.stack + '</pre>', res, outputMsg)
        setStatus('request', 'error', 'error')
        progressBar.style.display = 'none'
        res = undefined
    }

    if (!res) return false;

    lastPromptUsed = reqBody['prompt'];

    showImages(reqBody, res, outputContainer, false);

    return true;
}

async function checkTasks() {
    if (taskQueue.length === 0) {
        setStatus('request', 'done', 'success')
        setTimeout(checkTasks, 500)
        stopImageBtn.style.display = 'none'
        makeImageBtn.innerHTML = 'Make Image'

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
    makeImageBtn.innerHTML = 'Enqueue Next Image'
    bellPending = true

    previewTools.style.display = 'block'

    let task = taskQueue.pop()
    currentTask = task

    let time = new Date().getTime()

    let successCount = 0

    task.isProcessing = true
    task['stopTask'].innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop'
    task['taskStatusLabel'].innerText = "Processing"
    task['taskStatusLabel'].className += " activeTaskLabel"
    console.log(task['taskStatusLabel'].className)

    for (let i = 0; i < task.batchCount; i++) {
        task.reqBody['seed'] = task.seed + (i * task.reqBody['num_outputs'])

        let success = await doMakeImage(task)
        task.batchesDone++

        if (!task.isProcessing) {
            break
        }

        if (success) {
            successCount++
        }
    }

    task.isProcessing = false
    task['stopTask'].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    task['taskStatusLabel'].style.display = 'none'

    time = new Date().getTime() - time
    time /= 1000

    if (successCount === task.batchCount) {
        task.outputMsg.innerText = 'Processed ' + task.numOutputsTotal + ' images in ' + time + ' seconds'

        // setStatus('request', 'done', 'success')
    } else {
        task.outputMsg.innerText = 'Task ended after ' + time + ' seconds'
    }

    if (randomSeedField.checked) {
        seedField.value = task.seed
    }

    currentTask = null

    setTimeout(checkTasks, 10)
}
setTimeout(checkTasks, 0)

async function makeImage() {
    if (serverStatus !== 'online') {
        alert('The server is still starting up..')
        return
    }

    let task = {
        stopped: false,
        batchesDone: 0
    }

    let seed = (randomSeedField.checked ? Math.floor(Math.random() * 10000000) : parseInt(seedField.value))
    let numOutputsTotal = parseInt(numOutputsTotalField.value)
    let numOutputsParallel = parseInt(numOutputsParallelField.value)
    let batchCount = Math.ceil(numOutputsTotal / numOutputsParallel)
    let batchSize = numOutputsParallel

    let streamImageProgress = (numOutputsTotal > 50 ? false : streamImageProgressField.checked)

    let prompt = promptField.value
    if (activeTags.length > 0) {
        let promptTags = activeTags.map(x => x.name).join(", ");
        prompt += ", " + promptTags;
    }

    let reqBody = {
        session_id: sessionId,
        prompt: prompt,
        negative_prompt: negativePromptField.value.trim(),
        num_outputs: batchSize,
        num_inference_steps: numInferenceStepsField.value,
        guidance_scale: guidanceScaleField.value,
        width: widthField.value,
        height: heightField.value,
        // allow_nsfw: allowNSFWField.checked,
        turbo: turboField.checked,
        use_cpu: useCPUField.checked,
        use_full_precision: useFullPrecisionField.checked,
        stream_progress_updates: true,
        stream_image_progress: streamImageProgress,
        show_only_filtered_image: showOnlyFilteredImageField.checked
    }

    if (IMAGE_REGEX.test(initImagePreview.src)) {
        reqBody['init_image'] = initImagePreview.src
        reqBody['prompt_strength'] = promptStrengthField.value

        // if (IMAGE_REGEX.test(maskImagePreview.src)) {
        //     reqBody['mask'] = maskImagePreview.src
        // }
        if (maskSetting.checked) {
            reqBody['mask'] = inpaintingEditor.getImg()
        }

        reqBody['sampler'] = 'ddim'
    } else {
        reqBody['sampler'] = samplerField.value
    }

    if (saveToDiskField.checked && diskPathField.value.trim() !== '') {
        reqBody['save_to_disk_path'] = diskPathField.value.trim()
    }

    if (useFaceCorrectionField.checked) {
        reqBody['use_face_correction'] = 'GFPGANv1.3'
    }

    if (useUpscalingField.checked) {
        reqBody['use_upscale'] = upscaleModelField.value
    }

    let taskConfig = `Seed: ${seed}, Sampler: ${reqBody['sampler']}, Inference Steps: ${numInferenceStepsField.value}, Guidance Scale: ${guidanceScaleField.value}`

    if (negativePromptField.value.trim() !== '') {
        taskConfig += `, Negative Prompt: ${negativePromptField.value.trim()}`
    }

    if (reqBody['init_image'] !== undefined) {
        taskConfig += `, Prompt Strength: ${promptStrengthField.value}`
    }

    if (useFaceCorrectionField.checked) {
        taskConfig += `, Fix Faces: ${reqBody['use_face_correction']}`
    }

    if (useUpscalingField.checked) {
        taskConfig += `, Upscale: ${reqBody['use_upscale']}`
    }

    task['reqBody'] = reqBody
    task['seed'] = seed
    task['batchCount'] = batchCount
    task['isProcessing'] = false

    let taskEntry = document.createElement('div')
    taskEntry.className = 'imageTaskContainer'
    taskEntry.innerHTML = ` <div class="taskStatusLabel">Enqueued</div>
                            <button class="secondaryButton stopTask"><i class="fa-solid fa-trash-can"></i> Remove</button>
                            <div class="preview-prompt collapsible active"></div>
                            <div class="taskConfig">${taskConfig}</div>
                            <div class="collapsible-content" style="display: block">
                                <div class="outputMsg"></div>
                                <div class="progressBar"></div>
                                <div class="img-preview">
                            </div>`

    createCollapsibles(taskEntry)

    task['numOutputsTotal'] = numOutputsTotal
    task['taskStatusLabel'] = taskEntry.querySelector('.taskStatusLabel')
    task['outputContainer'] = taskEntry.querySelector('.img-preview')
    task['outputMsg'] = taskEntry.querySelector('.outputMsg')
    task['previewPrompt'] = taskEntry.querySelector('.preview-prompt')
    task['progressBar'] = taskEntry.querySelector('.progressBar')
    task['stopTask'] = taskEntry.querySelector('.stopTask')

    task['stopTask'].addEventListener('click', async function() {
        if (task['isProcessing']) {
            task.isProcessing = false
            try {
                let res = await fetch('/image/stop')
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

    imagePreview.insertBefore(taskEntry, previewTools.nextSibling)

    task['previewPrompt'].innerText = prompt

    taskQueue.unshift(task)

    initialText.style.display = 'none'
}

// create a file name with embedded prompt and metadata
// for easier cateloging and comparison
function createFileName(seed) {

    // Most important information is the prompt
    let underscoreName = lastPromptUsed.replace(/[^a-zA-Z0-9]/g, '_')
    underscoreName = underscoreName.substring(0, 100)
    const steps = numInferenceStepsField.value
    const guidance =  guidanceScaleField.value

    // name and the top level metadata
    let fileName = `${underscoreName}_Seed-${seed}_Steps-${steps}_Guidance-${guidance}`

    // add the tags
    // let tags = [];
    // let tagString = '';
    // document.querySelectorAll(modifyTagsSelector).forEach(function(tag) {
    //     tags.push(tag.innerHTML);
    // })

    // join the tags with a pipe
    // if (activeTags.length > 0) {
    //     tagString = '_Tags-';
    //     tagString += tags.join('|');
    // }

    // // append empty or populated tags
    // fileName += `${tagString}`;

    // add the file extension
    fileName += `.png`

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
        let res = await fetch('/image/stop')
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

soundToggle.addEventListener('click', handleBoolSettingChange(SOUND_ENABLED_KEY))
soundToggle.checked = isSoundEnabled()

saveToDiskField.checked = isSaveToDiskEnabled()
diskPathField.disabled = !saveToDiskField.checked

useFaceCorrectionField.addEventListener('click', handleBoolSettingChange(USE_FACE_CORRECTION_KEY))
useFaceCorrectionField.checked = isFaceCorrectionEnabled()

useUpscalingField.checked = isUpscalingEnabled()
upscaleModelField.disabled = !useUpscalingField.checked

showOnlyFilteredImageField.addEventListener('click', handleBoolSettingChange(SHOW_ONLY_FILTERED_IMAGE_KEY))
showOnlyFilteredImageField.checked = isShowOnlyFilteredImageEnabled()

useCPUField.addEventListener('click', handleBoolSettingChange(USE_CPU_KEY))
useCPUField.checked = isUseCPUEnabled()

useFullPrecisionField.addEventListener('click', handleBoolSettingChange(USE_FULL_PRECISION_KEY))
useFullPrecisionField.checked = isUseFullPrecisionEnabled()

turboField.addEventListener('click', handleBoolSettingChange(USE_TURBO_MODE_KEY))
turboField.checked = isUseTurboModeEnabled()

streamImageProgressField.addEventListener('click', handleBoolSettingChange(STREAM_IMAGE_PROGRESS_KEY))
streamImageProgressField.checked = isStreamImageProgressEnabled()

diskPathField.addEventListener('change', handleStringSettingChange(DISK_PATH_KEY))

saveToDiskField.addEventListener('click', function(e) {
    diskPathField.disabled = !this.checked
    handleBoolSettingChange(SAVE_TO_DISK_KEY)(e)
})

useUpscalingField.addEventListener('click', function(e) {
    upscaleModelField.disabled = !this.checked
    handleBoolSettingChange(USE_UPSCALING_KEY)(e)
})

function setPanelOpen(panelHandle) {
    let panelContents = panelHandle.nextElementSibling
    panelHandle.classList.add('active')
    panelContents.style.display = 'block'
}

if (isAdvancedPanelOpenEnabled()) {
    setPanelOpen(advancedPanelHandle)
}

if (isModifiersPanelOpenEnabled()) {
    setPanelOpen(modifiersPanelHandle)
}

makeImageBtn.addEventListener('click', makeImage)


function updateGuidanceScale() {
    guidanceScaleField.value = guidanceScaleSlider.value / 10
}

function updateGuidanceScaleSlider() {
    if (guidanceScaleField.value < 0) {
        guidanceScaleField.value = 0
    } else if (guidanceScaleField.value > 50) {
        guidanceScaleField.value = 50
    }

    guidanceScaleSlider.value = guidanceScaleField.value * 10
}

guidanceScaleSlider.addEventListener('input', updateGuidanceScale)
guidanceScaleField.addEventListener('input', updateGuidanceScaleSlider)
updateGuidanceScale()

function updatePromptStrength() {
    promptStrengthField.value = promptStrengthSlider.value / 100
}

function updatePromptStrengthSlider() {
    if (promptStrengthField.value < 0) {
        promptStrengthField.value = 0
    } else if (promptStrengthField.value > 0.99) {
        promptStrengthField.value = 0.99
    }

    promptStrengthSlider.value = promptStrengthField.value * 100
}

promptStrengthSlider.addEventListener('input', updatePromptStrength)
promptStrengthField.addEventListener('input', updatePromptStrengthSlider)
updatePromptStrength()

useBetaChannelField.addEventListener('click', async function(e) {
    if (serverStatus !== 'online') {
        // logError('The server is still starting up..')
        alert('The server is still starting up..')
        e.preventDefault()
        return false
    }

    let updateBranch = (this.checked ? 'beta' : 'main')

    try {
        let res = await fetch('/app_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                'update_branch': updateBranch
            })
        })
        res = await res.json()

        console.log('set config status response', res)
    } catch (e) {
        console.log('set config status error', e)
    }
})

async function getAppConfig() {
    try {
        let res = await fetch('/app_config')
        config = await res.json()

        if (config.update_branch === 'beta') {
            useBetaChannelField.checked = true
            updateBranchLabel.innerText = "(beta)"
        }

        console.log('get config status response', config)
    } catch (e) {
        console.log('get config status error', e)
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

    reader.addEventListener('load', function() {
        // console.log(file.name, reader.result)
        initImagePreview.src = reader.result
        initImagePreviewContainer.style.display = 'block'
        inpaintingEditorContainer.style.display = 'none'
        promptStrengthContainer.style.display = 'block'
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
    samplerSelectionContainer.style.display = 'block'
})

maskSetting.addEventListener('click', function() {
    inpaintingEditorContainer.style.display = (this.checked ? 'block' : 'none')
})

// function showMaskImagePreview() {
//     if (maskImageSelector.files.length === 0) {
//         // maskImagePreviewContainer.style.display = 'none'
//         return
//     }

//     let reader = new FileReader()
//     let file = maskImageSelector.files[0]

//     reader.addEventListener('load', function() {
//         // maskImagePreview.src = reader.result
//         // maskImagePreviewContainer.style.display = 'block'
//     })

//     if (file) {
//         reader.readAsDataURL(file)
//     }
// }
// maskImageSelector.addEventListener('change', showMaskImagePreview)
// showMaskImagePreview()

// maskImageClearBtn.addEventListener('click', function() {
//     maskImageSelector.value = null
//     maskImagePreview.src = ''
//     // maskImagePreviewContainer.style.display = 'none'
// })

// https://stackoverflow.com/a/8212878
function millisecondsToStr(milliseconds) {
    function numberEnding (number) {
        return (number > 1) ? 's' : '';
    }

    var temp = Math.floor(milliseconds / 1000);
    var hours = Math.floor((temp %= 86400) / 3600);
    var s = ''
    if (hours) {
        s += hours + ' hour' + numberEnding(hours) + ' ';
    }
    var minutes = Math.floor((temp %= 3600) / 60);
    if (minutes) {
        s += minutes + ' minute' + numberEnding(minutes) + ' ';
    }
    var seconds = temp % 60;
    if (!hours && minutes < 4 && seconds) {
        s += seconds + ' second' + numberEnding(seconds);
    }

    return s;
}

// https://gomakethings.com/finding-the-next-and-previous-sibling-elements-that-match-a-selector-with-vanilla-js/
function getNextSibling(elem, selector) {
	// Get the next sibling element
	var sibling = elem.nextElementSibling

	// If there's no selector, return the first sibling
	if (!selector) return sibling

	// If the sibling matches our selector, use it
	// If not, jump to the next sibling and continue the loop
	while (sibling) {
		if (sibling.matches(selector)) return sibling
		sibling = sibling.nextElementSibling
	}
}

function createCollapsibles(node) {
    if (!node) {
        node = document
    }

    let collapsibles = node.querySelectorAll(".collapsible")
    collapsibles.forEach(function(c) {
        let handle = document.createElement('span')
        handle.className = 'collapsible-handle'

        if (c.className.indexOf('active') !== -1) {
            handle.innerHTML = '&#x2796;' // minus
        } else {
            handle.innerHTML = '&#x2795;' // plus
        }
        c.insertBefore(handle, c.firstChild)

        c.addEventListener('click', function() {
            this.classList.toggle("active")
            let content = getNextSibling(this, '.collapsible-content')
            if (content.style.display === "block") {
                content.style.display = "none"
                handle.innerHTML = '&#x2795;' // plus
            } else {
                content.style.display = "block"
                handle.innerHTML = '&#x2796;' // minus
            }

            if (this == advancedPanelHandle) {
                let state = (content.style.display === 'block' ? 'true' : 'false')
                localStorage.setItem(ADVANCED_PANEL_OPEN_KEY, state)
            } else if (this == modifiersPanelHandle) {
                let state = (content.style.display === 'block' ? 'true' : 'false')
                localStorage.setItem(MODIFIERS_PANEL_OPEN_KEY, state)
            }
        })
    })
}
createCollapsibles()

function refreshTagsList() {
    editorModifierTagsList.innerHTML = '';

    if (activeTags.length == 0) {
        editorTagsContainer.style.display = 'none';
        return;
    } else {
        editorTagsContainer.style.display = 'block';
    }

    activeTags.forEach((tag, index) => {
        tag.element.querySelector('.modifier-card-image-overlay').innerText = '-';
        tag.element.classList.add('modifier-card-tiny');

        editorModifierTagsList.appendChild(tag.element);

        tag.element.addEventListener('click', () => {
            let idx = activeTags.indexOf(tag);

            if (idx !== -1) {
                activeTags[idx].originElement.classList.remove(activeCardClass);
                activeTags[idx].originElement.querySelector('.modifier-card-image-overlay').innerText = '+';

                activeTags.splice(idx, 1);
                refreshTagsList();
            }
        });
    });

    let brk = document.createElement('br')
    brk.style.clear = 'both'
    editorModifierTagsList.appendChild(brk)
}

async function getDiskPath() {
    try {
        let diskPath = getSavedDiskPath()

        if (diskPath !== '') {
            diskPathField.value = diskPath
            return
        }

        let res = await fetch('/output_dir')
        if (res.status === 200) {
            res = await res.json()
            res = res[0]

            document.querySelector('#diskPath').value = res
        }
    } catch (e) {
        console.log('error fetching output dir path', e)
    }
}

function createModifierCard(name, previews) {
    const modifierCard = document.createElement('div');
    modifierCard.className = 'modifier-card';
    modifierCard.innerHTML = `
    <div class="modifier-card-overlay"></div>
    <div class="modifier-card-image-container">
        <div class="modifier-card-image-overlay">+</div>
        <p class="modifier-card-error-label"></p>
        <img onerror="this.remove()" alt="Modifier Image" class="modifier-card-image">
    </div>
    <div class="modifier-card-container">
        <div class="modifier-card-label"><p></p></div>
    </div>`;

    const image = modifierCard.querySelector('.modifier-card-image');
    const errorText =  modifierCard.querySelector('.modifier-card-error-label');
    const label = modifierCard.querySelector('.modifier-card-label');

    errorText.innerText = 'No Image';

    if (typeof previews == 'object') {
        image.src = previews[0]; // portrait
        image.setAttribute('preview-type', 'portrait');
    } else {
        image.remove();
    }

    const maxLabelLength = 30;
    const nameWithoutBy = name.replace('by ', '');

    if(nameWithoutBy.length <= maxLabelLength) {
        label.querySelector('p').innerText = nameWithoutBy;
    } else {
        const tooltipText = document.createElement('span');
        tooltipText.className = 'tooltip-text';
        tooltipText.innerText = name;

        label.classList.add('tooltip');
        label.appendChild(tooltipText);

        label.querySelector('p').innerText = nameWithoutBy.substring(0, maxLabelLength) + '...';
    }

    return modifierCard;
}

function changePreviewImages(val) {
    const previewImages = document.querySelectorAll('.modifier-card-image-container img');

    let previewArr = [];

    modifiers.map(x => x.modifiers).forEach(x => previewArr.push(...x.map(m => m.previews)));
    
    previewArr = previewArr.map(x => {
        let obj = {};

        x.forEach(preview => {
            obj[preview.name] = preview.path;
        });
        
        return obj;
    });

    previewImages.forEach(previewImage => {
        const currentPreviewType = previewImage.getAttribute('preview-type');
        const relativePreviewPath = previewImage.src.split(modifierThumbnailPath + '/').pop();

        const previews = previewArr.find(preview => relativePreviewPath == preview[currentPreviewType]);

        if(typeof previews == 'object') {
            let preview = null;

            if (val == 'portrait') {
                preview = previews.portrait;
            }
            else if (val == 'landscape') {
                preview = previews.landscape;
            }

            if(preview != null) {
                previewImage.src = `${modifierThumbnailPath}/${preview}`;
                previewImage.setAttribute('preview-type', val);
            }
        }
    });
}

function resizeModifierCards(val) {
    const cardSizePrefix = 'modifier-card-size_';
    const modifierCardClass = 'modifier-card';

    const modifierCards = document.querySelectorAll(`.${modifierCardClass}`);
    const cardSize = n => `${cardSizePrefix}${n}`;

    modifierCards.forEach(card => {
        // remove existing size classes
        const classes = card.className.split(' ').filter(c => !c.startsWith(cardSizePrefix));
        card.className = classes.join(' ').trim();

        if(val != 0)
            card.classList.add(cardSize(val));
    });
}

async function loadModifiers() {
    try {
        let res = await fetch('/modifiers.json?v=2')
        if (res.status === 200) {
            res = await res.json()

            modifiers = res; // update global variable

            res.forEach((modifierGroup, idx) => {
                const title = modifierGroup.category;
                const modifiers = modifierGroup.modifiers;

                const titleEl = document.createElement('h5');
                titleEl.className = 'collapsible';
                titleEl.innerText = title;

                const modifiersEl = document.createElement('div');
                modifiersEl.classList.add('collapsible-content', 'editor-modifiers-leaf');

                if (idx == 0) {
                    titleEl.className += ' active'
                    modifiersEl.style.display = 'block'
                }

                modifiers.forEach(modObj => {
                    const modifierName = modObj.modifier;
                    const modifierPreviews = modObj?.previews?.map(preview => `${modifierThumbnailPath}/${preview.path}`);

                    const modifierCard = createModifierCard(modifierName, modifierPreviews);

                    if(typeof modifierCard == 'object') {
                        modifiersEl.appendChild(modifierCard);

                        modifierCard.addEventListener('click', () => {
                            if (activeTags.map(x => x.name).includes(modifierName)) {
                                // remove modifier from active array
                                activeTags = activeTags.filter(x => x.name != modifierName);
                                modifierCard.classList.remove(activeCardClass);
                                
                                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+';
                            } else {
                                // add modifier to active array
                                activeTags.push({
                                    'name': modifierName,
                                    'element': modifierCard.cloneNode(true),
                                    'originElement': modifierCard,
                                    'previews': modifierPreviews
                                });

                                modifierCard.classList.add(activeCardClass);

                                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '-';
                            }

                            refreshTagsList();
                        });
                    }
                });

                let brk = document.createElement('br')
                brk.style.clear = 'both'
                modifiersEl.appendChild(brk)

                let e = document.createElement('div')
                e.appendChild(titleEl)
                e.appendChild(modifiersEl)

                editorModifierEntries.appendChild(e)
            })

            createCollapsibles(editorModifierEntries)
        }
    } catch (e) {
        console.log('error fetching modifiers', e)
    }
}