const htmlTaskMap = new WeakMap()

const pauseBtn = document.querySelector("#pause")
const resumeBtn = document.querySelector("#resume")
const processOrder = document.querySelector("#process_order_toggle")

let TASK_CALLBACKS = {
    before_task_start: [],
    after_task_start: [],
    on_task_step: [],
    on_render_task_success: [],
    on_render_task_fail: [],
    on_all_tasks_complete: [],
}

let pauseClient = false

async function onIdle() {
    const serverCapacity = SD.serverCapacity
    if (pauseClient === true) {
        await resumeClient()
    }

    for (const taskEntry of getUncompletedTaskEntries()) {
        if (SD.activeTasks.size >= serverCapacity) {
            break
        }
        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            const taskStatusLabel = taskEntry.querySelector(".taskStatusLabel")
            taskStatusLabel.style.display = "none"
            continue
        }
        await onTaskStart(task)
    }
}

function getUncompletedTaskEntries() {
    const taskEntries = Array.from(document.querySelectorAll("#preview .imageTaskContainer .taskStatusLabel"))
        .filter((taskLabel) => taskLabel.style.display !== "none")
        .map(function(taskLabel) {
            let imageTaskContainer = taskLabel.parentNode
            while (!imageTaskContainer.classList.contains("imageTaskContainer") && imageTaskContainer.parentNode) {
                imageTaskContainer = imageTaskContainer.parentNode
            }
            return imageTaskContainer
        })
    if (!processOrder.checked) {
        taskEntries.reverse()
    }
    return taskEntries
}

async function onTaskStart(task) {
    if (!task.isProcessing || task.batchesDone >= task.batchCount) {
        return
    }

    if (typeof task.startTime !== "number") {
        task.startTime = Date.now()
    }
    if (!("instances" in task)) {
        task["instances"] = []
    }

    task["stopTask"].innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop'
    task["taskStatusLabel"].innerText = "Starting"
    task["taskStatusLabel"].classList.add("waitingTaskLabel")

    let newTaskReqBody = task.reqBody
    if (task.batchCount > 1) {
        // Each output render batch needs it's own task reqBody instance to avoid altering the other runs after they are completed.
        newTaskReqBody = Object.assign({}, task.reqBody)
        if (task.batchesDone == task.batchCount - 1) {
            // Last batch of the task
            // If the number of parallel jobs is no factor of the total number of images, the last batch must create less than "parallel jobs count" images
            // E.g. with numOutputsTotal = 6 and num_outputs = 5, the last batch shall only generate 1 image.
            newTaskReqBody.num_outputs = task.numOutputsTotal - task.reqBody.num_outputs * (task.batchCount - 1)
        }
    }

    const startSeed = task.seed || newTaskReqBody.seed
    const genSeeds = Boolean(
        typeof newTaskReqBody.seed !== "number" || (newTaskReqBody.seed === task.seed && task.numOutputsTotal > 1)
    )
    if (genSeeds) {
        newTaskReqBody.seed = parseInt(startSeed) + task.batchesDone * task.reqBody.num_outputs
    }

    const outputContainer = document.createElement("div")
    outputContainer.className = "img-batch"
    task.outputContainer.insertBefore(outputContainer, task.outputContainer.firstChild)

    const eventInfo = { reqBody: newTaskReqBody }
    const callbacksPromises = PLUGINS["TASK_CREATE"].map((hook) => {
        if (typeof hook !== "function") {
            console.error("The provided TASK_CREATE hook is not a function. Hook: %o", hook)
            return Promise.reject(new Error("hook is not a function."))
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
            console.error(
                `${factory ? "Factory " + String(factory) : "No factory defined"} for output format ${eventInfo.reqBody
                    ?.output_format || newTaskReqBody.output_format}. Instance is ${instance ||
                    "undefined"}. Using default renderer.`
            )
            instance = new SD.RenderTask(eventInfo.reqBody || newTaskReqBody)
        }
    }

    task["instances"].push(instance)
    task.batchesDone++

    TASK_CALLBACKS["before_task_start"].forEach((callback) => callback(task))

    instance.enqueue(getTaskUpdater(task, newTaskReqBody, outputContainer)).then(
        (renderResult) => {
            onRenderTaskCompleted(task, newTaskReqBody, instance, outputContainer, renderResult)
        },
        (reason) => {
            onTaskErrorHandler(task, newTaskReqBody, instance, reason)
        }
    )

    TASK_CALLBACKS["after_task_start"].forEach((callback) => callback(task))
}

function getTaskUpdater(task, reqBody, outputContainer) {
    const outputMsg = task["outputMsg"]
    const progressBar = task["progressBar"]
    const progressBarInner = progressBar.querySelector("div")

    const batchCount = task.batchCount
    let lastStatus = undefined
    return async function(event) {
        if (this.status !== lastStatus) {
            lastStatus = this.status
            switch (this.status) {
                case SD.TaskStatus.pending:
                    task["taskStatusLabel"].innerText = "Pending"
                    task["taskStatusLabel"].classList.add("waitingTaskLabel")
                    break
                case SD.TaskStatus.waiting:
                    task["taskStatusLabel"].innerText = "Waiting"
                    task["taskStatusLabel"].classList.add("waitingTaskLabel")
                    task["taskStatusLabel"].classList.remove("activeTaskLabel")
                    break
                case SD.TaskStatus.processing:
                case SD.TaskStatus.completed:
                    task["taskStatusLabel"].innerText = "Processing"
                    task["taskStatusLabel"].classList.add("activeTaskLabel")
                    task["taskStatusLabel"].classList.remove("waitingTaskLabel")
                    break
                case SD.TaskStatus.stopped:
                    break
                case SD.TaskStatus.failed:
                    if (!SD.isServerAvailable()) {
                        logError(
                            "Stable Diffusion is still starting up, please wait. If this goes on beyond a few minutes, Stable Diffusion has probably crashed. Please check the error message in the command-line window.",
                            event,
                            outputMsg
                        )
                    } else if (typeof event?.response === "object") {
                        let msg = "Stable Diffusion had an error reading the response:<br/><pre>"
                        if (this.exception) {
                            msg += `Error: ${this.exception.message}<br/>`
                        }
                        try {
                            // 'Response': body stream already read
                            msg += "Read: " + (await event.response.text())
                        } catch (e) {
                            msg += "Unexpected end of stream. "
                        }
                        const bufferString = event.reader.bufferedString
                        if (bufferString) {
                            msg += "Buffered data: " + bufferString
                        }
                        msg += "</pre>"
                        logError(msg, event, outputMsg)
                    }
                    break
            }
        }
        if ("update" in event) {
            const stepUpdate = event.update
            if (!("step" in stepUpdate)) {
                return
            }
            // task.instances can be a mix of different tasks with uneven number of steps (Render Vs Filter Tasks)
            const instancesWithProgressUpdates = task.instances.filter((instance) => instance.step !== undefined)
            const overallStepCount =
                instancesWithProgressUpdates.reduce(
                    (sum, instance) =>
                        sum +
                        (instance.isPending
                            ? Math.max(0, instance.step || stepUpdate.step) /
                              (instance.total_steps || stepUpdate.total_steps)
                            : 1),
                    0 // Initial value
                ) * stepUpdate.total_steps // Scale to current number of steps.
            const totalSteps = instancesWithProgressUpdates.reduce(
                (sum, instance) => sum + (instance.total_steps || stepUpdate.total_steps),
                stepUpdate.total_steps * (batchCount - task.batchesDone) // Initial value at (unstarted task count * Nbr of steps)
            )
            const percent = Math.min(100, 100 * (overallStepCount / totalSteps)).toFixed(0)

            const timeTaken = stepUpdate.step_time // sec
            const stepsRemaining = Math.max(0, totalSteps - overallStepCount)
            const timeRemaining = timeTaken < 0 ? "" : millisecondsToStr(stepsRemaining * timeTaken * 1000)
            outputMsg.innerHTML = `Batch ${task.batchesDone} of ${batchCount}. Generating image(s): ${percent}%. Time remaining (approx): ${timeRemaining}`
            outputMsg.style.display = "block"
            progressBarInner.style.width = `${percent}%`

            if (stepUpdate.output) {
                TASK_CALLBACKS["on_task_step"].forEach((callback) =>
                    callback(task, reqBody, stepUpdate, outputContainer)
                )
            }
        }
    }
}

function onRenderTaskCompleted(task, reqBody, instance, outputContainer, stepUpdate) {
    if (typeof stepUpdate === "object") {
        if (stepUpdate.status === "succeeded") {
            TASK_CALLBACKS["on_render_task_success"].forEach((callback) =>
                callback(task, reqBody, stepUpdate, outputContainer)
            )
        } else {
            task.isProcessing = false
            TASK_CALLBACKS["on_render_task_fail"].forEach((callback) =>
                callback(task, reqBody, stepUpdate, outputContainer)
            )
        }
    }
    if (task.isProcessing && task.batchesDone < task.batchCount) {
        task["taskStatusLabel"].innerText = "Pending"
        task["taskStatusLabel"].classList.add("waitingTaskLabel")
        task["taskStatusLabel"].classList.remove("activeTaskLabel")
        return
    }
    if ("instances" in task && task.instances.some((ins) => ins != instance && ins.isPending)) {
        return
    }

    task.isProcessing = false
    task["stopTask"].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
    task["taskStatusLabel"].style.display = "none"

    let time = millisecondsToStr(Date.now() - task.startTime)

    if (task.batchesDone == task.batchCount) {
        if (!task.outputMsg.innerText.toLowerCase().includes("error")) {
            task.outputMsg.innerText = `Processed ${task.numOutputsTotal} images in ${time}`
        }
        task.progressBar.style.height = "0px"
        task.progressBar.style.border = "0px solid var(--background-color3)"
        task.progressBar.classList.remove("active")
        // setStatus("request", "done", "success")
    } else {
        task.outputMsg.innerText += `. Task ended after ${time}`
    }

    // if (randomSeedField.checked) { // we already update this before the task starts
    //     seedField.value = task.seed
    // }

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

    TASK_CALLBACKS["on_all_tasks_complete"].forEach((callback) => callback())
}

function resumeClient() {
    if (pauseClient) {
        document.body.classList.remove("wait-pause")
        document.body.classList.add("pause")
    }
    return new Promise((resolve) => {
        let playbuttonclick = function() {
            resumeBtn.removeEventListener("click", playbuttonclick)
            resolve("resolved")
        }
        resumeBtn.addEventListener("click", playbuttonclick)
    })
}

function abortTask(task) {
    if (!task.isProcessing) {
        return false
    }
    task.isProcessing = false
    task.progressBar.classList.remove("active")
    task["taskStatusLabel"].style.display = "none"
    task["stopTask"].innerHTML = '<i class="fa-solid fa-trash-can"></i> Remove'
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

async function stopAllTasks() {
    getUncompletedTaskEntries().forEach((taskEntry) => {
        const taskStatusLabel = taskEntry.querySelector(".taskStatusLabel")
        if (taskStatusLabel) {
            taskStatusLabel.style.display = "none"
        }
        const task = htmlTaskMap.get(taskEntry)
        if (!task) {
            return
        }
        abortTask(task)
    })
}

function onTaskErrorHandler(task, reqBody, instance, reason) {
    if (!task.isProcessing) {
        return
    }
    console.log("Render request %o, Instance: %o, Error: %s", reqBody, instance, reason)
    abortTask(task)
    const outputMsg = task["outputMsg"]
    logError(
        "Stable Diffusion had an error. Please check the logs in the command-line window. <br/><br/>" +
            reason +
            "<br/><pre>" +
            reason.stack +
            "</pre>",
        task,
        outputMsg
    )
    // setStatus("request", "error", "error")
}

pauseBtn.addEventListener("click", function() {
    pauseClient = true
    pauseBtn.style.display = "none"
    resumeBtn.style.display = "inline"
    document.body.classList.add("wait-pause")
})

resumeBtn.addEventListener("click", function() {
    pauseClient = false
    resumeBtn.style.display = "none"
    pauseBtn.style.display = "inline"
    document.body.classList.remove("pause")
    document.body.classList.remove("wait-pause")
})
