/** SD-UI Backend control and classes.
 */
;(function() {
    "use strict"
    const RETRY_DELAY_IF_BUFFER_IS_EMPTY = 1000 // ms
    const RETRY_DELAY_IF_SERVER_IS_BUSY = 30 * 1000 // ms, status_code 503, already a task running
    const RETRY_DELAY_ON_ERROR = 4000 // ms
    const TASK_STATE_SERVER_UPDATE_DELAY = 1500 // ms
    const SERVER_STATE_VALIDITY_DURATION = 90 * 1000 // ms - 90 seconds to allow ping to timeout more than once before killing tasks.
    const HEALTH_PING_INTERVAL = 5000 // ms
    const IDLE_COOLDOWN = 2500 // ms
    const CONCURRENT_TASK_INTERVAL = 100 // ms

    /** Connects to an endpoint and resumes connection after reaching end of stream until all data is received.
     * Allows closing the connection while the server buffers more data.
     */
    class ChunkedStreamReader {
        #bufferedString = "" // Data received waiting to be read.
        #url
        #fetchOptions
        #response

        constructor(url, initialContent = "", options = {}) {
            if (typeof url !== "string" && !(url instanceof String)) {
                throw new Error("url is not a string.")
            }
            if (typeof initialContent !== "undefined" && typeof initialContent !== "string") {
                throw new Error("initialContent is not a string.")
            }
            this.#bufferedString = initialContent
            this.#url = url
            this.#fetchOptions = Object.assign(
                {
                    headers: {
                        "Content-Type": "application/json",
                    },
                },
                options
            )
            this.onNext = undefined
        }

        get url() {
            if (this.#response.redirected) {
                return this.#response.url
            }
            return this.#url
        }
        get bufferedString() {
            return this.#bufferedString
        }
        get status() {
            this.#response?.status
        }
        get statusText() {
            this.#response?.statusText
        }

        parse(value) {
            if (typeof value === "undefined") {
                return
            }
            if (!isArrayOrTypedArray(value)) {
                return [value]
            }
            if (value.length === 0) {
                return value
            }
            if (typeof this.textDecoder === "undefined") {
                this.textDecoder = new TextDecoder()
            }
            return [this.textDecoder.decode(value)]
        }
        onComplete(value) {
            return value
        }
        onError(response) {
            throw new Error(response.statusText)
        }
        onNext({ value, done }, response) {
            return { value, done }
        }

        async *[Symbol.asyncIterator]() {
            return this.open()
        }
        async *open() {
            let value = undefined
            let done = undefined
            do {
                if (this.#response) {
                    await asyncDelay(RETRY_DELAY_IF_BUFFER_IS_EMPTY)
                }
                this.#response = await fetch(this.#url, this.#fetchOptions)
                if (!this.#response.ok) {
                    if (this.#response.status === 425) {
                        continue
                    }
                    // Request status indicate failure
                    console.warn("Stream %o stopped unexpectedly.", this.#response)
                    value = await Promise.resolve(this.onError(this.#response))
                    if (typeof value === "boolean" && value) {
                        continue
                    }
                    return value
                }
                const reader = this.#response.body.getReader()
                done = false
                do {
                    const readState = await reader.read()
                    value = this.parse(readState.value)
                    if (value) {
                        for (let sVal of value) {
                            ;({ value: sVal, done } = await Promise.resolve(
                                this.onNext({ value: sVal, done: readState.done })
                            ))
                            yield sVal
                            if (done) {
                                return this.onComplete(sVal)
                            }
                        }
                    }
                    if (done) {
                        return
                    }
                } while (value && !done)
            } while (!done && (this.#response.ok || this.#response.status === 425))
        }
        *readStreamAsJSON(jsonStr, throwOnError) {
            if (typeof jsonStr !== "string") {
                throw new Error("jsonStr is not a string.")
            }
            do {
                if (this.#bufferedString.length > 0) {
                    // Append new data when required
                    if (jsonStr.length > 0) {
                        jsonStr = this.#bufferedString + jsonStr
                    } else {
                        jsonStr = this.#bufferedString
                    }
                    this.#bufferedString = ""
                }
                if (!jsonStr) {
                    return
                }
                // Find next delimiter
                let lastChunkIdx = jsonStr.indexOf("}{")
                if (lastChunkIdx >= 0) {
                    this.#bufferedString = jsonStr.substring(0, lastChunkIdx + 1)
                    jsonStr = jsonStr.substring(lastChunkIdx + 1)
                } else {
                    this.#bufferedString = jsonStr
                    jsonStr = ""
                }
                if (this.#bufferedString.length <= 0) {
                    return
                }
                // hack for a middleman buffering all the streaming updates, and unleashing them on the poor browser in one shot.
                // this results in having to parse JSON like {"step": 1}{"step": 2}{"step": 3}{"ste...
                // which is obviously invalid and can happen at any point while rendering.
                // So we need to extract only the next {} section
                try {
                    // Try to parse
                    const jsonObj = JSON.parse(this.#bufferedString)
                    this.#bufferedString = jsonStr
                    jsonStr = ""
                    yield jsonObj
                } catch (e) {
                    if (throwOnError) {
                        console.error(`Parsing: "${this.#bufferedString}", Buffer: "${jsonStr}"`)
                    }
                    this.#bufferedString += jsonStr
                    if (e instanceof SyntaxError && !throwOnError) {
                        return
                    }
                    throw e
                }
            } while (this.#bufferedString.length > 0 && this.#bufferedString.indexOf("}") >= 0)
        }
    }

    const EVENT_IDLE = "idle"
    const EVENT_STATUS_CHANGED = "statusChange"
    const EVENT_UNHANDLED_REJECTION = "unhandledRejection"
    const EVENT_TASK_QUEUED = "taskQueued"
    const EVENT_TASK_START = "taskStart"
    const EVENT_TASK_END = "taskEnd"
    const EVENT_TASK_ERROR = "task_error"
    const EVENT_PING = "ping"
    const EVENT_UNEXPECTED_RESPONSE = "unexpectedResponse"
    const EVENTS_TYPES = [
        EVENT_IDLE,
        EVENT_STATUS_CHANGED,
        EVENT_UNHANDLED_REJECTION,

        EVENT_TASK_QUEUED,
        EVENT_TASK_START,
        EVENT_TASK_END,
        EVENT_TASK_ERROR,
        EVENT_PING,

        EVENT_UNEXPECTED_RESPONSE,
    ]
    Object.freeze(EVENTS_TYPES)
    const eventSource = new GenericEventSource(EVENTS_TYPES)

    function setServerStatus(msgType, msg) {
        return eventSource.fireEvent(EVENT_STATUS_CHANGED, { type: msgType, message: msg })
    }

    const ServerStates = {
        init: "Init",
        loadingModel: "LoadingModel",
        online: "Online",
        rendering: "Rendering",
        unavailable: "Unavailable",
    }
    Object.freeze(ServerStates)

    let sessionId = Date.now()
    let serverState = { status: ServerStates.unavailable, time: Date.now() }

    async function healthCheck() {
        if (Date.now() < serverState.time + HEALTH_PING_INTERVAL / 2 && isServerAvailable()) {
            // Ping confirmed online less than half of HEALTH_PING_INTERVAL ago.
            return true
        }
        if (Date.now() >= serverState.time + SERVER_STATE_VALIDITY_DURATION) {
            console.warn("WARNING! SERVER_STATE_VALIDITY_DURATION has elapsed since the last Ping completed.")
        }
        try {
            let res = undefined
            if (typeof sessionId !== "undefined") {
                res = await fetch("/ping?session_id=" + sessionId)
            } else {
                res = await fetch("/ping")
            }
            serverState = await res.json()
            if (typeof serverState !== "object" || typeof serverState.status !== "string") {
                console.error(`Server reply didn't contain a state value.`)
                serverState = { status: ServerStates.unavailable, time: Date.now() }
                setServerStatus("error", "offline")
                return false
            }

            // Set status
            switch (serverState.status) {
                case ServerStates.init:
                    // Wait for init to complete before updating status.
                    break
                case ServerStates.online:
                    setServerStatus("online", "ready")
                    break
                case ServerStates.loadingModel:
                    setServerStatus("busy", "loading..")
                    break
                case ServerStates.rendering:
                    setServerStatus("busy", "rendering..")
                    break
                default:
                    // Unavailable
                    console.error("Ping received an unexpected server status. Status: %s", serverState.status)
                    setServerStatus("error", serverState.status.toLowerCase())
                    break
            }
            serverState.time = Date.now()
            await eventSource.fireEvent(EVENT_PING, serverState)
            return true
        } catch (e) {
            console.error(e)
            serverState = { status: ServerStates.unavailable, time: Date.now() }
            setServerStatus("error", "offline")
        }
        return false
    }

    function isServerAvailable() {
        if (typeof serverState !== "object") {
            console.error("serverState not set to a value. Connection to server could be lost...")
            return false
        }
        if (Date.now() >= serverState.time + SERVER_STATE_VALIDITY_DURATION) {
            console.warn("SERVER_STATE_VALIDITY_DURATION elapsed. Connection to server could be lost...")
            return false
        }
        switch (serverState.status) {
            case ServerStates.loadingModel:
            case ServerStates.rendering:
            case ServerStates.online:
                return true
            default:
                console.warn("Unexpected server status. Server could be unavailable... Status: %s", serverState.status)
                return false
        }
    }

    async function waitUntil(isReadyFn, delay, timeout) {
        if (typeof delay === "number") {
            const msDelay = delay
            delay = () => asyncDelay(msDelay)
        }
        if (typeof delay !== "function") {
            throw new Error("delay is not a number or a function.")
        }
        if (typeof timeout !== "undefined" && typeof timeout !== "number") {
            throw new Error("timeout is not a number.")
        }
        if (typeof timeout === "undefined" || timeout < 0) {
            timeout = Number.MAX_SAFE_INTEGER
        }
        timeout = Date.now() + timeout
        while (
            timeout > Date.now() &&
            Date.now() < serverState.time + SERVER_STATE_VALIDITY_DURATION &&
            !Boolean(await Promise.resolve(isReadyFn()))
        ) {
            await delay()
            if (!isServerAvailable()) {
                // Can fail if ping got frozen/suspended...
                if ((await healthCheck()) && isServerAvailable()) {
                    // Force a recheck of server status before failure...
                    continue // Continue waiting if last healthCheck confirmed the server is still alive.
                }
                throw new Error("Connection with server lost.")
            }
        }
        if (Date.now() >= serverState.time + SERVER_STATE_VALIDITY_DURATION) {
            console.warn("SERVER_STATE_VALIDITY_DURATION elapsed. Released waitUntil on stale server state.")
        }
    }

    const TaskStatus = {
        init: "init",
        pending: "pending", // Queued locally, not yet posted to server
        waiting: "waiting", // Waiting to run on server
        processing: "processing",
        stopped: "stopped",
        completed: "completed",
        failed: "failed",
    }
    Object.freeze(TaskStatus)

    const TASK_STATUS_ORDER = [
        TaskStatus.init,
        TaskStatus.pending,
        TaskStatus.waiting,
        TaskStatus.processing,
        //Don't add status that are final.
    ]

    const task_queue = new Map()
    const concurrent_generators = new Map()
    const weak_results = new WeakMap()

    class Task {
        // Private properties...
        _reqBody = {} // request body of this task.
        #reader = undefined
        #status = TaskStatus.init
        #id = undefined
        #exception = undefined

        constructor(options = {}) {
            this._reqBody = Object.assign({}, options)
            if (typeof this._reqBody.session_id === "undefined") {
                this._reqBody.session_id = sessionId
            } else if (
                this._reqBody.session_id !== SD.sessionId &&
                String(this._reqBody.session_id) !== String(SD.sessionId)
            ) {
                throw new Error("Use SD.sessionId to set the request session_id.")
            }
            this._reqBody.session_id = String(this._reqBody.session_id)
        }

        get id() {
            return this.#id
        }
        _setId(id) {
            if (typeof this.#id !== "undefined") {
                throw new Error("The task ID can only be set once.")
            }
            this.#id = id
        }

        get exception() {
            return this.#exception
        }
        async abort(exception) {
            if (this.isCompleted || this.isStopped || this.hasFailed) {
                return
            }
            if (typeof exception !== "undefined") {
                if (typeof exception === "string") {
                    exception = new Error(exception)
                }
                if (typeof exception !== "object") {
                    throw new Error("exception is not an object.")
                }
                if (!(exception instanceof Error)) {
                    throw new Error("exception is not an Error or a string.")
                }
            }
            const res = await fetch("/image/stop?task=" + this.id)
            if (!res.ok) {
                console.log("Stop response:", res)
                throw new Error(res.statusText)
            }
            task_queue.delete(this)
            this.#exception = exception
            this.#status = exception ? TaskStatus.failed : TaskStatus.stopped
        }

        get reqBody() {
            if (this.#status === TaskStatus.init) {
                return this._reqBody
            }
            console.warn("Task reqBody cannot be changed after the init state.")
            return Object.assign({}, this._reqBody)
        }

        get isPending() {
            return TASK_STATUS_ORDER.indexOf(this.#status) >= 0
        }
        get isCompleted() {
            return this.#status === TaskStatus.completed
        }
        get hasFailed() {
            return this.#status === TaskStatus.failed
        }
        get isStopped() {
            return this.#status === TaskStatus.stopped
        }
        get status() {
            return this.#status
        }
        _setStatus(status) {
            if (status === this.#status) {
                return
            }
            const currentIdx = TASK_STATUS_ORDER.indexOf(this.#status)
            if (currentIdx < 0) {
                throw Error(`The task status ${this.#status} is final and can't be changed.`)
            }
            const newIdx = TASK_STATUS_ORDER.indexOf(status)
            if (newIdx >= 0 && newIdx < currentIdx) {
                throw Error(`The task status ${status} can't replace ${this.#status}.`)
            }
            this.#status = status
        }

        /** Send current task to server.
         * @param {*} [timeout=-1] Optional timeout value in ms
         * @returns the response from the render request.
         * @memberof Task
         */
        async post(url, timeout = -1) {
            if (this.status !== TaskStatus.init && this.status !== TaskStatus.pending) {
                throw new Error(`Task status ${this.status} is not valid for post.`)
            }
            this._setStatus(TaskStatus.pending)
            Object.freeze(this._reqBody)

            const abortSignal = timeout >= 0 ? AbortSignal.timeout(timeout) : undefined
            let res = undefined
            try {
                this.checkReqBody()
                do {
                    abortSignal?.throwIfAborted()
                    res = await fetch(url, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(this._reqBody),
                        signal: abortSignal,
                    })
                    // status_code 503, already a task running.
                } while (res.status === 503 && (await asyncDelay(RETRY_DELAY_IF_SERVER_IS_BUSY)))
            } catch (err) {
                this.abort(err)
                throw err
            }
            if (!res.ok) {
                const err = new Error(`Unexpected response HTTP${res.status}. Details: ${res.statusText}`)
                this.abort(err)
                throw err
            }
            return await res.json()
        }

        static getReader(url) {
            const reader = new ChunkedStreamReader(url)
            const parseToString = reader.parse
            reader.parse = function(value) {
                value = parseToString.call(this, value)
                if (!value || value.length <= 0) {
                    return
                }
                return reader.readStreamAsJSON(value.join(""))
            }
            reader.onNext = function({ done, value }) {
                // By default is completed when the return value has a status defined.
                if (typeof value === "object" && "status" in value) {
                    done = true
                }
                return { done, value }
            }
            return reader
        }
        _setReader(reader) {
            if (typeof this.#reader !== "undefined") {
                throw new Error("The task reader can only be set once.")
            }
            this.#reader = reader
        }
        get reader() {
            if (this.#reader) {
                return this.#reader
            }
            if (!this.streamUrl) {
                throw new Error("The task has no stream Url defined.")
            }
            this.#reader = Task.getReader(this.streamUrl)
            const task = this
            const onNext = this.#reader.onNext
            this.#reader.onNext = function({ done, value }) {
                if (value && typeof value === "object") {
                    if (
                        task.status === TaskStatus.init ||
                        task.status === TaskStatus.pending ||
                        task.status === TaskStatus.waiting
                    ) {
                        task._setStatus(TaskStatus.processing)
                    }
                    if ("step" in value && "total_steps" in value) {
                        task.step = value.step
                        task.total_steps = value.total_steps
                    }
                }
                return onNext.call(this, { done, value })
            }
            this.#reader.onComplete = function(value) {
                task.result = value
                if (task.isPending) {
                    task._setStatus(TaskStatus.completed)
                }
                return value
            }
            this.#reader.onError = function(response) {
                const err = new Error(response.statusText)
                task.abort(err)
                throw err
            }
            return this.#reader
        }

        async waitUntil({ timeout = -1, callback, status, signal }) {
            const currentIdx = TASK_STATUS_ORDER.indexOf(this.#status)
            if (currentIdx <= 0) {
                return false
            }
            const stIdx = status ? TASK_STATUS_ORDER.indexOf(status) : currentIdx + 1
            if (stIdx >= 0 && stIdx <= currentIdx) {
                return true
            }
            if (stIdx < 0 && currentIdx < 0) {
                return this.#status === (status || TaskStatus.completed)
            }
            if (signal?.aborted) {
                return false
            }
            const task = this
            switch (this.#status) {
                case TaskStatus.pending:
                case TaskStatus.waiting:
                    // Wait for server status to include this task.
                    await waitUntil(
                        async () => {
                            if (
                                task.#id &&
                                typeof serverState.tasks === "object" &&
                                Object.keys(serverState.tasks).includes(String(task.#id))
                            ) {
                                return true
                            }
                            if ((await Promise.resolve(callback?.call(task))) || signal?.aborted) {
                                return true
                            }
                        },
                        TASK_STATE_SERVER_UPDATE_DELAY,
                        timeout
                    )
                    if (
                        this.#id &&
                        typeof serverState.tasks === "object" &&
                        Object.keys(serverState.tasks).includes(String(task.#id))
                    ) {
                        this._setStatus(TaskStatus.waiting)
                    }
                    if ((await Promise.resolve(callback?.call(this))) || signal?.aborted) {
                        return false
                    }
                    if (stIdx >= 0 && stIdx <= TASK_STATUS_ORDER.indexOf(TaskStatus.waiting)) {
                        return true
                    }
                    // Wait for task to start on server.
                    await waitUntil(
                        async () => {
                            if (
                                typeof serverState.tasks !== "object" ||
                                serverState.tasks[String(task.#id)] !== "pending"
                            ) {
                                return true
                            }
                            if ((await Promise.resolve(callback?.call(task))) || signal?.aborted) {
                                return true
                            }
                        },
                        TASK_STATE_SERVER_UPDATE_DELAY,
                        timeout
                    )
                    const state =
                        typeof serverState.tasks === "object" ? serverState.tasks[String(task.#id)] : undefined
                    if (state === "running" || state === "buffer" || state === "completed") {
                        this._setStatus(TaskStatus.processing)
                    }
                    if ((await Promise.resolve(callback?.call(task))) || signal?.aborted) {
                        return false
                    }
                    if (stIdx >= 0 && stIdx <= TASK_STATUS_ORDER.indexOf(TaskStatus.processing)) {
                        return true
                    }
                case TaskStatus.processing:
                    await waitUntil(
                        async () => {
                            if (
                                typeof serverState.tasks !== "object" ||
                                serverState.tasks[String(task.#id)] !== "running"
                            ) {
                                return true
                            }
                            if ((await Promise.resolve(callback?.call(task))) || signal?.aborted) {
                                return true
                            }
                        },
                        TASK_STATE_SERVER_UPDATE_DELAY,
                        timeout
                    )
                    await Promise.resolve(callback?.call(this))
                default:
                    return this.#status === (status || TaskStatus.completed)
            }
        }

        async enqueue(promiseGenerator, ...args) {
            if (this.status !== TaskStatus.init) {
                throw new Error(`Task is in an invalid status ${this.status} to add to queue.`)
            }
            this._setStatus(TaskStatus.pending)
            task_queue.set(this, promiseGenerator)
            await eventSource.fireEvent(EVENT_TASK_QUEUED, { task: this })
            await Task.enqueue(promiseGenerator, ...args)
            await this.waitUntil({ status: TaskStatus.completed })
            if (this.exception) {
                throw this.exception
            }
            return this.result
        }
        static async enqueue(promiseGenerator, ...args) {
            if (typeof promiseGenerator === "undefined") {
                throw new Error("To enqueue a concurrent task, a *Promise Generator is needed but undefined was found.")
            }
            //if (Symbol.asyncIterator in result || Symbol.iterator in result) {
            //concurrent_generators.set(result, Promise.resolve(args))
            if (typeof promiseGenerator === "function") {
                concurrent_generators.set(asGenerator({ callback: promiseGenerator }), Promise.resolve(args))
            } else {
                concurrent_generators.set(promiseGenerator, Promise.resolve(args))
            }
            await waitUntil(() => !concurrent_generators.has(promiseGenerator), CONCURRENT_TASK_INTERVAL)
            return weak_results.get(promiseGenerator)
        }
        static enqueueNew(task, classCtor, progressCallback) {
            if (task.status !== TaskStatus.init) {
                throw new Error("Task has an invalid status to add to queue.")
            }
            if (!(task instanceof classCtor)) {
                throw new Error("Task is not a instance of classCtor.")
            }
            let promiseGenerator = undefined
            if (typeof progressCallback === "undefined") {
                promiseGenerator = classCtor.start(task)
            } else if (typeof progressCallback === "function") {
                promiseGenerator = classCtor.start(task, progressCallback)
            } else {
                throw new Error("progressCallback is not a function.")
            }
            return Task.prototype.enqueue.call(task, promiseGenerator)
        }

        static async run(promiseGenerator, { callback, signal, timeout = -1 } = {}) {
            let value = undefined
            let done = undefined
            if (timeout < 0) {
                timeout = Number.MAX_SAFE_INTEGER
            }
            timeout = Date.now() + timeout
            do {
                ;({ value, done } = await Promise.resolve(promiseGenerator.next(value)))
                if (value instanceof Promise) {
                    value = await value
                }
                if (callback) {
                    ;({ value, done } = await Promise.resolve(callback.call(promiseGenerator, { value, done })))
                }
                if (value instanceof Promise) {
                    value = await value
                }
            } while (!done && !signal?.aborted && timeout > Date.now())
            return value
        }
        static async *asGenerator({ callback, generator, signal, timeout = -1 } = {}) {
            let value = undefined
            let done = undefined
            if (timeout < 0) {
                timeout = Number.MAX_SAFE_INTEGER
            }
            timeout = Date.now() + timeout
            do {
                ;({ value, done } = await Promise.resolve(generator.next(value)))
                if (value instanceof Promise) {
                    value = await value
                }
                if (callback) {
                    ;({ value, done } = await Promise.resolve(callback.call(generator, { value, done })))
                    if (value instanceof Promise) {
                        value = await value
                    }
                }
                value = yield value
            } while (!done && !signal?.aborted && timeout > Date.now())
            return value
        }
    }

    const TASK_REQUIRED = {
        session_id: "string",
        prompt: "string",
        negative_prompt: "string",
        width: "number",
        height: "number",
        seed: "number",

        sampler_name: "string",
        use_stable_diffusion_model: "string",
        clip_skip: "boolean",
        num_inference_steps: "number",
        guidance_scale: "number",

        num_outputs: "number",
        stream_progress_updates: "boolean",
        stream_image_progress: "boolean",
        show_only_filtered_image: "boolean",
        output_format: "string",
        output_quality: "number",
    }
    const TASK_DEFAULTS = {
        sampler_name: "plms",
        use_stable_diffusion_model: "sd-v1-4",
        clip_skip: false,
        num_inference_steps: 50,
        guidance_scale: 7.5,
        negative_prompt: "",

        num_outputs: 1,
        stream_progress_updates: true,
        stream_image_progress: true,
        show_only_filtered_image: true,
        block_nsfw: false,
        output_format: "png",
        output_quality: 75,
        output_lossless: false,
    }
    const TASK_OPTIONAL = {
        device: "string",
        init_image: "string",
        mask: "string",
        save_to_disk_path: "string",
        use_face_correction: "string",
        use_upscale: "string",
        use_vae_model: "string",
        use_hypernetwork_model: "string",
        hypernetwork_strength: "number",
        output_lossless: "boolean",
        tiling: "string",
    }

    // Higher values will result in...
    // pytorch_lightning/utilities/seed.py:60: UserWarning: X is not in bounds, numpy accepts from 0 to 4294967295
    const MAX_SEED_VALUE = 4294967295

    class RenderTask extends Task {
        constructor(options = {}) {
            super(options)
            if (typeof this._reqBody.seed === "undefined") {
                this._reqBody.seed = Math.floor(Math.random() * (MAX_SEED_VALUE + 1))
            }
            if (
                typeof typeof this._reqBody.seed === "number" &&
                (this._reqBody.seed > MAX_SEED_VALUE || this._reqBody.seed < 0)
            ) {
                throw new Error(`seed must be in range 0 to ${MAX_SEED_VALUE}.`)
            }

            if ("use_cpu" in this._reqBody) {
                if (this._reqBody.use_cpu) {
                    this._reqBody.device = "cpu"
                }
                delete this._reqBody.use_cpu
            }
            if (this._reqBody.init_image) {
                if (typeof this._reqBody.prompt_strength === "undefined") {
                    this._reqBody.prompt_strength = 0.8
                } else if (typeof this._reqBody.prompt_strength !== "number") {
                    throw new Error(
                        `prompt_strength need to be of type number but ${typeof this._reqBody
                            .prompt_strength} was found.`
                    )
                }
            }
            if ("modifiers" in this._reqBody) {
                if (Array.isArray(this._reqBody.modifiers) && this._reqBody.modifiers.length > 0) {
                    this._reqBody.modifiers = this._reqBody.modifiers.filter((val) => val.trim())
                    if (this._reqBody.modifiers.length > 0) {
                        this._reqBody.prompt = `${this._reqBody.prompt}, ${this._reqBody.modifiers.join(", ")}`
                    }
                }
                if (typeof this._reqBody.modifiers === "string" && this._reqBody.modifiers.length > 0) {
                    this._reqBody.modifiers = this._reqBody.modifiers.trim()
                    if (this._reqBody.modifiers.length > 0) {
                        this._reqBody.prompt = `${this._reqBody.prompt}, ${this._reqBody.modifiers}`
                    }
                }
                delete this._reqBody.modifiers
            }
            this.checkReqBody()
        }

        checkReqBody() {
            for (const key in TASK_DEFAULTS) {
                if (typeof this._reqBody[key] === "undefined") {
                    this._reqBody[key] = TASK_DEFAULTS[key]
                }
            }
            for (const key in TASK_REQUIRED) {
                if (typeof this._reqBody[key] !== TASK_REQUIRED[key]) {
                    throw new Error(
                        `${key} need to be of type ${TASK_REQUIRED[key]} but ${typeof this._reqBody[key]} was found.`
                    )
                }
            }
            for (const key in this._reqBody) {
                if (key in TASK_REQUIRED) {
                    continue
                }
                if (key in TASK_OPTIONAL) {
                    if (typeof this._reqBody[key] == "undefined") {
                        delete this._reqBody[key]
                        console.warn(`reqBody[${key}] was set to undefined. Removing optional key without value...`)
                        continue
                    }
                    if (typeof this._reqBody[key] !== TASK_OPTIONAL[key]) {
                        throw new Error(
                            `${key} need to be of type ${TASK_OPTIONAL[key]} but ${typeof this._reqBody[
                                key
                            ]} was found.`
                        )
                    }
                }
            }
        }

        /** Send current task to server.
         * @param {*} [timeout=-1] Optional timeout value in ms
         * @returns the response from the render request.
         * @memberof Task
         */
        async post(timeout = -1) {
            performance.mark("make-render-request")
            if (performance.getEntriesByName("click-makeImage", "mark").length > 0) {
                performance.measure("diff", "click-makeImage", "make-render-request")
                console.log(
                    "delay between clicking and making the server request:",
                    performance.getEntriesByName("diff", "measure")[0].duration + " ms"
                )
            }

            let jsonResponse = await super.post("/render", timeout)
            if (typeof jsonResponse?.task !== "number") {
                console.warn("Endpoint error response: ", jsonResponse)
                const event = Object.assign({ task: this }, jsonResponse)
                await eventSource.fireEvent(EVENT_UNEXPECTED_RESPONSE, event)
                if ("continueWith" in event) {
                    jsonResponse = await Promise.resolve(event.continueWith)
                }
                if (typeof jsonResponse?.task !== "number") {
                    const err = new Error(jsonResponse?.detail || "Endpoint response does not contains a task ID.")
                    this.abort(err)
                    throw err
                }
            }
            this._setId(jsonResponse.task)
            if (jsonResponse.stream) {
                this.streamUrl = jsonResponse.stream
            }
            this._setStatus(TaskStatus.waiting)
            return jsonResponse
        }

        enqueue(progressCallback) {
            return Task.enqueueNew(this, RenderTask, progressCallback)
        }
        *start(progressCallback) {
            if (typeof progressCallback !== "undefined" && typeof progressCallback !== "function") {
                throw new Error("progressCallback is not a function. progressCallback type: " + typeof progressCallback)
            }
            if (this.isStopped) {
                return
            }

            this._setStatus(TaskStatus.pending)
            progressCallback?.call(this, { reqBody: this._reqBody })
            Object.freeze(this._reqBody)

            // Post task request to backend
            let renderRequest = undefined
            try {
                renderRequest = yield this.post()
                yield progressCallback?.call(this, { renderResponse: renderRequest })
            } catch (e) {
                yield progressCallback?.call(this, { detail: e.message })
                throw e
            }
            try {
                // Wait for task to start on server.
                yield this.waitUntil({
                    callback: function() {
                        return progressCallback?.call(this, {})
                    },
                    status: TaskStatus.processing,
                })
            } catch (e) {
                this.abort(err)
                throw e
            }
            // Update class status and callback.
            const taskState = typeof serverState.tasks === "object" ? serverState.tasks[String(this.id)] : undefined
            switch (taskState) {
                case "pending": // Session has pending tasks.
                    console.error("Server %o render request %o is still waiting.", serverState, renderRequest)
                    //Only update status if not already set by waitUntil
                    if (this.status === TaskStatus.init || this.status === TaskStatus.pending) {
                        // Set status as Waiting in backend.
                        this._setStatus(TaskStatus.waiting)
                    }
                    break
                case "running":
                case "buffer":
                    // Normal expected messages.
                    this._setStatus(TaskStatus.processing)
                    break
                case "completed":
                    if (this.isPending) {
                        // Set state to processing until we read the reply
                        this._setStatus(TaskStatus.processing)
                    }
                    console.warn("Server %o render request %o completed unexpectedly", serverState, renderRequest)
                    break // Continue anyway to try to read cached result.
                case "error":
                    this._setStatus(TaskStatus.failed)
                    console.error("Server %o render request %o has failed", serverState, renderRequest)
                    break // Still valid, Update UI with error message
                case "stopped":
                    this._setStatus(TaskStatus.stopped)
                    console.log("Server %o render request %o was stopped", serverState, renderRequest)
                    return false
                default:
                    if (!progressCallback) {
                        const err = new Error("Unexpected server task state: " + taskState || "Undefined")
                        this.abort(err)
                        throw err
                    }
                    const response = yield progressCallback.call(this, {})
                    if (response instanceof Error) {
                        this.abort(response)
                        throw response
                    }
                    if (!response) {
                        return false
                    }
            }

            // Task started!
            // Open the reader.
            const reader = this.reader
            const task = this
            reader.onError = function(response) {
                if (progressCallback) {
                    task.abort(new Error(response.statusText))
                    return progressCallback.call(task, { response, reader })
                }
                return Task.prototype.onError.call(task, response)
            }
            yield progressCallback?.call(this, { reader })

            //Start streaming the results.
            const streamGenerator = reader.open()
            let value = undefined
            let done = undefined
            yield progressCallback?.call(this, { stream: streamGenerator })
            do {
                ;({ value, done } = yield streamGenerator.next())
                if (typeof value !== "object") {
                    continue
                }
                yield progressCallback?.call(this, { update: value })
            } while (!done)
            return value
        }
        static start(task, progressCallback) {
            if (typeof task !== "object") {
                throw new Error("task is not an object. task type: " + typeof task)
            }
            if (!(task instanceof Task)) {
                if (task.reqBody) {
                    task = new RenderTask(task.reqBody)
                } else {
                    task = new RenderTask(task)
                }
            }
            return task.start(progressCallback)
        }
        static run(task, progressCallback) {
            const promiseGenerator = RenderTask.start(task, progressCallback)
            return Task.run(promiseGenerator)
        }
    }
    class FilterTask extends Task {
        constructor(options = {}) {
            super(options)
        }
        /** Send current task to server.
         * @param {*} [timeout=-1] Optional timeout value in ms
         * @returns the response from the render request.
         * @memberof Task
         */
        async post(timeout = -1) {
            let jsonResponse = await super.post("/filter", timeout)
            if (typeof jsonResponse?.task !== "number") {
                console.warn("Endpoint error response: ", jsonResponse)
                const event = Object.assign({ task: this }, jsonResponse)
                await eventSource.fireEvent(EVENT_UNEXPECTED_RESPONSE, event)
                if ("continueWith" in event) {
                    jsonResponse = await Promise.resolve(event.continueWith)
                }
                if (typeof jsonResponse?.task !== "number") {
                    const err = new Error(jsonResponse?.detail || "Endpoint response does not contains a task ID.")
                    this.abort(err)
                    throw err
                }
            }
            this._setId(jsonResponse.task)
            if (jsonResponse.stream) {
                this.streamUrl = jsonResponse.stream
            }
            this._setStatus(TaskStatus.waiting)
            return jsonResponse
        }
        checkReqBody() {}
        enqueue(progressCallback) {
            return Task.enqueueNew(this, FilterTask, progressCallback)
        }
        *start(progressCallback) {
            if (typeof progressCallback !== "undefined" && typeof progressCallback !== "function") {
                throw new Error("progressCallback is not a function. progressCallback type: " + typeof progressCallback)
            }
            if (this.isStopped) {
                return
            }

            this._setStatus(TaskStatus.pending)
            progressCallback?.call(this, { reqBody: this._reqBody })
            Object.freeze(this._reqBody)

            // Post task request to backend
            let renderRes = undefined
            try {
                renderRes = yield this.post()
                yield progressCallback?.call(this, { renderResponse: renderRes })
            } catch (e) {
                yield progressCallback?.call(this, { detail: e.message })
                throw e
            }

            try {
                // Wait for task to start on server.
                yield this.waitUntil({
                    callback: function() {
                        return progressCallback?.call(this, {})
                    },
                    status: TaskStatus.processing,
                })
            } catch (e) {
                this.abort(err)
                throw e
            }

            // Task started!
            // Open the reader.
            const reader = this.reader
            const task = this
            reader.onError = function(response) {
                if (progressCallback) {
                    task.abort(new Error(response.statusText))
                    return progressCallback.call(task, { response, reader })
                }
                return Task.prototype.onError.call(task, response)
            }
            yield progressCallback?.call(this, { reader })

            //Start streaming the results.
            const streamGenerator = reader.open()
            let value = undefined
            let done = undefined
            yield progressCallback?.call(this, { stream: streamGenerator })
            do {
                ;({ value, done } = yield streamGenerator.next())
                if (typeof value !== "object") {
                    continue
                }
                if (value.status !== undefined) {
                    yield progressCallback?.call(this, value)
                    if (value.status === "succeeded" || value.status === "failed") {
                        done = true
                    }
                }
            } while (!done)
            return value
        }
        static start(task, progressCallback) {
            if (typeof task !== "object") {
                throw new Error("task is not an object. task type: " + typeof task)
            }
            if (!(task instanceof Task)) {
                if (task.reqBody) {
                    task = new FilterTask(task.reqBody)
                } else {
                    task = new FilterTask(task)
                }
            }
            return task.start(progressCallback)
        }
        static run(task, progressCallback) {
            const promiseGenerator = FilterTask.start(task, progressCallback)
            return Task.run(promiseGenerator)
        }
    }

    const getSystemInfo = debounce(
        async function() {
            let systemInfo = {
                devices: {
                    all: {},
                    active: {},
                },
                hosts: [],
            }
            try {
                const res = await fetch("/get/system_info")
                if (!res.ok) {
                    console.error("Invalid response fetching devices", res.statusText)
                    return systemInfo
                }
                systemInfo = await res.json()
            } catch (e) {
                console.error("error fetching system info", e)
            }
            return systemInfo
        },
        250,
        true
    )
    async function getDevices() {
        let systemInfo = getSystemInfo()
        return systemInfo.devices
    }
    async function getHosts() {
        let systemInfo = getSystemInfo()
        return systemInfo.hosts
    }

    async function getModels(scanForMalicious = true) {
        let models = {
            "stable-diffusion": [],
            vae: [],
        }
        try {
            const res = await fetch("/get/models?scan_for_malicious=" + scanForMalicious)
            if (!res.ok) {
                console.error("Invalid response fetching models", res.statusText)
                return models
            }
            models = await res.json()
            console.log("get models response", models)
        } catch (e) {
            console.log("get models error", e)
        }
        return models
    }

    function getServerCapacity() {
        let activeDevicesCount = Object.keys(serverState?.devices?.active || {}).length
        if (typeof window === "object" && window.document.visibilityState === "hidden") {
            activeDevicesCount = 1 + activeDevicesCount
        }
        return activeDevicesCount
    }

    let idleEventPromise = undefined
    function continueTasks() {
        if (typeof navigator?.scheduling?.isInputPending === "function") {
            const inputPendingOptions = {
                // Report mouse/pointer move events when queue is empty.
                // Delay idle after mouse moves stops.
                includeContinuous: Boolean(task_queue.size <= 0 && concurrent_generators.size <= 0),
            }
            if (navigator.scheduling.isInputPending(inputPendingOptions)) {
                // Browser/User still active.
                return asyncDelay(CONCURRENT_TASK_INTERVAL)
            }
        }
        const serverCapacity = getServerCapacity()
        if (task_queue.size <= 0 && concurrent_generators.size <= 0) {
            if (!idleEventPromise?.isPending) {
                idleEventPromise = makeQuerablePromise(
                    eventSource.fireEvent(EVENT_IDLE, { capacity: serverCapacity, idle: true })
                )
            }
            // Calling idle could result in task being added to queue.
            // if (task_queue.size <= 0 && concurrent_generators.size <= 0) {
            //     return asyncDelay(IDLE_COOLDOWN).then(() => idleEventPromise)
            // }
        }
        if (task_queue.size < serverCapacity) {
            if (!idleEventPromise?.isPending) {
                idleEventPromise = makeQuerablePromise(
                    eventSource.fireEvent(EVENT_IDLE, { capacity: serverCapacity - task_queue.size })
                )
            }
        }
        const completedTasks = []
        for (let [generator, promise] of concurrent_generators.entries()) {
            if (promise.isPending) {
                continue
            }
            let value = promise.resolvedValue?.value || promise.resolvedValue
            if (promise.isRejected) {
                console.error(promise.rejectReason)
                const event = { generator, reason: promise.rejectReason }
                eventSource.fireEvent(EVENT_UNHANDLED_REJECTION, event)
                if ("continueWith" in event) {
                    value = Promise.resolve(event.continueWith)
                } else {
                    concurrent_generators.delete(generator)
                    completedTasks.push({ generator, promise })
                    continue
                }
            }
            if (value instanceof Promise) {
                promise = makeQuerablePromise(value.then((val) => ({ done: promise.resolvedValue?.done, value: val })))
                concurrent_generators.set(generator, promise)
                continue
            }
            weak_results.set(generator, value)
            if (promise.resolvedValue?.done) {
                concurrent_generators.delete(generator)
                completedTasks.push({ generator, promise })
                continue
            }

            promise = generator.next(value)
            if (!(promise instanceof Promise)) {
                promise = Promise.resolve(promise)
            }
            promise = makeQuerablePromise(promise)
            concurrent_generators.set(generator, promise)
        }

        for (let [task, generator] of task_queue.entries()) {
            const cTsk = completedTasks.find((item) => item.generator === generator)
            if (cTsk?.promise?.rejectReason || task.hasFailed) {
                eventSource.fireEvent(EVENT_TASK_ERROR, {
                    task,
                    generator,
                    reason: cTsk?.promise?.rejectReason || task.exception,
                })
                task_queue.delete(task)
                continue
            }
            if (task.isCompleted || task.isStopped || cTsk) {
                const eventEndArgs = { task, generator }
                if (task.isStopped) {
                    eventEndArgs.stopped = true
                }
                eventSource.fireEvent(EVENT_TASK_END, eventEndArgs)
                task_queue.delete(task)
                continue
            }
            if (concurrent_generators.size > serverCapacity) {
                break
            }
            if (!generator) {
                if (typeof task.start === "function") {
                    generator = task.start()
                }
            } else if (concurrent_generators.has(generator)) {
                continue
            }
            const event = { task, generator }
            const beforeStart = eventSource.fireEvent(EVENT_TASK_START, event) // optional beforeStart promise to wait on before starting task.
            const promise = makeQuerablePromise(beforeStart.then(() => Promise.resolve(event.beforeStart)))
            concurrent_generators.set(event.generator, promise)
            task_queue.set(task, event.generator)
        }
        const promises = Array.from(concurrent_generators.values())
        if (promises.length <= 0) {
            return asyncDelay(CONCURRENT_TASK_INTERVAL)
        }
        return Promise.race(promises).finally(continueTasks)
    }
    let taskPromise = undefined
    function startCheck() {
        if (taskPromise?.isPending) {
            return
        }
        do {
            if (taskPromise?.resolvedValue instanceof Promise) {
                taskPromise = makeQuerablePromise(taskPromise.resolvedValue)
                continue
            }
            if (typeof navigator?.scheduling?.isInputPending === "function" && navigator.scheduling.isInputPending()) {
                return
            }
            const continuePromise = continueTasks().catch(async function(err) {
                console.error(err)
                await eventSource.fireEvent(EVENT_UNHANDLED_REJECTION, { reason: err })
                await asyncDelay(RETRY_DELAY_ON_ERROR)
            })
            taskPromise = makeQuerablePromise(continuePromise)
        } while (taskPromise?.isResolved)
    }

    const SD = {
        ChunkedStreamReader,
        ServerStates,
        TaskStatus,
        Task,
        RenderTask,
        FilterTask,

        Events: EVENTS_TYPES,
        init: async function(options = {}) {
            if ("events" in options) {
                for (const key in options.events) {
                    eventSource.addEventListener(key, options.events[key])
                }
            }
            await healthCheck()
            setInterval(healthCheck, HEALTH_PING_INTERVAL)
            setInterval(startCheck, CONCURRENT_TASK_INTERVAL)
        },

        /** Add a new event listener
         */
        addEventListener: (...args) => eventSource.addEventListener(...args),
        /** Remove the event listener
         */
        removeEventListener: (...args) => eventSource.removeEventListener(...args),

        isServerAvailable,
        getServerCapacity,

        getSystemInfo,
        getDevices,
        getHosts,

        getModels,

        render: (...args) => RenderTask.run(...args),
        filter: (...args) => FilterTask.run(...args),
        waitUntil,
    }

    Object.defineProperties(SD, {
        serverState: {
            configurable: false,
            get: () => serverState,
        },
        isAvailable: {
            configurable: false,
            get: () => isServerAvailable(),
        },
        serverCapacity: {
            configurable: false,
            get: () => getServerCapacity(),
        },
        sessionId: {
            configurable: false,
            get: () => sessionId,
            set: (val) => {
                if (typeof val === "undefined") {
                    throw new Error("Can't set sessionId to undefined.")
                }
                sessionId = val
            },
        },
        MAX_SEED_VALUE: {
            configurable: false,
            get: () => MAX_SEED_VALUE,
        },
        activeTasks: {
            configurable: false,
            get: () => task_queue,
        },
    })
    Object.defineProperties(getGlobal(), {
        SD: {
            configurable: false,
            get: () => SD,
        },
        sessionId: {
            //TODO Remove in the future in favor of SD.sessionId
            configurable: false,
            get: () => {
                console.warn("Deprecated window.sessionId has been replaced with SD.sessionId.")
                console.trace()
                return SD.sessionId
            },
            set: (val) => {
                console.warn("Deprecated window.sessionId has been replaced with SD.sessionId.")
                console.trace()
                SD.sessionId = val
            },
        },
    })
})()
