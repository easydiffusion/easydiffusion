const PLUGIN_API_VERSION = "1.0"

const PLUGINS = {
    /**
     * Register new buttons to show on each output image.
     * 
     * Example:
     * PLUGINS['IMAGE_INFO_BUTTONS'].push({
     *   text: 'Make a Similar Image',
     *   on_click: function(origRequest, image) {
     *     let newTaskRequest = getCurrentUserRequest()
     *     newTaskRequest.reqBody = Object.assign({}, origRequest, {
     *       init_image: image.src,
     *       prompt_strength: 0.7,
     *       seed: Math.floor(Math.random() * 10000000)
     *     })
     *     newTaskRequest.seed = newTaskRequest.reqBody.seed
     *     createTask(newTaskRequest)
     *   },
     *   filter: function(origRequest, image) {
     *     // this is an optional function. return true/false to show/hide the button
     *     // if this function isn't set, the button will always be visible
     *     return true
     *   }
     * })
     */
    IMAGE_INFO_BUTTONS: [],
    GET_PROMPTS_HOOK: [],
    MODIFIERS_LOAD: [],
    TASK_CREATE: [],
    OUTPUTS_FORMATS: new ServiceContainer(
        function png() { return (reqBody) => new SD.RenderTask(reqBody) }
        , function jpeg() { return (reqBody) => new SD.RenderTask(reqBody) }
        , function webp() { return (reqBody) => new SD.RenderTask(reqBody) }
    ),
}
PLUGINS.OUTPUTS_FORMATS.register = function(...args) {
    const service = ServiceContainer.prototype.register.apply(this, args)
    if (typeof outputFormatField !== 'undefined') {
        const newOption = document.createElement("option")
        newOption.setAttribute("value", service.name)
        newOption.innerText = service.name
        outputFormatField.appendChild(newOption)
    }
    return service
}

function loadScript(url) {
    const script = document.createElement('script')
    const promiseSrc = new PromiseSource()
    script.addEventListener('error', () => promiseSrc.reject(new Error(`Script "${url}" couldn't be loaded.`)))
    script.addEventListener('load', () => promiseSrc.resolve(url))
    script.src = url + '?t=' + Date.now()

    console.log('loading script', url)
    document.head.appendChild(script)

    return promiseSrc.promise
}

async function loadUIPlugins() {
    try {
        const res = await fetch('/get/ui_plugins')
        if (!res.ok) {
            console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
            return
        }
        const plugins = await res.json()
        const loadingPromises = plugins.map(loadScript)
        return await Promise.allSettled(loadingPromises)
    } catch (e) {
        console.log('error fetching plugin paths', e)
    }
}
