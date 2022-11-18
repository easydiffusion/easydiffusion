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
    TASK_CREATE: [],
}

async function loadUIPlugins() {
    try {
        const res = await fetch('/get/ui_plugins')
        if (!res.ok) {
            console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
            return
        }
        const plugins = await res.json()
        const loadingPromises = plugins.map((pluginPath) => {
            const script = document.createElement('script')
            const promiseSrc = new PromiseSource()
            script.addEventListener('error', () => promiseSrc.reject(new Error(`Plugin "${pluginPath}" couldn't be loaded.`)))
            script.addEventListener('load', () => promiseSrc.resolve(pluginPath))
            script.src = pluginPath + '?t=' + Date.now()

            console.log('loading plugin', pluginPath)
            document.head.appendChild(script)

            return promiseSrc.promise
        })
        return await Promise.allSettled(loadingPromises)
    } catch (e) {
        console.log('error fetching plugin paths', e)
    }
}
