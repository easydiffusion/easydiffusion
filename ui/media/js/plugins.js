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
    MODIFIERS_LOAD: []
}

async function loadUIPlugins() {
    try {
        let res = await fetch('/get/ui_plugins')
        if (res.status === 200) {
            res = await res.json()
            res.forEach(pluginPath => {
                let script = document.createElement('script')
                script.src = pluginPath + '?t=' + Date.now()

                console.log('loading plugin', pluginPath)

                document.head.appendChild(script)
            })
        }
    } catch (e) {
        console.log('error fetching plugin paths', e)
    }
}
