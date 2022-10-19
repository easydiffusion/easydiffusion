const PLUGIN_API_VERSION = "1.0"

const PLUGINS = {
    /**
     * Register new buttons to show on each output image.
     * 
     * Example:
     * PLUGINS['IMAGE_INFO_BUTTONS']['myCustomVariationButton'] = {
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
     *   }
     * }
     */
    IMAGE_INFO_BUTTONS: {}
}
