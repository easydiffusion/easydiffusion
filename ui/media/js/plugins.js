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
    IMAGE_INFO_BUTTONS: []
}


PLUGINS['IMAGE_INFO_BUTTONS'].push({ text: 'Double Size', on_click: getStartNewTaskHandler('img2img_X2') })
PLUGINS['IMAGE_INFO_BUTTONS'].push({ text: 'Redo', on_click: getStartNewTaskHandler('img2img') })
PLUGINS['IMAGE_INFO_BUTTONS'].push({ text: 'Upscale', on_click: getStartNewTaskHandler('upscale'), filter: (req, img) => !req.use_upscale })

function getStartNewTaskHandler(mode) {
    return function(reqBody, img) {
        const newTaskRequest = getCurrentUserRequest()
        switch (mode) {
            case 'img2img':
            case 'img2img_X2':
                newTaskRequest.reqBody = Object.assign({}, reqBody, {
                    num_outputs: 1,
                    use_cpu: useCPUField.checked,
                })
                if (!newTaskRequest.reqBody.init_image || mode === 'img2img_X2') {
                    newTaskRequest.reqBody.sampler = 'ddim'
                    newTaskRequest.reqBody.prompt_strength = '0.5'
                    newTaskRequest.reqBody.init_image = img.src
                    delete newTaskRequest.reqBody.mask
                } else {
                    newTaskRequest.reqBody.seed = 1 + newTaskRequest.reqBody.seed
                }
                if (mode === 'img2img_X2') {
                    newTaskRequest.reqBody.width = reqBody.width * 2
                    newTaskRequest.reqBody.height = reqBody.height * 2
                    newTaskRequest.reqBody.num_inference_steps = Math.min(100, reqBody.num_inference_steps * 2)
                    if (useUpscalingField.checked) {
                        newTaskRequest.reqBody.use_upscale = upscaleModelField.value
                    } else {
                        delete newTaskRequest.reqBody.use_upscale
                    }
                }
                break
            case 'upscale':
                newTaskRequest.reqBody = Object.assign({}, reqBody, {
                    num_outputs: 1,
                    //use_face_correction: 'GFPGANv1.3',
                    use_upscale: upscaleModelField.value,
                })
                break
            default:
                throw new Error("Unknown upscale mode: " + mode)
        }
        newTaskRequest.seed = newTaskRequest.reqBody.seed
        newTaskRequest.numOutputsTotal = 1
        newTaskRequest.batchCount = 1
        createTask(newTaskRequest)
    }
}
