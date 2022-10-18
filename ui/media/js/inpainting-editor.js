const INPAINTING_EDITOR_SIZE = 450

let inpaintingEditorContainer = document.querySelector('#inpaintingEditor')
let inpaintingEditor = new DrawingBoard.Board('inpaintingEditor', {
    color: "#ffffff",
    background: false,
    size: 30,
    webStorage: false,
    controls: [{'DrawingMode': {'filler': false}}, 'Size', 'Navigation']
})
let inpaintingEditorCanvasBackground = document.querySelector('.drawing-board-canvas-wrapper')

function resizeInpaintingEditor(widthValue, heightValue) {
    if (widthValue === heightValue) {
        widthValue = INPAINTING_EDITOR_SIZE
        heightValue = INPAINTING_EDITOR_SIZE
    } else if (widthValue > heightValue) {
        heightValue = (heightValue / widthValue) * INPAINTING_EDITOR_SIZE
        widthValue = INPAINTING_EDITOR_SIZE
    } else {
        widthValue = (widthValue / heightValue) * INPAINTING_EDITOR_SIZE
        heightValue = INPAINTING_EDITOR_SIZE
    }
    if (inpaintingEditor.opts.aspectRatio === (widthValue / heightValue).toFixed(3)) {
        // Same ratio, don't reset the canvas.
        return
    }
    inpaintingEditor.opts.aspectRatio = (widthValue / heightValue).toFixed(3)

    inpaintingEditorContainer.style.width = widthValue + 'px'
    inpaintingEditorContainer.style.height = heightValue + 'px'
    inpaintingEditor.opts.enlargeYourContainer = true

    inpaintingEditor.opts.size = inpaintingEditor.ctx.lineWidth
    inpaintingEditor.resize()

    inpaintingEditor.ctx.lineCap = "round"
    inpaintingEditor.ctx.lineJoin = "round"
    inpaintingEditor.ctx.lineWidth = inpaintingEditor.opts.size
    inpaintingEditor.setColor(inpaintingEditor.opts.color)
}