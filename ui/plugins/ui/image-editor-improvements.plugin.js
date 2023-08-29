/*
    Image Editor Improvements
    by Patrice

    Image editor improvements:
    - Shows the actual brush in the image editor for increased precision.
    - Add img2img source image via drag & drop from external file or browser image (incl. rendered image). Just drop the image in the editor pane.
    - Add img2img source image by pasting an image from the clipboard
    - Integrates seamlessly with Scrolling Panes 1.8+
    - Adds support for reloading task from metadata embedded in PNG and JPEG images (use Ctrl+Drop image in the editor pane)
    - Automatically sets the size of the output image to the size of the image used for img2img if its dimensions are both valid options (works with both copy/paste and drag & drop).
    - makes the brushes more visible in the image/inpainting editor.
*/
(function() {
    "use strict"

    let imageBrushPreview
    let imageCanvas
    let canvasType
    let activeBrush

    function setupBrush() {
        // capture active brush
        activeBrush = document.querySelector(canvasType + ' .image_editor_brush_size .editor-options-container .active')

        // create a copy of the brush if needed
        if (imageBrushPreview == undefined) {
            // create brush to display on canvas
            imageBrushPreview = activeBrush.cloneNode(true)
            imageBrushPreview.className = 'image-brush-preview'
            imageBrushPreview.style.display = 'none'
            imageCanvas.parentElement.appendChild(imageBrushPreview)
        }

        // render the brush
        imageBrushPreview.style.width = activeBrush.offsetWidth + 'px'
        imageBrushPreview.style.height = activeBrush.offsetWidth + 'px'
        imageBrushPreview.style.display = 'block'
    }

    function cleanupBrush() {
        // delete the brush copy if the mouse moves out of the canvas
        imageCanvas.style.cursor = ''
        if (imageBrushPreview !== undefined) {
            imageBrushPreview.remove()
            imageBrushPreview = undefined
        }
    }

    function disableRightClick(e) {
        e.preventDefault()
    }

    function setupCanvas(canvas) {
        canvasType = canvas
        imageCanvas = document.querySelector(canvas + ' .editor-canvas-overlay')
        imageCanvas.addEventListener("contextmenu", disableRightClick)
        imageCanvas.addEventListener("mousemove", updateMouseCursor)
        imageCanvas.addEventListener("mouseenter", setupBrush)
        imageCanvas.addEventListener("mouseleave", cleanupBrush)
    }

    document.getElementById("init_image_button_draw").addEventListener("click", () => {
        setupCanvas('#image-editor')
    })

    document.getElementById("init_image_button_inpaint").addEventListener("click", () => {
        setupCanvas('#image-inpainter')
    })

    function updateMouseCursor(e) {
        // move the brush
        if (imageBrushPreview !== undefined) {
            imageBrushPreview.style.left = e.clientX + 'px'
            imageBrushPreview.style.top = e.clientY + 'px'
        }
    }

    /* ADD SUPPORT FOR PASTING SOURCE IMAGE FROM CLIPBOARD */
    let imageObj = new Image()
    let canvas = document.createElement('canvas')
    let context = canvas.getContext('2d')

    imageObj.onload = function() {
        // Calculate the maximum cropped dimensions
        const step = customWidthField.step

        const maxCroppedWidth = Math.floor(this.width / step) * step;
        const maxCroppedHeight = Math.floor(this.height / step) * step;

        canvas.width = maxCroppedWidth;
        canvas.height = maxCroppedHeight;

        // Calculate the x and y coordinates to center the cropped image
        const x = (maxCroppedWidth - this.width) / 2;
        const y = (maxCroppedHeight - this.height) / 2;

        // Draw the image with centered coordinates
        context.drawImage(imageObj, x, y, this.width, this.height);

        let bestWidth = maxCroppedWidth - maxCroppedWidth % IMAGE_STEP_SIZE
        let bestHeight = maxCroppedHeight - maxCroppedHeight % IMAGE_STEP_SIZE

        addImageSizeOption(bestWidth)
        addImageSizeOption(bestHeight)

        // Set the width and height to the closest aspect ratio and closest to original dimensions
        widthField.value = bestWidth;
        heightField.value = bestHeight;

        initImagePreview.src = canvas.toDataURL('image/png');
    };
    
	function handlePaste(e) {
	    for (let i = 0 ; i < e.clipboardData.items.length ; i++) {
	        const item = e.clipboardData.items[i]
	        if (item.type.indexOf("image") != -1) {
                imageObj.src = URL.createObjectURL(item.getAsFile())
	        }
	    }
	}
    document.addEventListener('paste', handlePaste)

    // replace the default file open listener
    initImageSelector.removeEventListener('change', loadImg2ImgFromFile);
    function ieiLoadImg2ImgFromFile() {
        if (initImageSelector.files.length === 0) {
            return
        }

        let reader = new FileReader()
        let file = initImageSelector.files[0]

        reader.addEventListener('load', function(event) {
            imageObj.src = reader.result
        })

        if (file) {
            reader.readAsDataURL(file)
        }
    }
    initImageSelector.addEventListener('change', ieiLoadImg2ImgFromFile)


    /* ADD SUPPORT FOR DRAG-AND-DROPPING SOURCE IMAGE (from file or straight from UI) */
    
    /* DROP AREAS */

    function createDropAreas(container) {
        // Create two drop areas
        const dropAreaI2I = createElement("div", {id: "drop-area-I2I"}, ["drop-area"], "Use as Image2Image source")
        container.appendChild(dropAreaI2I)
        
        const dropAreaMD = createElement("div", {id: "drop-area-MD"}, ["drop-area"], "Extract embedded metadata")
        container.appendChild(dropAreaMD)
        
        const dropAreaCN = createElement("div", {id: "drop-area-CN"}, ["drop-area"], "Use as Controlnet image")
        container.appendChild(dropAreaCN)
        
        // Add event listeners to drop areas
        dropAreaCN.addEventListener("dragenter", function(event) {
            event.preventDefault()
            dropAreaCN.style.backgroundColor = 'darkGreen'
        })
        dropAreaCN.addEventListener("dragleave", function(event) {
            event.preventDefault()
            dropAreaCN.style.backgroundColor = ''
        })
        dropAreaCN.addEventListener("drop", function(event) {
            event.stopPropagation()
            event.preventDefault()
            hideDropAreas()

            getImageFromDropEvent(event, e => controlImagePreview.src=e)
        })

        dropAreaI2I.addEventListener("dragenter", function(event) {
            event.preventDefault()
            dropAreaI2I.style.backgroundColor = 'darkGreen'
        })
        dropAreaI2I.addEventListener("dragleave", function(event) {
            event.preventDefault()
            dropAreaI2I.style.backgroundColor = ''
        })
       
        function getImageFromDropEvent(event, callback) {
            // Find the first image file, uri, or moz-url in the items list
            let imageItem = null
            for (let i = 0; i < event.dataTransfer.items.length; i++) {
                let item = event.dataTransfer.items[i]
                if (item.kind === 'file' && item.type.startsWith('image/')) {
                    imageItem = item;
                    break;
                }
            }
            
            if (!imageItem) {
                // If no file matches, try to find a text/uri-list item
                for (let i = 0; i < event.dataTransfer.items.length; i++) {
                    let item = event.dataTransfer.items[i];
                    if (item.type === 'text/uri-list') {
                        imageItem = item;
                        break;
                    }
                }
            }
            
            if (!imageItem) {
                // If there are no image files or uris, fallback to moz-url
                for (let i = 0; i < event.dataTransfer.items.length; i++) {
                    let item = event.dataTransfer.items[i];
                    if (item.type === 'text/x-moz-url') {
                        imageItem = item;
                        break;
                    }
                }
            }
        
            if (imageItem) {
                if (imageItem.kind === 'file') {
                    // If the item is an image file, handle it as before
                    let file = imageItem.getAsFile();
        
                    // Create a FileReader object to read the dropped file as a data URL
                    let reader = new FileReader();
                    reader.onload = function(e) {
                        callback(e.target.result)
                    };
                    reader.readAsDataURL(file);
                } else {
                    // If the item is a URL, retrieve it and use it to load the image
                    imageItem.getAsString(callback)
                }
            }
        }

        dropAreaI2I.addEventListener("drop", function(event) {
            event.stopPropagation()
            event.preventDefault()
            hideDropAreas()
            
            getImageFromDropEvent(event, e => imageObj.src=e)
        })
        
        dropAreaMD.addEventListener("dragenter", function(event) {
            event.preventDefault()
            dropAreaMD.style.backgroundColor = 'darkGreen'
        })
        dropAreaMD.addEventListener("dragleave", function(event) {
            event.preventDefault()
            dropAreaMD.style.backgroundColor = ''
        })
        
        dropAreaMD.addEventListener("drop", function(event) {
            let items = []
            hideDropAreas()
            if (event?.dataTransfer?.items) { // Use DataTransferItemList interface
                items = Array.from(event.dataTransfer.items)
                items = items.filter(item => item.kind === 'file' && (item.type === 'image/png' || item.type === 'image/jpeg' || item.type === 'image/webp'))
                items = items.map(item => item.getAsFile())
            } else if (event?.dataTransfer?.files) { // Use DataTransfer interface
                items = Array.from(event.dataTransfer.files)
            }
            // check if image has embedded metadata, load task if it does
            if (items[0].type === "image/png") {
                readPNGMetadata(items[0])
            } else if (items[0].type === "image/jpeg" || items[0].type === "image/webp") {
                readJPEGMetadata(items[0]);
            } else {
                console.log("File must be a PNG, WEBP or JPEG image.");
            }
            event.preventDefault()
        })
        
        document.addEventListener("drop", function(event) {
            event.preventDefault()
            hideDropAreas()
        })
        
        document.addEventListener("dragexit", function(event) {
            event.preventDefault()
            hideDropAreas()
        })
    }

    function showDropAreasDnD(event) {
        event.preventDefault()
        // Find the first image file, uri, or moz-url in the items list
        let imageItem = null;
        for (let i = 0; i < event.dataTransfer.items.length; i++) {
            let item = event.dataTransfer.items[i];
            if ((item.kind === 'file' && item.type.startsWith('image/')) || item.type === 'text/uri-list') {
                imageItem = item;
                break;
            } else if (item.type === 'text/x-moz-url') {
                // If there are no image files or uris, fallback to moz-url
                if (!imageItem) {
                    imageItem = item;
                }
            }
        }
    
        if (imageItem) {
            showDropAreas()
        }
    }
    
    function hideDropAreasDnD(event) {
        if (event.fromElement && !document.querySelector('#editor').contains(event.fromElement) && !document.querySelector('#editor').contains(event.fromElement.parentNode.host)) {
            hideDropAreas()
        }
    }

    function showDropAreas() {
        const dropAreas = document.querySelectorAll(".drop-area")
        dropAreas.forEach(function(dropArea) {
            dropArea.style.display = 'inline-block'
        })
    }
    
    function hideDropAreas() {
        const dropAreas = document.querySelectorAll(".drop-area")
        dropAreas.forEach(function(dropArea) {
            dropArea.style.display = 'none'
            dropArea.style.backgroundColor = ''
        })
    }

    const dndContainer = document.getElementById("editor-inputs-init-image")
    createDropAreas(dndContainer)
    document.querySelector('#editor').addEventListener("dragenter", showDropAreasDnD)
    document.querySelector('#editor').addEventListener("dragleave", hideDropAreasDnD)

    /* METADATA EXTRACTION HELPER FUNCTION */
    function clearAllImageTagCards() {
        // clear existing image tag cards
        editorTagsContainer.style.display = 'none'
        editorModifierTagsList.querySelectorAll('.modifier-card').forEach(modifierCard => {
            modifierCard.remove()
        })

        // reset modifier cards state
        document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
            const modifierName = modifierCard.querySelector('.modifier-card-label').innerText
            if (activeTags.map(x => x.name).includes(modifierName)) {
                modifierCard.classList.remove(activeCardClass)
                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
            }
        })
        activeTags = []
        document.dispatchEvent(new Event('refreshImageModifiers')) // notify the 
    }

    /* PNG METADATA EXTRACTION */
    
    function readPNGMetadata(image) {
        const fileReader = new FileReader()
        fileReader.onload = function () {
            extractTextChunks(image).then(function (chunks) {
                let reqBody =  {}
                for (let key in chunks) {
                    reqBody[key] = chunks[key]
                }
                if (Object.keys(reqBody).length !== 0) {
                    if (reqBody["seed"] !== undefined) {
                        let task = { numOutputsTotal: reqBody["num_outputs"], seed: reqBody["seed"] }
                        task['reqBody'] = reqBody
                        clearAllImageTagCards()
                        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
                    }
                }
            }).catch(function (error) {
            console.error(error);
        })}
        fileReader.readAsArrayBuffer(image);
    }

    function extractTextChunks(file) {
        return new Promise(function (resolve, reject) {
            let reader = new FileReader();
            reader.onload = function () {
                let arrayBuffer = reader.result;
                let dataView = new DataView(arrayBuffer);
                
                // Verify that the PNG signature is present
                let signature = new Uint8Array(arrayBuffer, 0, 8);
                if (String.fromCharCode.apply(null, signature) !== "\x89PNG\r\n\x1a\n") {
                    reject(new Error("Invalid PNG file"));
                    return;
                }

                // Iterate through the chunks
                let chunks = {};
                let offset = 8;
                while (offset < arrayBuffer.byteLength) {
                    // Get the length and type of the chunk
                    let length = dataView.getUint32(offset);
                    let type = String.fromCharCode(dataView.getUint8(offset + 4), dataView.getUint8(offset + 5), dataView.getUint8(offset + 6), dataView.getUint8(offset + 7));
                    offset += 8;

                    // Get the data of the chunk
                    let data = new Uint8Array(arrayBuffer, offset, length);
                    offset += length;

                    // Get the CRC of the chunk
                    let crc = dataView.getUint32(offset);
                    offset += 4;

                    // If it's a tEXt chunk, convert the data to a human-readable string
                    if (type === "tEXt") {
                        let nullIndex = data.indexOf(0);
                        let key = String.fromCharCode.apply(null, data.slice(0, nullIndex));
                        let value = String.fromCharCode.apply(null, data.slice(nullIndex + 1));
                        chunks[key] = value;
                    }
                }
                resolve(chunks);
            };
            reader.readAsArrayBuffer(file);
        });
    };

    /* JPEG or WEBP METADATA EXTRACTION */
    function readJPEGMetadata(image) {
        const fileReader = new FileReader()
        fileReader.onload = function (e) {
            ExifReader.load(e.target.result).then(tags => {
                const exifData = String.fromCharCode(...tags['UserComment'].value)
                if (exifData !== undefined) {
                    try {
                        const isUnicode = (exifData.toLowerCase().startsWith('unicode'))
                        let keys = JSON.parse(isUnicode ? decodeUnicode(exifData.slice(8)) : exifData.slice(8))
                        let reqBody =  {}
                        for (let key in keys) {
                            reqBody[key] = keys[key]
                        }
                        let task = { numOutputsTotal: reqBody["num_outputs"], seed: reqBody["seed"] }
                        task['reqBody'] = reqBody
                        clearAllImageTagCards()
                        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
                    } catch (e) {
                        console.error('No valid JSON in EXIF data')
                    }
                }
            })
                                                         
        }
        fileReader.readAsDataURL(image);
    }

    function decodeUnicode(unicodeString) {
        const encoder = new TextEncoder()
        const input = new Uint16Array(encoder.encode(unicodeString))

        let decodedString = ''
        for (let i = 0; i < input.length; i+=2) {
            decodedString += String.fromCharCode(input[i] << 8 | input[i+1])
        }
        
        return decodedString
    }
})()
