"use strict"

/**
 * @typedef {object} ImageModalRequest
 * @property {string} src
 * @property {ImageModalRequest | () => ImageModalRequest | undefined} previous
 * @property {ImageModalRequest | () => ImageModalRequest | undefined} next
 */

/**
 * @type {(() => (string | ImageModalRequest) | string | ImageModalRequest) => {}}
 */
const imageModal = (function() {
    const backElem = createElement("i", undefined, ["fa-solid", "fa-arrow-left", "tertiaryButton"])

    const forwardElem = createElement("i", undefined, ["fa-solid", "fa-arrow-right", "tertiaryButton"])

    const zoomElem = createElement("i", undefined, ["fa-solid", "tertiaryButton"])

    const closeElem = createElement("i", undefined, ["fa-solid", "fa-xmark", "tertiaryButton"])

    const menuBarElem = createElement("div", undefined, "menu-bar", [backElem, forwardElem, zoomElem, closeElem])

    const imageContainer = createElement("div", undefined, "image-wrapper")

    const backdrop = createElement("div", undefined, "backdrop")

    const modalContainer = createElement("div", undefined, "content", [menuBarElem, imageContainer])

    const modalElem = createElement("div", { id: "viewFullSizeImgModal" }, ["popup"], [backdrop, modalContainer])
    document.body.appendChild(modalElem)

    const setZoomLevel = (value) => {
        const img = imageContainer.querySelector("img")

        if (value) {
            zoomElem.classList.remove("fa-magnifying-glass-plus")
            zoomElem.classList.add("fa-magnifying-glass-minus")
            if (img) {
                img.classList.remove("natural-zoom")

                let zoomLevel = typeof value === "number" ? value : img.dataset.zoomLevel
                if (!zoomLevel) {
                    zoomLevel = 100
                }

                img.dataset.zoomLevel = zoomLevel
                img.width = img.naturalWidth * (+zoomLevel / 100)
                img.height = img.naturalHeight * (+zoomLevel / 100)
            }
        } else {
            zoomElem.classList.remove("fa-magnifying-glass-minus")
            zoomElem.classList.add("fa-magnifying-glass-plus")
            if (img) {
                img.classList.add("natural-zoom")
                img.removeAttribute("width")
                img.removeAttribute("height")
            }
        }
    }

    zoomElem.addEventListener("click", () =>
        setZoomLevel(imageContainer.querySelector("img")?.classList?.contains("natural-zoom"))
    )

    const initialState = () => ({
        previous: undefined,
        next: undefined,

        start: {
            x: 0,
            y: 0,
        },

        scroll: {
            x: 0,
            y: 0,
        },
    })

    const state = initialState()

    // Allow grabbing the image to scroll
    const stopGrabbing = (e) => {
        if(imageContainer.classList.contains("grabbing")) {
            imageContainer.classList.remove("grabbing")
            e?.preventDefault()
            console.log(`stopGrabbing()`, e)
        }
    }

    const addImageGrabbing = (image) => {
        image?.addEventListener('mousedown', (e) => {
            if (!image.classList.contains("natural-zoom")) {
                e.stopPropagation()
                e.stopImmediatePropagation()
                e.preventDefault()

                imageContainer.classList.add("grabbing")
                state.start.x = e.pageX - imageContainer.offsetLeft
                state.scroll.x = imageContainer.scrollLeft
                state.start.y = e.pageY - imageContainer.offsetTop
                state.scroll.y = imageContainer.scrollTop
            }
        })

        image?.addEventListener('mouseup', stopGrabbing)
        image?.addEventListener('mouseleave', stopGrabbing)
        image?.addEventListener('mousemove', (e) => {
            if(imageContainer.classList.contains("grabbing")) {
                e.stopPropagation()
                e.stopImmediatePropagation()
                e.preventDefault()

                // Might need to increase this multiplier based on the image size to window size ratio
                // The default 1:1 is pretty slow
                const multiplier = 1.0

                const deltaX = e.pageX - imageContainer.offsetLeft - state.start.x
                imageContainer.scrollLeft = state.scroll.x - (deltaX * multiplier)
                const deltaY = e.pageY - imageContainer.offsetTop - state.start.y
                imageContainer.scrollTop = state.scroll.y - (deltaY * multiplier)
            }
        })
    }

    const clear = () => {
        imageContainer.innerHTML = ""

        Object.entries(initialState()).forEach(([key, value]) => state[key] = value)

        stopGrabbing()
    }

    const close = () => {
        clear()
        modalElem.classList.remove("active")
        document.body.style.overflow = "initial"
    }

    /**
     * @param {() => (string | ImageModalRequest) | string | ImageModalRequest} optionsFactory
     */
    function init(optionsFactory) {
        if (!optionsFactory) {
            close()
            return
        }

        clear()

        const options = typeof optionsFactory === "function" ? optionsFactory() : optionsFactory
        const src = typeof options === "string" ? options : options.src

        const imgElem = createElement("img", { src }, "natural-zoom")
        addImageGrabbing(imgElem)
        imageContainer.appendChild(imgElem)
        modalElem.classList.add("active")
        document.body.style.overflow = "hidden"
        setZoomLevel(false)

        if (typeof options === "object" && options.previous) {
            state.previous = options.previous
            backElem.style.display = "unset"
        } else {
            backElem.style.display = "none"
        }

        if (typeof options === "object" && options.next) {
            state.next = options.next
            forwardElem.style.display = "unset"
        } else {
            forwardElem.style.display = "none"
        }
    }

    const back = () => {
        if (state.previous) {
            init(state.previous)
        } else {
            backElem.style.display = "none"
        }
    }

    const forward = () => {
        if (state.next) {
            init(state.next)
        } else {
            forwardElem.style.display = "none"
        }
    }

    window.addEventListener("keydown", (e) => {
        if (modalElem.classList.contains("active")) {
            switch (e.key) {
                case "Escape":
                    close()
                    break
                case "ArrowLeft":
                    back()
                    break
                case "ArrowRight":
                    forward()
                    break
            }
        }
    })
    window.addEventListener("click", (e) => {
        if (modalElem.classList.contains("active")) {
            if (e.target === backdrop || e.target === closeElem) {
                close()
            }

            e.stopPropagation()
            e.stopImmediatePropagation()
            e.preventDefault()
        }
    })

    backElem.addEventListener("click", back)

    forwardElem.addEventListener("click", forward)

    /**
     * @param {() => (string | ImageModalRequest) | string | ImageModalRequest} optionsFactory
     */
    return (optionsFactory) => init(optionsFactory)
})()
