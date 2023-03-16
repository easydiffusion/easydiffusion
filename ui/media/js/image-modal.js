"use strict"

const imageModal = (function() {
    const zoomElem = createElement(
        'i',
        undefined,
        ['fa-solid', 'tertiaryButton'],
    )

    const closeElem = createElement(
        'i',
        undefined,
        ['fa-solid', 'fa-xmark', 'tertiaryButton'],
    )

    const menuBarElem = createElement('div', undefined, 'menu-bar', [zoomElem, closeElem])

    const imageContainer = createElement('div', undefined, 'image-wrapper')

    const backdrop = createElement('div', undefined, 'backdrop')

    const modalContainer = createElement('div', undefined, 'content', [menuBarElem, imageContainer])

    const modalElem = createElement(
        'div',
        { id: 'viewFullSizeImgModal' },
        ['popup'],
        [backdrop, modalContainer],
    )
    document.body.appendChild(modalElem)

    const setZoomLevel = (value) => {
        const img = imageContainer.querySelector('img')

        if (value) {
            zoomElem.classList.remove('fa-magnifying-glass-plus')
            zoomElem.classList.add('fa-magnifying-glass-minus')
            if (img) {
                img.classList.remove('natural-zoom')

                let zoomLevel = typeof value === 'number' ? value : img.dataset.zoomLevel
                if (!zoomLevel) {
                    zoomLevel = 100
                }

                img.dataset.zoomLevel = zoomLevel
                img.width = img.naturalWidth * (+zoomLevel / 100)
                img.height = img.naturalHeight * (+zoomLevel / 100)
            }
        } else {
            zoomElem.classList.remove('fa-magnifying-glass-minus')
            zoomElem.classList.add('fa-magnifying-glass-plus')
            if (img) {
                img.classList.add('natural-zoom')
                img.removeAttribute('width')
                img.removeAttribute('height')
            }
        }
    }

    zoomElem.addEventListener(
        'click',
        () => setZoomLevel(imageContainer.querySelector('img')?.classList?.contains('natural-zoom')),
    )

    const close = () => {
        imageContainer.innerHTML = ''
        modalElem.classList.remove('active')
        document.body.style.overflow = 'initial'
    }

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modalElem.classList.contains('active')) {
            close()
        }
    })
    window.addEventListener('click', (e) => {
        if (modalElem.classList.contains('active')) {
            if (e.target === backdrop || e.target === closeElem) {
                close()
            }

            e.stopPropagation()
            e.stopImmediatePropagation()
            e.preventDefault()
        }
    })

    return (optionsFactory) => {
        const options = typeof optionsFactory === 'function' ? optionsFactory() : optionsFactory
        const src = typeof options === 'string' ? options : options.src

        // TODO center it if < window size
        const imgElem = createElement('img', { src }, 'natural-zoom')
        imageContainer.appendChild(imgElem)
        modalElem.classList.add('active')
        document.body.style.overflow = 'hidden'
        setZoomLevel(false)
    }
})()
