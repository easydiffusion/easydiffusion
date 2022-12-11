/*
    Right-click an image to display the buttons (rather than just hovering over it)
*/
(function() {
    "use strict"

    var styleSheet = document.createElement("style")
    styleSheet.textContent = `
        .imgItemInfo {
            display: none;
        }
    `
    document.head.appendChild(styleSheet)

    // listen for right click
    let contextMenu
    let clickedImage
    window.addEventListener('contextmenu', (event) => {
        const clickedElem = document.elementFromPoint(event.clientX, event.clientY)
        if (clickedElem !== null) {
            clickedImage = clickedElem.closest(".imgContainer")
            if (clickedImage !== null) {
                event.preventDefault()
                contextMenu = clickedImage.parentNode.querySelector('.imgItemInfo')
                if (contextMenu !==  null) {
                    if (contextMenu.style.display == 'flex') {
                        contextMenu.style.display = 'none'
                    }
                    else
                    {
                        contextMenu.style.display =  'flex'
                        clickedImage.addEventListener('mouseleave', hideImageContextMenu, {capture: true})
                    }
                }
            }
        }
    })

    // listen for click
    window.addEventListener('click', (event) => {
        const clickedElem = document.elementFromPoint(event.clientX, event.clientY)
        if (clickedElem !== null) {
            clickedImage = clickedElem.closest(".imgContainer")
            if (clickedImage !== null) {
                event.preventDefault()
                contextMenu = clickedImage.parentNode.querySelector('.imgItemInfo')
                if (contextMenu !==  null) {
                    if (contextMenu.style.display == 'flex') {
                        hideImageContextMenu()
                    }
                }
            }
        }
    })

    // hide the menu as applicable
    function hideImageContextMenu(event) {
        let eventElement
        let imageRect
        if (event !== undefined && clickedImage !== null) {
            eventElement = document.elementFromPoint(event.clientX, event.clientY)
            imageRect = clickedImage.getBoundingClientRect()
        }
        
        // reducing the image rectangle by 2 pixels to compensate for decimal coordinates rounding errors
        if (event == undefined || (clickedImage !== null && event.clientX < imageRect.left+2 || event.clientX > imageRect.right-2 || event.clientY < imageRect.top+2 || event.clientY > imageRect.bottom-2)) {
            contextMenu.style.display = 'none'
            clickedImage.removeEventListener('mouseleave', hideImageContextMenu, {capture: true})
        }
    }
})()
