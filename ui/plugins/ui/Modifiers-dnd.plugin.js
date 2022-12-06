(function () { "use strict"
    if (typeof editorModifierTagsList !== 'object') {
        console.error('editorModifierTagsList missing...')
        return
    }

    const styleSheet = document.createElement("style");
    styleSheet.textContent = `
        .modifier-card-tiny.drag-sort-active {
            background: transparent;
            border: 2px dashed white;
            opacity:0.2;
        }
    `;
    document.head.appendChild(styleSheet);

    // observe for changes in tag list
    const observer = new MutationObserver(function (mutations) {
    //    mutations.forEach(function (mutation) {
            if (editorModifierTagsList.childNodes.length > 0) {
                ModifierDragAndDrop(editorModifierTagsList)
            }
    //    })
    })
    
    observer.observe(editorModifierTagsList, {
            childList: true
    })

    let current
    function ModifierDragAndDrop(target) {
        let overlays = document.querySelector('#editor-inputs-tags-list').querySelectorAll('.modifier-card-overlay')
        overlays.forEach (i => {
            i.parentElement.draggable = true;
            
            i.parentElement.ondragstart = (e) => {
                current = i
                i.parentElement.getElementsByClassName('modifier-card-image-overlay')[0].innerText = ''
                i.parentElement.draggable = true
                i.parentElement.classList.add('drag-sort-active')
                for(let item of document.querySelector('#editor-inputs-tags-list').getElementsByClassName('modifier-card-image-overlay')) {
                    if (item.parentElement.parentElement.getElementsByClassName('modifier-card-overlay')[0] != current) {
                        item.parentElement.parentElement.getElementsByClassName('modifier-card-image-overlay')[0].style.opacity = 0
                        if(item.parentElement.getElementsByClassName('modifier-card-image').length > 0) {
                            item.parentElement.getElementsByClassName('modifier-card-image')[0].style.filter = 'none'
                        }
                        item.parentElement.parentElement.style.transform = 'none'
                        item.parentElement.parentElement.style.boxShadow = 'none'
                    }
                    item.innerText = ''
                }
            }
            
            i.ondragenter = (e) => {
                e.preventDefault()
                if (i != current) {
                    let currentPos = 0, droppedPos = 0;
                    for (let it = 0; it < overlays.length; it++) {
                        if (current == overlays[it]) { currentPos = it; }
                        if (i == overlays[it]) { droppedPos = it; }
                    }

                    if (i.parentElement != current.parentElement) {
                        let currentPos = 0, droppedPos = 0
                        for (let it = 0; it < overlays.length; it++) {
                            if (current == overlays[it]) { currentPos = it }
                            if (i == overlays[it]) { droppedPos = it }
                        }
                        if (currentPos < droppedPos) {
                            current = i.parentElement.parentNode.insertBefore(current.parentElement, i.parentElement.nextSibling).getElementsByClassName('modifier-card-overlay')[0]
                        } else {
                            current = i.parentElement.parentNode.insertBefore(current.parentElement, i.parentElement).getElementsByClassName('modifier-card-overlay')[0]
                        }
                        // update activeTags
                        const tag = activeTags.splice(currentPos, 1)
                        activeTags.splice(droppedPos, 0, tag[0])
                    }
                }
            };

            i.ondragover = (e) => {
                e.preventDefault()
            }
            
            i.parentElement.ondragend = (e) => {
                i.parentElement.classList.remove('drag-sort-active')
                for(let item of document.querySelector('#editor-inputs-tags-list').getElementsByClassName('modifier-card-image-overlay')) {
                    item.style.opacity = ''
                    item.innerText = '-'
                }
            }
        })
    }
})()
