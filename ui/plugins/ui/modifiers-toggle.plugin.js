(function () {
    "use strict"

    var styleSheet = document.createElement("style");
    styleSheet.textContent = `
        .modifier-card-tiny.modifier-toggle-inactive {
            background: transparent;
            border: 2px dashed red;
            opacity:0.2;
        }
    `;
    document.head.appendChild(styleSheet);

    // observe for changes in tag list
    var observer = new MutationObserver(function (mutations) {
    //    mutations.forEach(function (mutation) {
            if (editorModifierTagsList.childNodes.length > 0) {
                ModifierToggle()
            }
    //    })
    })

    observer.observe(editorModifierTagsList, {
            childList: true
    })

    function ModifierToggle() {
        let overlays = document.querySelector('#editor-inputs-tags-list').querySelectorAll('.modifier-card-overlay')
        overlays.forEach (i => {
            i.oncontextmenu = (e) => {
                e.preventDefault()

                if (i.parentElement.classList.contains('modifier-toggle-inactive')) {
                    i.parentElement.classList.remove('modifier-toggle-inactive')
                }
                else
                {
                    i.parentElement.classList.add('modifier-toggle-inactive')
                }
                // refresh activeTags
                let modifierName = i.parentElement.getElementsByClassName('modifier-card-label')[0].getElementsByTagName("p")[0].dataset.fullName
                activeTags = activeTags.map(obj => {
                    if (trimModifiers(obj.name) === trimModifiers(modifierName)) {
                        return {...obj, inactive: (obj.element.classList.contains('modifier-toggle-inactive'))};
                    }
                    
                    return obj;
                });
                document.dispatchEvent(new Event('refreshImageModifiers'))
            }
        })
    }
})()
