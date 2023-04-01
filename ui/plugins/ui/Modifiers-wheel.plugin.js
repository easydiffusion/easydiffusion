(function () { "use strict"
    if (typeof editorModifierTagsList !== 'object') {
        console.error('editorModifierTagsList missing...')
        return
    }

    // observe for changes in tag list
    const observer = new MutationObserver(function (mutations) {
    //    mutations.forEach(function (mutation) {
            if (editorModifierTagsList.childNodes.length > 0) {
                ModifierMouseWheel(editorModifierTagsList)
            }
    //    })
    })
    
    observer.observe(editorModifierTagsList, {
            childList: true
    })

    function ModifierMouseWheel(target) {
        let overlays = document.querySelector('#editor-inputs-tags-list').querySelectorAll('.modifier-card-overlay')
        overlays.forEach (i => {
            i.onwheel = (e) => {
                if (e.ctrlKey == true) {
                    e.preventDefault()
    
                    const delta = Math.sign(event.deltaY)
                    let s = i.parentElement.getElementsByClassName('modifier-card-label')[0].getElementsByTagName("p")[0].innerText
                    let t
                    // find the corresponding tag
                    for (let it = 0; it < overlays.length; it++) {
                        if (i == overlays[it]) {
                            t = activeTags[it].name
                            break
                        }
                    }
                    if (delta < 0) {
                        // wheel scrolling up
                        if (s.substring(0, 1) == '[' && s.substring(s.length-1) == ']') {
                            s = s.substring(1, s.length - 1)
                            t = t.substring(1, t.length - 1)
                        }
                        else
                        {
                            if (s.substring(0, 10) !== '('.repeat(10) && s.substring(s.length-10) !== ')'.repeat(10)) {
                                s = '(' + s + ')'
                                t = '(' + t + ')'
                            }
                        }
                    }
                    else{
                        // wheel scrolling down
                        if (s.substring(0, 1) == '(' && s.substring(s.length-1) == ')') {
                            s = s.substring(1, s.length - 1)
                            t = t.substring(1, t.length - 1)
                        }
                        else
                        {
                            if (s.substring(0, 10) !== '['.repeat(10) && s.substring(s.length-10) !== ']'.repeat(10)) {
                                s = '[' + s + ']'
                                t = '[' + t + ']'
                            }
                        }
                    }
                    i.parentElement.getElementsByClassName('modifier-card-label')[0].getElementsByTagName("p")[0].innerText = s
                    // update activeTags
                    for (let it = 0; it < overlays.length; it++) {
                        if (i == overlays[it]) {
                            activeTags[it].name = t
                            break
                        }
                    }
                    document.dispatchEvent(new Event('refreshImageModifiers'))
                }
            }
        })
    }
})()
