(function () {
    "use strict"

    let autoScroll = document.querySelector("#auto_scroll")
    
    // save/restore the toggle state
    autoScroll.addEventListener('click', (e) => {
        localStorage.setItem('auto_scroll', autoScroll.checked)
    })
    autoScroll.checked = localStorage.getItem('auto_scroll') == "true"

    // observe for changes in the preview pane
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.target.className == 'img-batch') {
                Autoscroll(mutation.target)
            }
        })
    })
    
    observer.observe(document.getElementById('preview'), {
            childList: true,
            subtree: true
    })

    function Autoscroll(target) {
        if (autoScroll.checked && target !== null) {
            const img = target.querySelector('img')
            img.addEventListener('load', function() {
                img.closest('.imageTaskContainer').scrollIntoView()
            }, { once: true })
        }
    }
})()
