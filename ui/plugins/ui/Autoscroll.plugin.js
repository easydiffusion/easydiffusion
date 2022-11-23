(function () {
    "use strict"

    var styleSheet = document.createElement("style");
    styleSheet.textContent = `
        .auto-scroll {
            float: right;
        }
    `;
    document.head.appendChild(styleSheet);
    
    const autoScrollControl = document.createElement('div');
    autoScrollControl.innerHTML = `<input id="auto_scroll" name="auto_scroll" type="checkbox" checked>
                            <label for="auto_scroll">Auto-scroll</label>`
    autoScrollControl.className = "auto-scroll"
    previewTools.appendChild(autoScrollControl)
    prettifyInputs(document);
    let autoScroll = document.querySelector("#auto_scroll")

    SETTINGS_IDS_LIST.push("auto_scroll")
    initSettings()

    // observe for changes in tag list
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            console.log(mutation.target.class)
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
            target.parentElement.parentElement.parentElement.scrollIntoView();
        }
    }
})()
