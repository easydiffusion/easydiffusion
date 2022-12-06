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
    autoScrollControl.innerHTML = `<input id="auto_scroll" name="auto_scroll" type="checkbox">
                            <label for="auto_scroll">Scroll to generated image</label>`
    autoScrollControl.className = "auto-scroll"
    clearAllPreviewsBtn.parentNode.insertBefore(autoScrollControl, clearAllPreviewsBtn.nextSibling)
    prettifyInputs(document);
    let autoScroll = document.querySelector("#auto_scroll")

    /**
     * the use of initSettings() in the autoscroll plugin seems to be breaking the models dropdown and the save-to-disk folder field
     * in the settings tab. They're both blank, because they're being re-initialized. Their earlier values came from the API call,
     * but those values aren't stored in localStorage, since they aren't user-specified.
     * So when initSettings() is called a second time, it overwrites the values with an empty string.
     *
     * We could either rework how new components can register themselves to be auto-saved, without having to call initSettings() again.
     * Or we could move the autoscroll code into the main code, and include it in the list of fields in auto-save.js
     */
    // SETTINGS_IDS_LIST.push("auto_scroll")
    // initSettings()

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
            target.parentElement.parentElement.parentElement.scrollIntoView();
        }
    }
})()
