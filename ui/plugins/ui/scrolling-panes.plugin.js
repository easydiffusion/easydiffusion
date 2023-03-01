/*
    - Allows editor and preview panes to scroll independently.
    - adds a toggle to hide the editor pane (click again or move the mouse to the left edge to show it)
*/
(function () {
    "use strict"

    // default timeout settings
    const HIDE_TOOLBAR_TIMER = 4000
    const SHOW_EDITOR_PANE_TIMER = 150
    const HIDE_EDITOR_PANE_TIMER = 250

    var styleSheet = document.createElement("style")
    styleSheet.textContent = `
        @media (min-width: 700px) {
            body {
                overflow-y: hidden;
            }
            
            #top-nav {
                position: fixed;
                background: var(--background-color4);
                display: flex;
                width: 100%;
                z-index: 10;
            }

            #editor {
                z-index: 1;
                width: min-content;
                min-width: 380pt;
                position: fixed;
                overflow-x: hidden;
                overflow-y: auto;
                top: 0;
                bottom: 0;
                left: 0;
            }
            /*
            #paneToggleContainer:hover + #editor {
                transition: 0.25s;
            }
            */
            
            #preview {
                position:  fixed;
                overflow-y: auto;
                top: 0;
                bottom: 0;
                left: 0;
                right:0;
                padding: 0 16px 0 16px;
                outline: none;
            }
            /*
            #paneToggle:hover ~ #preview {
                transition: 0.25s;
            }
            */
    
            #preview-tools {
                background: var(--background-color1);
                position: sticky;
                top: -100px; /* hide the toolbar by default */
                transition: 0.25s;
                z-index: 1;
                padding: 10px 10px 10px 10px;
                /*
                -webkit-mask-image: linear-gradient(to bottom, black 0%, black 90%, transparent 100%);
                mask-image: linear-gradient(to bottom, black 0%, black 90%, transparent 100%);
                */
                opacity: 90%;
            }
        
            #editor-modifiers {
                overflow-y: initial;
                overflow-x: initial;
            }
            
            .image_preview_container {
                padding: 6px;
            }

            /* pane toggle */
            #paneToggleContainer {
                width: 8px;
                top: 0;
                left: 0;
                background: var(--background-color1);
                margin: 0;
                border-radius: 5px;
                position: fixed;
                z-index: 1000;
            }
            
            #paneToggle {
                width: 8px;
                height: 100px;
                left: 0;
                background: var(--background-color2);
                margin: 0;
                border-radius: 5px;
                position: relative;
                top: 50%;
                -ms-transform: translateY(-50%);
                transform: translateY(-50%);
            }
            
            .arrow-right {
                width: 0; 
                height: 0; 
                border-top: 8px solid transparent;
                border-bottom: 8px solid transparent;
                border-left: 8px solid var(--accent-color);
                
                margin: 0;
                position: absolute;
                top: 50%;
                -ms-transform: translateY(-50%);
                transform: translateY(-50%);
            }
            
            .arrow-left {
                width: 0; 
                height: 0; 
                border-top: 8px solid transparent;
                border-bottom: 8px solid transparent; 
                border-right:8px solid var(--accent-color);
                
                margin: 0;
                position: absolute;
                top: 50%;
                -ms-transform: translateY(-50%);
                transform: translateY(-50%);
            }
        }

        @media (max-width: 700px) {
            #hidden-top-nav {
                display: none;
            }
        }

        /* STICKY FOOTER */
        #preview {
            display: flex;
            flex-direction: column;
        }
        #preview-content {
            flex: 1 0 auto;
        }
        #footer {
            padding-left: 4px;
            flex-shrink: 0;
        }

        /* SCROLLBARS */
        :root {
            --scrollbar-width: 12px;
            --scrollbar-radius: 10px;
        }
        
        .scrollbar-preview::-webkit-scrollbar {
            width: var(--scrollbar-width);
        }
        
        .scrollbar-preview::-webkit-scrollbar-track {
            box-shadow: inset 0 0 5px var(--input-border-color);
            border-radius: var(--input-border-radius);
        }
        
        .scrollbar-preview::-webkit-scrollbar-thumb {
            background: var(--background-color2);
            border-radius: var(--scrollbar-radius);
        }
        
        ::-webkit-scrollbar {
            width: var(--scrollbar-width);
        }
        
        ::-webkit-scrollbar-track {
            box-shadow: inset 0 0 5px var(--input-border-color);
            border-radius: var(--input-border-radius);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--background-color2);
            border-radius: var(--scrollbar-radius);
        }
        
        .scrollbar-preview::-moz-scrollbar {
            width: var(--scrollbar-width);
        }
        
        .scrollbar-preview::-moz-scrollbar-track {
            box-shadow: inset 0 0 5px var(--input-border-color);
            border-radius: var(--input-border-radius);
        }
        
        .scrollbar-preview::-moz-scrollbar-thumb {
            background: var(--background-color2);
            border-radius: var(--scrollbar-radius);
        }
        
        ::-moz-scrollbar {
            width: var(--scrollbar-width);
        }
        
        ::-moz-scrollbar-track {
            box-shadow: inset 0 0 5px var(--input-border-color);
            border-radius: var(--input-border-radius);
        }
        
        ::-moz-scrollbar-thumb {
            background: var(--background-color2);
            border-radius: var(--scrollbar-radius);
        }
    `
    document.head.appendChild(styleSheet)
    
    const topNavbar  = document.querySelector('#top-nav')
    const tabContainer = document.querySelector('#tab-container')
    const footerPane = document.querySelector('#footer')
    document.querySelector('#preview').appendChild(footerPane)

    // create a placeholder header hidden behind the fixed header to push regular tab's content down in place
    const hiddenHeader = document.createElement('div')
    topNavbar.parentNode.insertBefore(hiddenHeader, topNavbar.nextSibling)
    hiddenHeader.id = 'hidden-top-nav'
    hiddenHeader.position = 'absolute'
    hiddenHeader.style.width = '100%'

    // create the arrow zone
    const editorContainer = document.querySelector('#editor')
    editorContainer.insertAdjacentHTML('beforebegin', `
        <div  id="paneToggleContainer">
            <div id="paneToggle">
                <div id="editorToggleArrow" class="arrow-left"></div>
            </div>
        </div>
    `)

    // update editor style and size
    const editorToggleContainer = document.querySelector('#paneToggleContainer')
    const editorToggleButton = document.querySelector('#paneToggle')
    const editorToggleArrow = document.querySelector('#editorToggleArrow')
    const editorModifiers = document.querySelector('#editor-modifiers')
    editorContainer.classList.add('pane-toggle')
    imagePreview.classList.add('scrollbar-preview')
    updatePreviewSize()

    /* EDITOR PANE TOGGLE */
    // restore the editor pane toggled state
    let loadingScrollingPane = true
    let timerShow, timerHide, toolbarTimerHide
    let editorPaneOpen = localStorage.getItem('editor_pane_open') == null ? true : localStorage.getItem('editor_pane_open') == 'true'
    if (!editorPaneOpen) {
        hideEditorPane()
    }

    function showEditorPane(showPreview) {
        editorContainer.style.transition = '0.2s'
        if (editorPaneOpen) {
            editorToggleArrow.classList.remove('arrow-right')
            editorToggleArrow.classList.add('arrow-left')
        }
        editorContainer.style.left = '0px'
        if (showPreview) {
            updatePreviewSize(editorContainer.offsetWidth)
        }
    }

    function showEditorPaneWithTimer() {
        clearTimeout(timerHide)
        timerShow = setTimeout(function() {
            showEditorPane(false)
        }, SHOW_EDITOR_PANE_TIMER);
    }

    function cancelShowTimer() {
        clearTimeout(timerShow)
    }

    function cancelHideTimer() {
        clearTimeout(timerHide)        
    }

    function hideEditorPane(hideEditorPane) {
        if (!editorPaneOpen) {
            timerHide = setTimeout(function() {
                if (!loadingScrollingPane) {
                    editorContainer.style.transition = '0.25s'
                }
                editorToggleArrow.classList.remove('arrow-left')
                editorToggleArrow.classList.add('arrow-right')
                editorContainer.style.left = -(editorContainer.offsetWidth) + 'px'
                updatePreviewSize(0)
            }, HIDE_EDITOR_PANE_TIMER)
        }
    }

    function toggleEditorPane() {
        cancelShowTimer()
        if (editorPaneOpen) {
            editorPaneOpen = false
            hideEditorPane()
        }
        else
        {
            editorPaneOpen = true
            showEditorPane(true)
        }
        localStorage.setItem('editor_pane_open', editorPaneOpen)
    }
    editorToggleButton.addEventListener("click", toggleEditorPane)
    editorToggleContainer.addEventListener("mouseenter", showEditorPaneWithTimer)
    editorToggleContainer.addEventListener("mouseleave", cancelShowTimer)
    editorContainer.addEventListener("mouseenter", cancelHideTimer)
    editorContainer.addEventListener("mouseleave", hideEditorPane)
    editorToggleContainer.addEventListener("dragenter", showEditorPaneWithTimer)
    imagePreview.addEventListener("dragenter", hideEditorPane)
    previewTools.addEventListener("mouseenter", cancelHideToolbarTimer)
    previewTools.addEventListener("mouseleave", showToolbarWithTimer)

    /* PREVIEW TOOLBAR */
    function showToolbarWithTimer() {
        showToolbar()
        clearTimeout(toolbarTimerHide)
        toolbarTimerHide = setTimeout(function() {
            hideToolbar()
        }, HIDE_TOOLBAR_TIMER);
    }

    function cancelHideToolbarTimer() {
        clearTimeout(toolbarTimerHide)        
    }

    function hideToolbar() {
        //if (previewTools.style.top !== -previewTools.offsetHeight + 'px') {
            clearTimeout(toolbarTimerHide)
            previewTools.style.top = -previewTools.offsetHeight + 'px'
        //}
    }
    
    function showToolbar() {
        if (previewTools.style.top !== '0') {
            previewTools.style.top = '0'
        }
    }

    // update toolbar visibility
    let yPos
    let scrollTop
    let lastScrollTop
    let sampling
    
    function touchStart() {
        yPos = undefined
        scrollTop = undefined
        lastScrollTop = undefined
        sampling = 0
    }
    imagePreview.addEventListener("touchstart", touchStart)
    //imagePreview.addEventListener("touchend", touchEnd)
    //imagePreview.addEventListener("touchcancel", touchEnd)
    //imagePreview.addEventListener("touchleave", touchEnd)
    
    function updateToolbarVisibility(event) {
        if (window.innerWidth < 700) {
            return
        }
       
        // handle touch events
        if (event.changedTouches !== undefined && event.changedTouches.length > 0) {
            // sample the events to mitigate noise caused by finger position on the screen
            sampling += 1
            if (sampling % 5 == 0) {
                yPos = -event.changedTouches[0].pageY
            }
            
            // adjust the preview-tools visibility
            scrollTop = yPos
            if (scrollTop > lastScrollTop) {
                hideToolbar()
            }
            else if (scrollTop < lastScrollTop) {
                showToolbarWithTimer()
            }
            lastScrollTop = scrollTop
        }
        
        // handle wheel events
        if (event.deltaY !== undefined) {
            if (Math.sign(event.deltaY) < 0) {
                // wheel scrolling up
                showToolbarWithTimer()
            }
            else
            {
                // wheel scrolling down
                hideToolbar()
            }
        }

        // handle keyboard events
        const KEYBOARD_UP = ['ArrowUp', 'Home', 'PageUp']
        const KEYBOARD_DOWN = ['ArrowDown', 'End', 'PageDown']
        if (KEYBOARD_UP.find(elem => elem == event.key) !== undefined) {
            // key up
            showToolbarWithTimer()
        }
        else if (KEYBOARD_DOWN.find(elem => elem == event.key) !== undefined) {
            // key down
            hideToolbar()
        }
    }
    imagePreview.addEventListener("touchmove", updateToolbarVisibility)
    imagePreview.addEventListener("wheel", updateToolbarVisibility)
    imagePreview.addEventListener("keydown", updateToolbarVisibility) // arrow up/down, page up/down, home/end
    imagePreview.tabIndex = 0 // required to receive 'keydown' events on a DIV

    /* EDITOR AND PREVIEW PANE LAYOUT */
    // update preview pane size and position
    function updatePreviewSize(leftPosition) {
        if (window.innerWidth < 700) {
            return
        }

        // adjust the topnav placeholder's height
        hiddenHeader.style.height = topNavbar.offsetHeight + 'px'

        // resize the editor and preview panes as needed
        topNavbar.style.marginTop = '' // fix for the proper menubar Chrome extension that changes the margin-top
        editorContainer.style.top = (topNavbar.offsetTop + topNavbar.offsetHeight) + 'px'
        imagePreview.style.top = (topNavbar.offsetTop + topNavbar.offsetHeight) + 'px'
        imagePreview.style.left = (typeof leftPosition == 'number' ? leftPosition : (editorContainer.offsetLeft + editorContainer.offsetWidth)) + 'px'
        // reposition the toggle container and button
        editorToggleContainer.style.top = editorContainer.style.top
        editorToggleContainer.style.bottom = '0px'
    };
    window.addEventListener("resize", updatePreviewSize)

    document.addEventListener("tabClick", (e) => {
        // update the body's overflow-y depending on the selected tab
        if (e.detail.name == 'main') {
            document.body.style.overflowY = 'hidden'
        }
        else
        {
            document.body.style.overflowY = 'auto'
        }
    })

    function observeResize(element, callbackFunction) {
        const resizeObserver = new ResizeObserver(callbackFunction)
        resizeObserver.observe(element, { box : 'border-box' })
    }
    observeResize(editorContainer, updatePreviewSize)
    observeResize(topNavbar, updatePreviewSize)
    loadingScrollingPane = false
})()
