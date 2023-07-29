let activeTags = []
let modifiers = []
let customModifiersGroupElement = undefined
let customModifiersInitialContent = ""
let modifierPanelFreezed = false

let modifiersMainContainer = document.querySelector("#editor-modifiers")
let modifierDropdown = document.querySelector("#image-modifier-dropdown")
let editorModifiersContainer = document.querySelector("#editor-modifiers")
let editorModifierEntries = document.querySelector("#editor-modifiers-entries")
let editorModifierTagsList = document.querySelector("#editor-inputs-tags-list")
let editorTagsContainer = document.querySelector("#editor-inputs-tags-container")
let modifierCardSizeSlider = document.querySelector("#modifier-card-size-slider")
let previewImageField = document.querySelector("#preview-image")
let modifierSettingsBtn = document.querySelector("#modifier-settings-btn")
let modifiersContainerSizeBtn = document.querySelector("#modifiers-container-size-btn")
let modifiersCloseBtn = document.querySelector("#modifiers-close-button")
let modifiersCollapsiblesBtn = document.querySelector("#modifiers-action-collapsibles-btn")
let modifierSettingsDialog = document.querySelector("#modifier-settings-config")
let customModifiersTextBox = document.querySelector("#custom-modifiers-input")
let customModifierEntriesToolbar = document.querySelector("#editor-modifiers-subheader")
let modifierSettingsCloseBtn = document.querySelector("#modifier-settings-close-button")

const modifierThumbnailPath = "media/modifier-thumbnails"
const activeCardClass = "modifier-card-active"
const CUSTOM_MODIFIERS_KEY = "customModifiers"

function createModifierCard(name, previews, removeBy) {
    let cardPreviewImageType = previewImageField.value

    const modifierCard = document.createElement("div")
    modifierCard.className = "modifier-card"
    modifierCard.innerHTML = `
    <div class="modifier-card-overlay"></div>
    <div class="modifier-card-image-container">
        <div class="modifier-card-image-overlay">+</div>
        <p class="modifier-card-error-label">No Image</p>
        <img onerror="this.remove()" alt="Modifier Image" class="modifier-card-image">
    </div>
    <div class="modifier-card-container">
        <div class="modifier-card-label">
            <span class="long-label hidden"></span>
            <p class="regular-label"></p>
        </div>
    </div>`

    const image = modifierCard.querySelector(".modifier-card-image")
    const longLabel = modifierCard.querySelector(".modifier-card-label span.long-label")
    const regularLabel = modifierCard.querySelector(".modifier-card-label p.regular-label")

    if (typeof previews == "object") {
        image.src = previews[cardPreviewImageType == "portrait" ? 0 : 1] // 0 index is portrait, 1 landscape
        image.setAttribute("preview-type", cardPreviewImageType)
    } else {
        image.remove()
    }

    const maxLabelLength = 30
    const cardLabel = removeBy ? name.replace("by ", "") : name

    function getFormattedLabel(length) {
        if (cardLabel?.length <= length) {
            return cardLabel
        } else {
            return cardLabel.substring(0, length) + "..."
        }
    }

    modifierCard.dataset.fullName = name // preserve the full name
    regularLabel.dataset.fullName = name // preserve the full name, legacy support for older plugins
    
    longLabel.innerText = getFormattedLabel(maxLabelLength * 2)
    regularLabel.innerText = getFormattedLabel(maxLabelLength)

    if (cardLabel.length > maxLabelLength) {
        modifierCard.classList.add("support-long-label")

        if (cardLabel.length > maxLabelLength * 2) {
            modifierCard.title = `"${name}"`
        }
    }
    
    return modifierCard
}

function createModifierGroup(modifierGroup, isInitiallyOpen, removeBy) {
    const title = modifierGroup.category
    const modifiers = modifierGroup.modifiers

    const titleEl = document.createElement("h5")
    titleEl.className = "collapsible"
    titleEl.innerText = title

    const modifiersEl = document.createElement("div")
    modifiersEl.classList.add("collapsible-content", "editor-modifiers-leaf")

    if (isInitiallyOpen === true) {
        titleEl.classList.add("active")
    }

    modifiers.forEach((modObj) => {
        const modifierName = modObj.modifier
        const modifierPreviews = modObj?.previews?.map(
            (preview) =>
                `${IMAGE_REGEX.test(preview.image) ? preview.image : modifierThumbnailPath + "/" + preview.path}`
        )

        const modifierCard = createModifierCard(modifierName, modifierPreviews, removeBy)

        if (typeof modifierCard == "object") {
            modifiersEl.appendChild(modifierCard)
            const trimmedName = trimModifiers(modifierName)

            modifierCard.addEventListener("click", () => {
                if (activeTags.map((x) => trimModifiers(x.name)).includes(trimmedName)) {
                    // remove modifier from active array
                    activeTags = activeTags.filter((x) => trimModifiers(x.name) != trimmedName)
                    toggleCardState(trimmedName, false)
                } else {
                    // add modifier to active array
                    activeTags.push({
                        name: modifierName,
                        element: modifierCard.cloneNode(true),
                        originElement: modifierCard,
                        previews: modifierPreviews,
                    })
                    toggleCardState(trimmedName, true)
                }

                refreshTagsList()
                document.dispatchEvent(new Event("refreshImageModifiers"))
            })
        }
    })

    let brk = document.createElement("br")
    brk.style.clear = "both"
    modifiersEl.appendChild(brk)

    let e = document.createElement("div")
    e.className = "modifier-category"
    e.appendChild(titleEl)
    e.appendChild(modifiersEl)

    editorModifierEntries.prepend(e)

    return e
}

function trimModifiers(tag) {
    // Remove trailing '-' and/or '+'
    tag = tag.replace(/[-+]+$/, "")
    // Remove parentheses at beginning and end
    return tag.replace(/^[(]+|[\s)]+$/g, "")
}

async function loadModifiers() {
    try {
        let res = await fetch("/get/modifiers")
        if (res.status === 200) {
            res = await res.json()

            modifiers = res // update global variable

            res.reverse()

            res.forEach((modifierGroup, idx) => {
                const isInitiallyOpen = false // idx === res.length - 1
                const removeBy = modifierGroup === "Artist" ? true : false // only remove "By " for artists

                createModifierGroup(modifierGroup, isInitiallyOpen, removeBy) 
            })

            createCollapsibles(editorModifierEntries)
        }
    } catch (e) {
        console.error("error fetching modifiers", e)
    }

    loadCustomModifiers()
    resizeModifierCards(modifierCardSizeSlider.value)
    document.dispatchEvent(new Event("loadImageModifiers"))
}

function refreshModifiersState(newTags, inactiveTags) {
    // clear existing modifiers
    document
        .querySelector("#editor-modifiers")
        .querySelectorAll(".modifier-card")
        .forEach((modifierCard) => {
            const modifierName = modifierCard.dataset.fullName // pick the full modifier name
            if (activeTags.map((x) => x.name).includes(modifierName)) {
                modifierCard.classList.remove(activeCardClass)
                modifierCard.querySelector(".modifier-card-image-overlay").innerText = "+"
            }
        })
    activeTags = []

    // set new modifiers
    newTags.forEach((tag) => {
        let found = false
        document
            .querySelector("#editor-modifiers")
            .querySelectorAll(".modifier-card")
            .forEach((modifierCard) => {
                const modifierName = modifierCard.dataset.fullName
                const shortModifierName = modifierCard.querySelector(".modifier-card-label p").innerText

                if (trimModifiers(tag) == trimModifiers(modifierName)) {
                    // add modifier to active array
                    if (!activeTags.map((x) => x.name).includes(tag)) {
                        // only add each tag once even if several custom modifier cards share the same tag
                        const imageModifierCard = modifierCard.cloneNode(true)
                        imageModifierCard.querySelector(".modifier-card-label p").innerText = tag.replace(
                            modifierName,
                            shortModifierName
                        )
                        activeTags.push({
                            name: tag,
                            element: imageModifierCard,
                            originElement: modifierCard,
                        })
                    }
                    modifierCard.classList.add(activeCardClass)
                    modifierCard.querySelector(".modifier-card-image-overlay").innerText = "-"
                    found = true
                }
            })
        if (found == false) {
            // custom tag went missing, create one here
            let modifierCard = createModifierCard(tag, undefined, false) // create a modifier card for the missing tag, no image

            modifierCard.addEventListener("click", () => {
                if (activeTags.map((x) => x.name).includes(tag)) {
                    // remove modifier from active array
                    activeTags = activeTags.filter((x) => x.name != tag)
                    modifierCard.classList.remove(activeCardClass)

                    modifierCard.querySelector(".modifier-card-image-overlay").innerText = "+"
                }
                refreshTagsList()
            })

            activeTags.push({
                name: tag,
                element: modifierCard,
                originElement: undefined, // no origin element for missing tags
            })
        }
    })
    refreshTagsList(inactiveTags)
}

function refreshInactiveTags(inactiveTags) {
    // update inactive tags
    if (inactiveTags !== undefined && inactiveTags.length > 0) {
        activeTags.forEach((tag) => {
            if (inactiveTags.find((element) => element === tag.name) !== undefined) {
                tag.inactive = true
            }
        })
    }

    // update cards
    let overlays = editorModifierTagsList.querySelectorAll(".modifier-card-overlay")
    overlays.forEach((i) => {
        let modifierName = i.parentElement.dataset.fullName

        if (inactiveTags?.find((element) => trimModifiers(element) === modifierName) !== undefined) {
            i.parentElement.classList.add("modifier-toggle-inactive")
        }
    })
}

function refreshTagsList(inactiveTags) {
    editorModifierTagsList.innerHTML = ""

    if (activeTags.length == 0) {
        editorTagsContainer.style.display = "none"
        return
    } else {
        editorTagsContainer.style.display = "block"
    }

    if(activeTags.length > 15) {
        editorModifierTagsList.style["overflow-y"] = "auto"
    } else {
        editorModifierTagsList.style["overflow-y"] = "unset"
    }

    activeTags.forEach((tag, index) => {
        tag.element.querySelector(".modifier-card-image-overlay").innerText = "-"
        tag.element.classList.add("modifier-card-tiny")

        editorModifierTagsList.appendChild(tag.element)

        tag.element.addEventListener("click", () => {
            let idx = activeTags.findIndex((o) => {
                return o.name === tag.name
            })

            if (idx !== -1) {
                toggleCardState(activeTags[idx].name, false)

                activeTags.splice(idx, 1)
                refreshTagsList()
            }
            document.dispatchEvent(new Event("refreshImageModifiers"))
        })
    })

    let brk = document.createElement("br")
    brk.style.clear = "both"

    editorModifierTagsList.appendChild(brk)

    refreshInactiveTags(inactiveTags)

    document.dispatchEvent(new Event("refreshImageModifiers")) // notify plugins that the image tags have been refreshed
}

function toggleCardState(modifierName, makeActive) {
    const cards = [...document.querySelectorAll("#editor-modifiers .modifier-card")]
        .filter(cardElem => trimModifiers(cardElem.dataset.fullName) == trimModifiers(modifierName))

    const cardExists = typeof cards == "object" && cards?.length > 0

    if (cardExists) {
        const card = cards[0]
    
        if (makeActive) {
            card.classList.add(activeCardClass)
            card.querySelector(".modifier-card-image-overlay").innerText = "-"
        } else {
            card.classList.remove(activeCardClass)
            card.querySelector(".modifier-card-image-overlay").innerText = "+"
        }
    }
}

function changePreviewImages(val) {
    const previewImages = document.querySelectorAll(".modifier-card-image-container img")

    const previewArr = modifiers.flatMap((x) => x.modifiers.map((m) => m.previews))
        .map((x) => x.reduce((obj, preview) => {
            obj[preview.name] = preview.path

            return obj
        }, {}))

    previewImages.forEach((previewImage) => {
        const currentPreviewType = previewImage.getAttribute("preview-type")
        const relativePreviewPath = previewImage.src.split(modifierThumbnailPath + "/").pop()

        const previews = previewArr.find((preview) => relativePreviewPath == preview[currentPreviewType])

        if (typeof previews == "object") {
            let preview = null

            if (val == "portrait") {
                preview = previews.portrait
            } else if (val == "landscape") {
                preview = previews.landscape
            }

            if (preview) {
                previewImage.src = `${modifierThumbnailPath}/${preview}`
                previewImage.setAttribute("preview-type", val)
            }
        }
    })
}

function resizeModifierCards(val) {
    const cardSizePrefix = "modifier-card-size_"
    const modifierCardClass = "modifier-card"

    const modifierCards = document.querySelectorAll(`.${modifierCardClass}`)
    const cardSize = (n) => `${cardSizePrefix}${n}`

    modifierCards.forEach((card) => {
        // remove existing size classes
        const classes = card.className.split(" ").filter((c) => !c.startsWith(cardSizePrefix))
        card.className = classes.join(" ").trim()

        if (val != 0) {
            card.classList.add(cardSize(val))
        }
    })
}

function saveCustomModifiers() {
    localStorage.setItem(CUSTOM_MODIFIERS_KEY, customModifiersTextBox.value.trim())

    loadCustomModifiers()
}

function loadCustomModifiers() {
    PLUGINS["MODIFIERS_LOAD"].forEach((fn) => fn.loader.call())
}

function showModifierContainer() {
    document.addEventListener("mousedown", checkIfClickedOutsideDropdownElem)

    modifierDropdown.dataset.active = true
    editorModifiersContainer.classList.add("active")
}

function hideModifierContainer() {
    document.removeEventListener("click", checkIfClickedOutsideDropdownElem)

    modifierDropdown.dataset.active = false
    editorModifiersContainer.classList.remove("active")
}

function checkIfClickedOutsideDropdownElem(e) {
    const clickedElement = e.target

    const clickedInsideSpecificElems = [modifierDropdown, editorModifiersContainer, modifierSettingsDialog].some((div) => 
        div && (div.contains(clickedElement) || div === clickedElement))

    if (!clickedInsideSpecificElems && !modifierPanelFreezed) {
        hideModifierContainer()
    }
}

function collapseAllModifierCategory() {
    collapseAll(".modifier-category .collapsible")
}

function expandAllModifierCategory() {
    expandAll(".modifier-category .collapsible")
}

customModifiersTextBox.addEventListener("change", saveCustomModifiers)

modifierCardSizeSlider.onchange = () => resizeModifierCards(modifierCardSizeSlider.value)
previewImageField.onchange = () => changePreviewImages(previewImageField.value)

modifierSettingsDialog.addEventListener("keydown", function(e) {
    switch (e.key) {
        case "Escape": // Escape to cancel
            customModifiersTextBox.value = customModifiersInitialContent // undo the changes
            modifierSettingsDialog.close()
            e.stopPropagation()
            break
        case "Enter":
            if (e.ctrlKey) {
                // Ctrl+Enter to confirm
                modifierSettingsDialog.close()
                e.stopPropagation()
                break
            }
    }
})

modifierDropdown.addEventListener("click", e => {
    const targetElem = e.target
    const isDropdownActive = targetElem.dataset.active == "true" ? true : false

    if (!isDropdownActive) 
        showModifierContainer()
    else
        hideModifierContainer()
})

let collapsiblesBtnState = false

modifiersCollapsiblesBtn.addEventListener("click", (e) => {
    const btnElem = modifiersCollapsiblesBtn

    const collapseText = "Collapse Categories"
    const expandText = "Expand Categories"

    const collapseIconClasses = ["fa-solid", "fa-square-minus"]
    const expandIconClasses = ["fa-solid", "fa-square-plus"]

    const iconElem = btnElem.querySelector(".modifiers-action-icon")
    const textElem = btnElem.querySelector(".modifiers-action-text")

    if (collapsiblesBtnState) {
        collapseAllModifierCategory()

        collapsiblesBtnState = false

        collapseIconClasses.forEach((c) => iconElem.classList.remove(c))
        expandIconClasses.forEach((c) => iconElem.classList.add(c))

        textElem.innerText = expandText
    } else {
        expandAllModifierCategory()

        collapsiblesBtnState = true

        expandIconClasses.forEach((c) => iconElem.classList.remove(c))
        collapseIconClasses.forEach((c) => iconElem.classList.add(c))

        textElem.innerText = collapseText
    }
})

let containerSizeBtnState = false

modifiersContainerSizeBtn.addEventListener("click", (e) => {
    const btnElem = modifiersContainerSizeBtn

    const maximizeIconClasses = ["fa-solid", "fa-expand"]
    const revertIconClasses = ["fa-solid", "fa-compress"]

    modifiersMainContainer.classList.toggle("modifiers-maximized")

    if(containerSizeBtnState) {
        revertIconClasses.forEach((c) => btnElem.classList.remove(c))
        maximizeIconClasses.forEach((c) => btnElem.classList.add(c))

        containerSizeBtnState = false
    } else {
        maximizeIconClasses.forEach((c) => btnElem.classList.remove(c))
        revertIconClasses.forEach((c) => btnElem.classList.add(c))
        
        containerSizeBtnState = true
    }
})

modifierSettingsBtn.addEventListener("click", (e) => {
    modifierSettingsDialog.showModal()
    customModifiersTextBox.setSelectionRange(0, 0)
    customModifiersTextBox.focus()
    customModifiersInitialContent = customModifiersTextBox.value // preserve the initial content
    e.stopPropagation()
})

modifiersCloseBtn.addEventListener("click", (e) => {
    hideModifierContainer()
})

// prevents the modifier panel closing at the same time as the settings overlay
new MutationObserver(() => {
    const isActive = modifierSettingsDialog.open

    if (!isActive) {
        modifierPanelFreezed = true

        setTimeout(() => modifierPanelFreezed = false, 25)
    }
}).observe(modifierSettingsDialog, { attributes: true })

modifierSettingsCloseBtn.addEventListener("click", (e) => {
    modifierSettingsDialog.close()    
})

modalDialogCloseOnBackdropClick(modifierSettingsDialog)
makeDialogDraggable(modifierSettingsDialog)

