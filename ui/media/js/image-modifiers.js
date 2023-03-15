let activeTags = []
let modifiers = []
let customModifiersGroupElement = undefined
let customModifiersInitialContent

let editorModifierEntries = document.querySelector('#editor-modifiers-entries')
let editorModifierTagsList = document.querySelector('#editor-inputs-tags-list')
let editorTagsContainer = document.querySelector('#editor-inputs-tags-container')
let modifierCardSizeSlider = document.querySelector('#modifier-card-size-slider')
let previewImageField = document.querySelector('#preview-image')
let modifierSettingsBtn = document.querySelector('#modifier-settings-btn')
let modifierSettingsOverlay = document.querySelector('#modifier-settings-config')
let customModifiersTextBox = document.querySelector('#custom-modifiers-input')
let customModifierEntriesToolbar = document.querySelector('#editor-modifiers-entries-toolbar')

const modifierThumbnailPath = 'media/modifier-thumbnails'
const activeCardClass = 'modifier-card-active'
const CUSTOM_MODIFIERS_KEY = "customModifiers"

function createModifierCard(name, previews, removeBy) {
    const modifierCard = document.createElement('div')
    let style = previewImageField.value
    let styleIndex = (style=='portrait') ? 0 : 1

    modifierCard.className = 'modifier-card'
    modifierCard.innerHTML = `
    <div class="modifier-card-overlay"></div>
    <div class="modifier-card-image-container">
        <div class="modifier-card-image-overlay">+</div>
        <p class="modifier-card-error-label"></p>
        <img onerror="this.remove()" alt="Modifier Image" class="modifier-card-image">
    </div>
    <div class="modifier-card-container">
        <div class="modifier-card-label"><p></p></div>
    </div>`

    const image = modifierCard.querySelector('.modifier-card-image')
    const errorText =  modifierCard.querySelector('.modifier-card-error-label')
    const label = modifierCard.querySelector('.modifier-card-label')

    errorText.innerText = 'No Image'

    if (typeof previews == 'object') {
        image.src = previews[styleIndex]; // portrait
        image.setAttribute('preview-type', style)
    } else {
        image.remove()
    }

    const maxLabelLength = 30
    const cardLabel = removeBy ? name.replace('by ', '') : name

    if(cardLabel.length <= maxLabelLength) {
        label.querySelector('p').innerText = cardLabel
    } else {
        const tooltipText = document.createElement('span')
        tooltipText.className = 'tooltip-text'
        tooltipText.innerText = name

        label.classList.add('tooltip')
        label.appendChild(tooltipText)

        label.querySelector('p').innerText = cardLabel.substring(0, maxLabelLength) + '...'
    }
    label.querySelector('p').dataset.fullName = name // preserve the full name

    return modifierCard
}

function createModifierGroup(modifierGroup, initiallyExpanded, removeBy) {
    const title = modifierGroup.category
    const modifiers = modifierGroup.modifiers

    const titleEl = document.createElement('h5')
    titleEl.className = 'collapsible'
    titleEl.innerText = title

    const modifiersEl = document.createElement('div')
    modifiersEl.classList.add('collapsible-content', 'editor-modifiers-leaf')

    if (initiallyExpanded === true) {
        titleEl.className += ' active'
    }

    modifiers.forEach(modObj => {
        const modifierName = modObj.modifier
        const modifierPreviews = modObj?.previews?.map(preview => `${IMAGE_REGEX.test(preview.image) ? preview.image : modifierThumbnailPath + '/' + preview.path}`)

        const modifierCard = createModifierCard(modifierName, modifierPreviews, removeBy)

        if(typeof modifierCard == 'object') {
            modifiersEl.appendChild(modifierCard)
            const trimmedName = trimModifiers(modifierName)

            modifierCard.addEventListener('click', () => {
                if (activeTags.map(x => trimModifiers(x.name)).includes(trimmedName)) {
                    // remove modifier from active array
                    activeTags = activeTags.filter(x => trimModifiers(x.name) != trimmedName)
                    toggleCardState(trimmedName, false)
                } else {
                    // add modifier to active array
                    activeTags.push({
                        'name': modifierName,
                        'element': modifierCard.cloneNode(true),
                        'originElement': modifierCard,
                        'previews': modifierPreviews
                    })
                    toggleCardState(trimmedName, true)
                }

                refreshTagsList()
                document.dispatchEvent(new Event('refreshImageModifiers'))
            })
        }
    })

    let brk = document.createElement('br')
    brk.style.clear = 'both'
    modifiersEl.appendChild(brk)

    let e = document.createElement('div')
    e.className = 'modifier-category'
    e.appendChild(titleEl)
    e.appendChild(modifiersEl)

    editorModifierEntries.insertBefore(e, customModifierEntriesToolbar.nextSibling)

    return e
}

function trimModifiers(tag) {
    return tag.replace(/^\(+|\)+$/g, '').replace(/^\[+|\]+$/g, '')
}

async function loadModifiers() {
    try {
        let res = await fetch('/get/modifiers')
        if (res.status === 200) {
            res = await res.json()

            modifiers = res; // update global variable

            res.reverse()

            res.forEach((modifierGroup, idx) => {
                createModifierGroup(modifierGroup, idx === res.length - 1, modifierGroup === 'Artist' ? true : false) // only remove "By " for artists
            })

            createCollapsibles(editorModifierEntries)
        }
    } catch (e) {
        console.error('error fetching modifiers', e)
    }

    loadCustomModifiers()
    resizeModifierCards(modifierCardSizeSlider.value)
    document.dispatchEvent(new Event('loadImageModifiers'))
}

function refreshModifiersState(newTags) {
    // clear existing modifiers
    document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
        const modifierName = modifierCard.querySelector('.modifier-card-label p').dataset.fullName // pick the full modifier name
        if (activeTags.map(x => x.name).includes(modifierName)) {
            modifierCard.classList.remove(activeCardClass)
            modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
        }
    })
    activeTags = []

    // set new modifiers
    newTags.forEach(tag => {
        let found = false
        document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
            const modifierName = modifierCard.querySelector('.modifier-card-label p').dataset.fullName
            const shortModifierName = modifierCard.querySelector('.modifier-card-label p').innerText
            if (trimModifiers(tag) == trimModifiers(modifierName)) {
                // add modifier to active array
                if (!activeTags.map(x => x.name).includes(tag)) { // only add each tag once even if several custom modifier cards share the same tag
                    const imageModifierCard = modifierCard.cloneNode(true)
                    imageModifierCard.querySelector('.modifier-card-label p').innerText = tag.replace(modifierName, shortModifierName)
                    activeTags.push({
                        'name': tag,
                        'element': imageModifierCard,
                        'originElement': modifierCard
                    })
                }
                modifierCard.classList.add(activeCardClass)
                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '-'
                found = true
            }
        })
        if (found == false) { // custom tag went missing, create one here
            let modifierCard = createModifierCard(tag, undefined, false) // create a modifier card for the missing tag, no image
            
            modifierCard.addEventListener('click', () => {
                if (activeTags.map(x => x.name).includes(tag)) {
                    // remove modifier from active array
                    activeTags = activeTags.filter(x => x.name != tag)
                    modifierCard.classList.remove(activeCardClass)

                    modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
                }
                refreshTagsList()
            })

            activeTags.push({
                'name': tag,
                'element': modifierCard,
                'originElement': undefined  // no origin element for missing tags
            })
        }
    })
    refreshTagsList()
}

function refreshInactiveTags(inactiveTags) {
    // update inactive tags
    if (inactiveTags !== undefined && inactiveTags.length > 0) {
        activeTags.forEach (tag => {
            if (inactiveTags.find(element => element === tag.name) !== undefined) {
                tag.inactive = true
            }
        })
    }
    
    // update cards
    let overlays = document.querySelector('#editor-inputs-tags-list').querySelectorAll('.modifier-card-overlay')
    overlays.forEach (i => {
        let modifierName = i.parentElement.getElementsByClassName('modifier-card-label')[0].getElementsByTagName("p")[0].innerText
        if (inactiveTags.find(element => element === modifierName) !== undefined) {
            i.parentElement.classList.add('modifier-toggle-inactive')
        }
    })
}

function refreshTagsList() {
    editorModifierTagsList.innerHTML = ''

    if (activeTags.length == 0) {
        editorTagsContainer.style.display = 'none'
        return
    } else {
        editorTagsContainer.style.display = 'block'
    }

    activeTags.forEach((tag, index) => {
        tag.element.querySelector('.modifier-card-image-overlay').innerText = '-'
        tag.element.classList.add('modifier-card-tiny')

        editorModifierTagsList.appendChild(tag.element)

        tag.element.addEventListener('click', () => {
            let idx = activeTags.findIndex(o => { return o.name === tag.name })            

            if (idx !== -1) {
                toggleCardState(activeTags[idx].name, false)

                activeTags.splice(idx, 1)
                refreshTagsList()
            }
            document.dispatchEvent(new Event('refreshImageModifiers'))
        })
    })

    let brk = document.createElement('br')
    brk.style.clear = 'both'
    editorModifierTagsList.appendChild(brk)
    document.dispatchEvent(new Event('refreshImageModifiers')) // notify plugins that the image tags have been refreshed 
}

function toggleCardState(modifierName, makeActive) {
    document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(card => {
        const name = card.querySelector('.modifier-card-label').innerText
        if (   trimModifiers(modifierName) == trimModifiers(name)
            || trimModifiers(modifierName) == 'by ' + trimModifiers(name)) {
            if(makeActive) {
                card.classList.add(activeCardClass)
                card.querySelector('.modifier-card-image-overlay').innerText = '-'
            }
            else{
                card.classList.remove(activeCardClass)
                card.querySelector('.modifier-card-image-overlay').innerText = '+'
            }
        }
    })
}

function changePreviewImages(val) {
    const previewImages = document.querySelectorAll('.modifier-card-image-container img')

    let previewArr = []

    modifiers.map(x => x.modifiers).forEach(x => previewArr.push(...x.map(m => m.previews)))
    
    previewArr = previewArr.map(x => {
        let obj = {}

        x.forEach(preview => {
            obj[preview.name] = preview.path
        })
        
        return obj
    })

    previewImages.forEach(previewImage => {
        const currentPreviewType = previewImage.getAttribute('preview-type')
        const relativePreviewPath = previewImage.src.split(modifierThumbnailPath + '/').pop()

        const previews = previewArr.find(preview => relativePreviewPath == preview[currentPreviewType])

        if(typeof previews == 'object') {
            let preview = null

            if (val == 'portrait') {
                preview = previews.portrait
            }
            else if (val == 'landscape') {
                preview = previews.landscape
            }

            if(preview != null) {
                previewImage.src = `${modifierThumbnailPath}/${preview}`
                previewImage.setAttribute('preview-type', val)
            }
        }
    })
}

function resizeModifierCards(val) {
    const cardSizePrefix = 'modifier-card-size_'
    const modifierCardClass = 'modifier-card'

    const modifierCards = document.querySelectorAll(`.${modifierCardClass}`)
    const cardSize = n => `${cardSizePrefix}${n}`

    modifierCards.forEach(card => {
        // remove existing size classes
        const classes = card.className.split(' ').filter(c => !c.startsWith(cardSizePrefix))
        card.className = classes.join(' ').trim()

        if(val != 0) {
            card.classList.add(cardSize(val))
        }
    })
}

modifierCardSizeSlider.onchange = () => resizeModifierCards(modifierCardSizeSlider.value)
previewImageField.onchange = () => changePreviewImages(previewImageField.value)

modifierSettingsBtn.addEventListener('click', function(e) {
    modifierSettingsOverlay.classList.add("active")
    customModifiersTextBox.setSelectionRange(0, 0)
    customModifiersTextBox.focus()
    customModifiersInitialContent = customModifiersTextBox.value // preserve the initial content
    e.stopPropagation()
})

modifierSettingsOverlay.addEventListener('keydown', function(e) {
    switch (e.key) {
        case "Escape": // Escape to cancel
            customModifiersTextBox.value = customModifiersInitialContent // undo the changes
            modifierSettingsOverlay.classList.remove("active")
            e.stopPropagation()
            break
        case "Enter":
            if (e.ctrlKey) { // Ctrl+Enter to confirm
                modifierSettingsOverlay.classList.remove("active")
                e.stopPropagation()
                break
            }
    }
})

function saveCustomModifiers() {
    localStorage.setItem(CUSTOM_MODIFIERS_KEY, customModifiersTextBox.value.trim())

    loadCustomModifiers()
}

function loadCustomModifiers() {
    PLUGINS['MODIFIERS_LOAD'].forEach(fn=>fn.loader.call())
}

customModifiersTextBox.addEventListener('change', saveCustomModifiers)
