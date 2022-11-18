let activeTags = []
let modifiers = []
let customModifiersGroupElement = undefined

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

function createModifierCard(name, previews) {
    const modifierCard = document.createElement('div')
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
        image.src = previews[0]; // portrait
        image.setAttribute('preview-type', 'portrait')
    } else {
        image.remove()
    }

    const maxLabelLength = 30
    const nameWithoutBy = name.replace('by ', '')

    if(nameWithoutBy.length <= maxLabelLength) {
        label.querySelector('p').innerText = nameWithoutBy
    } else {
        const tooltipText = document.createElement('span')
        tooltipText.className = 'tooltip-text'
        tooltipText.innerText = name

        label.classList.add('tooltip')
        label.appendChild(tooltipText)

        label.querySelector('p').innerText = nameWithoutBy.substring(0, maxLabelLength) + '...'
    }

    return modifierCard
}

function createModifierGroup(modifierGroup, initiallyExpanded) {
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
        const modifierPreviews = modObj?.previews?.map(preview => `${modifierThumbnailPath}/${preview.path}`)

        const modifierCard = createModifierCard(modifierName, modifierPreviews)

        if(typeof modifierCard == 'object') {
            modifiersEl.appendChild(modifierCard)

            modifierCard.addEventListener('click', () => {
                if (activeTags.map(x => x.name).includes(modifierName)) {
                    // remove modifier from active array
                    activeTags = activeTags.filter(x => x.name != modifierName)
                    modifierCard.classList.remove(activeCardClass)

                    modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
                } else {
                    // add modifier to active array
                    activeTags.push({
                        'name': modifierName,
                        'element': modifierCard.cloneNode(true),
                        'originElement': modifierCard,
                        'previews': modifierPreviews
                    })

                    modifierCard.classList.add(activeCardClass)

                    modifierCard.querySelector('.modifier-card-image-overlay').innerText = '-'
                }

                refreshTagsList()
            })
        }
    })

    let brk = document.createElement('br')
    brk.style.clear = 'both'
    modifiersEl.appendChild(brk)

    let e = document.createElement('div')
    e.appendChild(titleEl)
    e.appendChild(modifiersEl)

    editorModifierEntries.insertBefore(e, customModifierEntriesToolbar.nextSibling)

    return e
}

async function loadModifiers() {
    try {
        let res = await fetch('/get/modifiers')
        if (res.status === 200) {
            res = await res.json()

            modifiers = res; // update global variable

            res.reverse()

            res.forEach((modifierGroup, idx) => {
                createModifierGroup(modifierGroup, idx === res.length - 1)
            })

            createCollapsibles(editorModifierEntries)
        }
    } catch (e) {
        console.log('error fetching modifiers', e)
    }

    loadCustomModifiers()
}

function refreshModifiersState(newTags) {
    // clear existing modifiers
    document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
        const modifierName = modifierCard.querySelector('.modifier-card-label').innerText
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
            const modifierName = modifierCard.querySelector('.modifier-card-label').innerText
            if (tag == modifierName) {
                // add modifier to active array
                activeTags.push({
                    'name': modifierName,
                    'element': modifierCard.cloneNode(true),
                    'originElement': modifierCard
                })
                modifierCard.classList.add(activeCardClass)
                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '-'
                found = true
            }
        })
        if (found == false) { // custom tag went missing, create one here
            let modifierCard = createModifierCard(tag, undefined) // create a modifier card for the missing tag, no image
            
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
            let idx = activeTags.indexOf(tag)

            if (idx !== -1 && activeTags[idx].originElement !== undefined) {
                activeTags[idx].originElement.classList.remove(activeCardClass)
                activeTags[idx].originElement.querySelector('.modifier-card-image-overlay').innerText = '+'

                activeTags.splice(idx, 1)
                refreshTagsList()
            }
        })
    })

    let brk = document.createElement('br')
    brk.style.clear = 'both'
    editorModifierTagsList.appendChild(brk)
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
    e.stopPropagation()
})

function saveCustomModifiers() {
    localStorage.setItem(CUSTOM_MODIFIERS_KEY, customModifiersTextBox.value.trim())

    loadCustomModifiers()
}

function loadCustomModifiers() {
    let customModifiers = localStorage.getItem(CUSTOM_MODIFIERS_KEY, '')
    customModifiersTextBox.value = customModifiers

    if (customModifiersGroupElement !== undefined) {
        customModifiersGroupElement.remove()
    }

    if (customModifiers && customModifiers.trim() !== '') {
        customModifiers = customModifiers.split('\n')
        customModifiers = customModifiers.filter(m => m.trim() !== '')
        customModifiers = customModifiers.map(function(m) {
            return {
                "modifier": m
            }
        })

        let customGroup = {
            'category': 'Custom Modifiers',
            'modifiers': customModifiers
        }

        customModifiersGroupElement = createModifierGroup(customGroup, true)

        createCollapsibles(customModifiersGroupElement)
    }
}

customModifiersTextBox.addEventListener('change', saveCustomModifiers)
