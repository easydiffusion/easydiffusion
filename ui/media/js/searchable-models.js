"use strict"

let modelsCache
let modelsOptions

/*
*** SEARCHABLE MODELS ***
Creates searchable dropdowns for SD, VAE, or HN models.
Also adds a reload models button (placed next to SD models, reloads everything including VAE and HN models).
More reload buttons may be added at strategic UI locations as needed.
Merely calling getModels() makes all the magic happen behind the scene to refresh the dropdowns.

HOW TO CREATE A MODEL DROPDOWN:
1) Create an input element. Make sure to add a data-path property, as this is how model dropdowns are identified in auto-save.js.
<input id="stable_diffusion_model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" />

2) Just declare one of these for your own dropdown (remember to change the element id, e.g. #stable_diffusion_models to your own input's id).
let stableDiffusionModelField = new ModelDropdown(document.querySelector('#stable_diffusion_model'), 'stable-diffusion')
let vaeModelField = new ModelDropdown(document.querySelector('#vae_model'), 'vae', 'None')
let hypernetworkModelField = new ModelDropdown(document.querySelector('#hypernetwork_model'), 'hypernetwork', 'None')

3) Model dropdowns will be refreshed automatically when the reload models button is invoked.
*/
class ModelDropdown
{
    modelFilter //= document.querySelector("#model-filter")
    modelFilterArrow //= document.querySelector("#model-filter-arrow")
    modelList //= document.querySelector("#model-list")
    modelResult //= document.querySelector("#model-result")
    modelNoResult //= document.querySelector("#model-no-result")
    
    currentSelection //= { elem: undefined, value: '', path: ''}
    highlightedModelEntry //= undefined
    activeModel //= undefined

    inputModels //= undefined
    modelKey //= undefined
    flatModelList //= []
    noneEntry //= ''
    modelFilterInitialized //= undefined

    /* MIMIC A REGULAR INPUT FIELD */
    get parentElement() {
        return this.modelFilter.parentElement
    }
    get parentNode() {
        return this.modelFilter.parentNode
    }
    get value() {
        return this.modelFilter.dataset.path
    }
    set value(path) {
        this.modelFilter.dataset.path = path
        this.selectEntry(path)
    }
    get disabled() {
        return this.modelFilter.disabled
    }
    set disabled(state) {
        this.modelFilter.disabled = state
        if (this.modelFilterArrow) {
            this.modelFilterArrow.style.color = state ? 'dimgray' : ''
        }
    }
    get modelElements() {
        return this.modelList.querySelectorAll('.model-file')
    }
    addEventListener(type, listener, options) {
        return this.modelFilter.addEventListener(type, listener, options)
    }
    dispatchEvent(event) {
        return this.modelFilter.dispatchEvent(event)
    }
    appendChild(option) {
        // do nothing
    }

    // remember 'this' - http://blog.niftysnippets.org/2008/04/you-must-remember-this.html
    bind(f, obj) {
        return function() {
            return f.apply(obj, arguments)
        }
    }

    /* SEARCHABLE INPUT */    
    constructor (input, modelKey, noneEntry = '') {
        this.modelFilter = input
        this.noneEntry = noneEntry
        this.modelKey = modelKey

        if (modelsOptions !== undefined) { // reuse models from cache (only useful for plugins, which are loaded after models)
            this.inputModels = modelsOptions[this.modelKey]
            this.populateModels()
        }
        document.addEventListener("refreshModels", this.bind(function(e) {
            // reload the models
            this.inputModels = modelsOptions[this.modelKey]
            this.populateModels()
        }, this))
    }

    saveCurrentSelection(elem, value, path) {
        this.currentSelection.elem = elem
        this.currentSelection.value = value
        this.currentSelection.path = path
        this.modelFilter.dataset.path = path
        this.modelFilter.value = value
        this.modelFilter.dispatchEvent(new Event('change'))
    }
    
    processClick(e) {
        e.preventDefault()
        if (e.srcElement.classList.contains('model-file')) {
            this.saveCurrentSelection(e.srcElement, e.srcElement.innerText, e.srcElement.dataset.path)
            this.hideModelList()
            this.modelFilter.focus()
            this.modelFilter.select()
        }
    }

    getPreviousVisibleSibling(elem) {
        const modelElements = Array.from(this.modelElements)
        const index = modelElements.indexOf(elem)
        if (index <= 0) {
            return undefined
        }

        return modelElements.slice(0, index).reverse().find(e => e.style.display === 'list-item')
    }

    getLastVisibleChild(elem) {
        let lastElementChild = elem.lastElementChild
        if (lastElementChild.style.display == 'list-item') return lastElementChild
        return this.getPreviousVisibleSibling(lastElementChild)
    }
    
    getNextVisibleSibling(elem) {
        const modelElements = Array.from(this.modelElements)
        const index = modelElements.indexOf(elem)
        return modelElements.slice(index + 1).find(e => e.style.display === 'list-item')
    }
    
    getFirstVisibleChild(elem) {
        let firstElementChild = elem.firstElementChild
        if (firstElementChild.style.display == 'list-item') return firstElementChild
        return this.getNextVisibleSibling(firstElementChild)
    }
    
    selectModelEntry(elem) {
        if (elem) {
            if (this.highlightedModelEntry !== undefined) {
                this.highlightedModelEntry.classList.remove('selected')
            }
            this.saveCurrentSelection(elem, elem.innerText, elem.dataset.path)
            elem.classList.add('selected')
            elem.scrollIntoView({block: 'nearest'})
            this.highlightedModelEntry = elem
        }
    }
    
    selectPreviousFile() {
        const elem = this.getPreviousVisibleSibling(this.highlightedModelEntry)
        if (elem) {
            this.selectModelEntry(elem)
        }
        else
        {
            //this.highlightedModelEntry.parentElement.parentElement.scrollIntoView({block: 'nearest'})
            this.highlightedModelEntry.closest('.model-list').scrollTop = 0
        }
        this.modelFilter.select()
    }
    
    selectNextFile() {
        this.selectModelEntry(this.getNextVisibleSibling(this.highlightedModelEntry))
        this.modelFilter.select()
    }
    
    selectFirstFile() {
        this.selectModelEntry(this.modelList.querySelector('.model-file'))
        this.highlightedModelEntry.scrollIntoView({block: 'nearest'})
        this.modelFilter.select()
    }
    
    selectLastFile() {
        const elems = this.modelList.querySelectorAll('.model-file:last-child')
        this.selectModelEntry(elems[elems.length -1])
        this.modelFilter.select()
    }

    resetSelection() {
        this.hideModelList()
        this.showAllEntries()
        this.modelFilter.value = this.currentSelection.value
        this.modelFilter.focus()
        this.modelFilter.select()
    }

    validEntrySelected() {
        return (this.modelNoResult.style.display === 'none')
    }
    
    processKey(e) {
        switch (e.key) {
            case 'Escape':
                e.preventDefault()
                this.resetSelection()
                break
            case 'Enter':
                e.preventDefault()
                if (this.validEntrySelected()) {
                    if (this.modelList.style.display != 'block') {
                        this.showModelList()
                    }
                    else
                    {
                        this.saveCurrentSelection(this.highlightedModelEntry, this.highlightedModelEntry.innerText, this.highlightedModelEntry.dataset.path)
                        this.hideModelList()
                        this.showAllEntries()
                    }
                    this.modelFilter.focus()
                }
                else
                {
                    this.resetSelection()
                }
                break
            case 'ArrowUp':
                e.preventDefault()
                if (this.validEntrySelected()) {
                    this.selectPreviousFile()
                }
                break
            case 'ArrowDown':
                e.preventDefault()
                if (this.validEntrySelected()) {
                    this.selectNextFile()
                }
                break
            case 'ArrowLeft':
                if (this.modelList.style.display != 'block') {
                    e.preventDefault()
                }
                break
            case 'ArrowRight':
                if (this.modelList.style.display != 'block') {
                    e.preventDefault()
                }
                break
            case 'PageUp':
                e.preventDefault()
                if (this.validEntrySelected()) {
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                    this.selectPreviousFile()
                }
                break
            case 'PageDown':
                e.preventDefault()
                if (this.validEntrySelected()) {
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                    this.selectNextFile()
                }
                break
            case 'Home':
                //if (this.modelList.style.display != 'block') {
                    e.preventDefault()
                    if (this.validEntrySelected()) {
                        this.selectFirstFile()
                    }
                //}
                break
            case 'End':
                //if (this.modelList.style.display != 'block') {
                    e.preventDefault()
                    if (this.validEntrySelected()) {
                        this.selectLastFile()
                    }
                //}
                break
            default:
                //console.log(e.key)
        }
    }
    
    modelListFocus() {
        this.selectEntry()
        this.showAllEntries()
    }
    
    showModelList() {
        this.modelList.style.display = 'block'
        this.selectEntry()
        this.showAllEntries()
        //this.modelFilter.value = ''
        this.modelFilter.select() // preselect the entire string so user can just start typing.
        this.modelFilter.focus()
        this.modelFilter.style.cursor = 'auto'
    }
    
    hideModelList() {
        this.modelList.style.display = 'none'
        this.modelFilter.value = this.currentSelection.value
        this.modelFilter.style.cursor = ''
    }
    
    toggleModelList(e) {
        e.preventDefault()
        if (!this.modelFilter.disabled) {
            if (this.modelList.style.display != 'block') {
                this.showModelList()
            }
            else
            {
                this.hideModelList()
                this.modelFilter.select()
            }
        }
    }
    
    selectEntry(path) {
        if (path !== undefined) {
            const entries = this.modelElements;

            for (const elem of entries) {
                if (elem.dataset.path == path) {
                    this.saveCurrentSelection(elem, elem.innerText, elem.dataset.path)
                    this.highlightedModelEntry = elem
                    elem.scrollIntoView({block: 'nearest'})
                    break
                }
            }
        }
        
        if (this.currentSelection.elem !== undefined) {
            // select the previous element
            if (this.highlightedModelEntry !== undefined && this.highlightedModelEntry != this.currentSelection.elem) {
                this.highlightedModelEntry.classList.remove('selected')
            }
            this.currentSelection.elem.classList.add('selected')
            this.highlightedModelEntry = this.currentSelection.elem
            this.currentSelection.elem.scrollIntoView({block: 'nearest'})
        }
        else
        {
            this.selectFirstFile()
        }
    }
    
    highlightModelAtPosition(e) {
        let elem = document.elementFromPoint(e.clientX, e.clientY)
        
        if (elem.classList.contains('model-file')) {
            this.highlightModel(elem)
        }
    }
    
    highlightModel(elem) {
        if (elem.classList.contains('model-file')) {
            if (this.highlightedModelEntry !== undefined && this.highlightedModelEntry != elem) {
                this.highlightedModelEntry.classList.remove('selected')
            }
            elem.classList.add('selected')
            this.highlightedModelEntry = elem
        }
    }
    
    showAllEntries() {
        this.modelList.querySelectorAll('li').forEach(function(li) {
            if (li.id !== 'model-no-result') {
                li.style.display = 'list-item'
            }
        })
        this.modelNoResult.style.display = 'none'
    }
    
    filterList(e) {
        const filter = this.modelFilter.value.toLowerCase()
        let found = false
        let showAllChildren = false
        
        this.modelList.querySelectorAll('li').forEach(function(li) {
            if (li.classList.contains('model-folder')) {
                showAllChildren = false
            }
            if (filter == '') {
                li.style.display = 'list-item'
                found = true
            } else if (showAllChildren || li.textContent.toLowerCase().match(filter)) {
                li.style.display = 'list-item'
                if (li.classList.contains('model-folder') && li.firstChild.textContent.toLowerCase().match(filter)) {
                    showAllChildren = true
                }
                found = true
            } else {
                li.style.display = 'none'
            }
        })
    
        if (found) {
            this.modelResult.style.display = 'list-item'
            this.modelNoResult.style.display = 'none'
            const elem = this.getNextVisibleSibling(this.modelList.querySelector('.model-file'))
            this.highlightModel(elem)
            elem.scrollIntoView({block: 'nearest'})
        }
        else
        {
            this.modelResult.style.display = 'none'
            this.modelNoResult.style.display = 'list-item'
        }
        this.modelList.style.display = 'block'
    }

    /* MODEL LOADER */
    getElementDimensions(element) {
        // Clone the element
        const clone = element.cloneNode(true)
        
        // Copy the styles of the original element to the cloned element
        const originalStyles = window.getComputedStyle(element)
        for (let i = 0; i < originalStyles.length; i++) {
            const property = originalStyles[i]
            clone.style[property] = originalStyles.getPropertyValue(property)
        }
        
        // Set its visibility to hidden and display to inline-block
        clone.style.visibility = "hidden"
        clone.style.display = "inline-block"
        
        // Put the cloned element next to the original element
        element.parentNode.insertBefore(clone, element.nextSibling)
        
        // Get its width and height
        const width = clone.offsetWidth
        const height = clone.offsetHeight
        
        // Remove it from the DOM
        clone.remove()
        
        // Return its width and height
        return { width, height }
    }
    
    /**
     * @param {Array<string>} models 
     */
    sortStringArray(models) {
        models.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
    }

    populateModels() {
        this.activeModel = this.modelFilter.dataset.path
        
        this.currentSelection = { elem: undefined, value: '', path: ''}
        this.highlightedModelEntry = undefined
        this.flatModelList = []

        if(this.modelList !== undefined) {
            this.modelList.remove()
            this.modelFilterArrow.remove()
        }
        this.createDropdown()
    }

    createDropdown() {
        // create dropdown entries
        this.modelFilter.insertAdjacentElement('afterend', this.createRootModelList(this.inputModels))
        this.modelFilter.insertAdjacentElement(
            'afterend',
            this.createElement(
                'i',
                { id: `${this.modelFilter.id}-model-filter-arrow` },
                ['model-selector-arrow', 'fa-solid', 'fa-angle-down'],
            ),
        )
        this.modelFilter.classList.add('model-selector')
        this.modelFilterArrow = document.querySelector(`#${this.modelFilter.id}-model-filter-arrow`)
        if (this.modelFilterArrow) {
            this.modelFilterArrow.style.color = this.modelFilter.disabled ? 'dimgray' : ''
        }
        this.modelList = document.querySelector(`#${this.modelFilter.id}-model-list`)
        this.modelResult = document.querySelector(`#${this.modelFilter.id}-model-result`)
        this.modelNoResult = document.querySelector(`#${this.modelFilter.id}-model-no-result`)
        
        if (this.modelFilterInitialized !== true) {
            this.modelFilter.addEventListener('input', this.bind(this.filterList, this))
            this.modelFilter.addEventListener('focus', this.bind(this.modelListFocus, this))
            this.modelFilter.addEventListener('blur', this.bind(this.hideModelList, this))
            this.modelFilter.addEventListener('click', this.bind(this.showModelList, this))
            this.modelFilter.addEventListener('keydown', this.bind(this.processKey, this))

            this.modelFilterInitialized = true
        }
        this.modelFilterArrow.addEventListener('mousedown', this.bind(this.toggleModelList, this))
        this.modelList.addEventListener('mousemove', this.bind(this.highlightModelAtPosition, this))
        this.modelList.addEventListener('mousedown', this.bind(this.processClick, this))

        this.selectEntry(this.activeModel)
    }

    /**
     * 
     * @param {string} tag 
     * @param {object} attributes
     * @param {Array<string>} classes
     * @returns {HTMLElement}
     */
    createElement(tagName, attributes, classes, text, icon) {
        const element = document.createElement(tagName)
        if (attributes) {
            Object.entries(attributes).forEach(([key, value]) => {
                element.setAttribute(key, value)
            })
        }
        if (classes) {
            classes.forEach(className => element.classList.add(className))
        }
        if (icon) {
            let iconEl = document.createElement('i')
            iconEl.className = icon + ' icon'
            element.appendChild(iconEl)
        }
        if (text) {
            element.appendChild(document.createTextNode(text))
        }
        return element
    }

    /**
     * @param {Array<string | object} modelTree
     * @param {string} folderName 
     * @param {boolean} isRootFolder 
     * @returns {HTMLElement}
     */
    createModelNodeList(folderName, modelTree, isRootFolder) {
        const listElement = this.createElement('ul')

        const foldersMap = new Map()
        const modelsMap = new Map()

        modelTree.forEach(model => {
            if (Array.isArray(model)) {
                const [childFolderName, childModels] = model
                foldersMap.set(
                    childFolderName,
                    this.createModelNodeList(
                        `${folderName || ''}/${childFolderName}`,
                        childModels,
                        false,
                    ),
                )
            } else {
                const classes = ['model-file']
                if (isRootFolder) {
                    classes.push('in-root-folder')
                }
                // Remove the leading slash from the model path
                const fullPath = folderName ? `${folderName.substring(1)}/${model}` : model
                modelsMap.set(
                    model,
                    this.createElement('li', { 'data-path': fullPath }, classes, model, 'fa-regular fa-file'),
                )
            }
        })

        const childFolderNames = Array.from(foldersMap.keys())
        this.sortStringArray(childFolderNames)
        const folderElements = childFolderNames.map(name => foldersMap.get(name))

        const modelNames = Array.from(modelsMap.keys())
        this.sortStringArray(modelNames)
        const modelElements = modelNames.map(name => modelsMap.get(name))

        if (modelElements.length && folderName) {
            listElement.appendChild(this.createElement('li', undefined, ['model-folder'], folderName.substring(1), 'fa-solid fa-folder-open'))
        }

        const allModelElements = isRootFolder ? [...folderElements, ...modelElements] : [...modelElements, ...folderElements]
        allModelElements.forEach(e => listElement.appendChild(e))
        return listElement
    }

    /**
     * @param {object} modelTree
     * @returns {HTMLElement}
     */
    createRootModelList(modelTree) {
        const rootList = this.createElement(
            'ul',
            { id: `${this.modelFilter.id}-model-list` },
            ['model-list'],
        )
        rootList.appendChild(
            this.createElement(
                'li',
                { id: `${this.modelFilter.id}-model-no-result` },
                ['model-no-result'],
                'No result'
            ),
        )

        if (this.noneEntry) {
            rootList.appendChild(
                this.createElement(
                    'li',
                    { 'data-path': '' },
                    ['model-file', 'in-root-folder'],
                    this.noneEntry,
                ),
            )
        }

        const containerListItem = this.createElement(
            'li',
            { id: `${this.modelFilter.id}-model-result` },
            ['model-result'],
        )
        containerListItem.appendChild(this.createModelNodeList(undefined, modelTree, true))
        rootList.appendChild(containerListItem)

        return rootList
    }
}

/* (RE)LOAD THE MODELS */
async function getModels() {
    try {
        modelsCache = await SD.getModels()
        modelsOptions = modelsCache['options']
        if ("scan-error" in modelsCache) {
            // let previewPane = document.getElementById('tab-content-wrapper')
            let previewPane = document.getElementById('preview')
            previewPane.style.background="red"
            previewPane.style.textAlign="center"
            previewPane.innerHTML = '<H1>🔥Malware alert!🔥</H1><h2>The file <i>' + modelsCache['scan-error'] + '</i> in your <tt>models/stable-diffusion</tt> folder is probably malware infected.</h2><h2>Please delete this file from the folder before proceeding!</h2>After deleting the file, reload this page.<br><br><button onClick="window.location.reload();">Reload Page</button>'
            makeImageBtn.disabled = true
        }

        /* This code should no longer be needed. Commenting out for now, will cleanup later.
        const sd_model_setting_key = "stable_diffusion_model"
        const vae_model_setting_key = "vae_model"
        const hypernetwork_model_key = "hypernetwork_model"

        const stableDiffusionOptions = modelsOptions['stable-diffusion']
        const vaeOptions = modelsOptions['vae']
        const hypernetworkOptions = modelsOptions['hypernetwork']

        // TODO: set default for model here too
        SETTINGS[sd_model_setting_key].default = stableDiffusionOptions[0]
        if (getSetting(sd_model_setting_key) == '' || SETTINGS[sd_model_setting_key].value == '') {
            setSetting(sd_model_setting_key, stableDiffusionOptions[0])
        }
        */

        // notify ModelDropdown objects to refresh
        document.dispatchEvent(new Event('refreshModels'))
    } catch (e) {
        console.log('get models error', e)
    }
}

// reload models button
document.querySelector('#reload-models').addEventListener('click', getModels)
