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
<input id="stable_diffusion_model" type="text" spellcheck="false" class="model-filter" data-path="" />

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
    
    findPreviousSibling(elem, previous = true) {
        let sibling = previous ? elem.previousElementSibling : elem
        let lastSibling = elem
        
        while (sibling && sibling.classList.contains('model-file')) {
            if (sibling.style.display == 'list-item') return sibling
            lastSibling = sibling
            sibling = sibling.previousElementSibling
        }
        
        // no more siblings, look for previous parent if any
        if (sibling && sibling.classList.contains('model-folder')) {
            return this.findPreviousSibling(sibling.firstElementChild.lastElementChild, false)
        }
        else if (lastSibling.parentElement.parentElement && lastSibling.parentElement.parentElement.previousElementSibling && lastSibling.parentElement.parentElement.previousElementSibling.firstElementChild && lastSibling.parentElement.parentElement.previousElementSibling.firstElementChild.lastElementChild) {
            return this.findPreviousSibling(lastSibling.parentElement.parentElement.previousElementSibling.firstElementChild.lastElementChild, false)
        }
        else if (lastSibling.parentElement.parentElement.previousElementSibling) {
            return this.findPreviousSibling(lastSibling.parentElement.parentElement.previousElementSibling, false)
        }
    }
    
    findNextSibling(elem, next = true) {
        let sibling = next ? elem.nextElementSibling : elem
        let lastSibling = elem
    
        while (sibling && sibling.classList.contains('model-file')) {
            if (sibling.style.display == 'list-item') return sibling
            lastSibling = sibling
            sibling = sibling.nextElementSibling
        }
        
        // no more siblings, look for next parent if any
        if (lastSibling.nextElementSibling) {
            return this.findNextSibling(lastSibling.nextElementSibling.firstElementChild.firstElementChild, false)
        }
        else if (lastSibling.parentElement.parentElement.nextElementSibling && lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild && lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild.firstElementChild) {
            return this.findNextSibling(lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild.firstElementChild, false)
        }
        else if (lastSibling.parentElement.parentElement.nextElementSibling && lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild && lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild.firstElementChild) {
            return this.findNextSibling(lastSibling.parentElement.parentElement.nextElementSibling.firstElementChild.firstElementChild, false)
        }
        else if (lastSibling.parentElement.parentElement.nextElementSibling && lastSibling.parentElement.parentElement.nextElementSibling) {
            return this.findNextSibling(lastSibling.parentElement.parentElement.nextElementSibling, false)
        }
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
        const elem = this.findPreviousSibling(this.highlightedModelEntry)
        if (elem) {
            this.selectModelEntry(elem)
        }
        else
        {
            this.highlightedModelEntry.parentElement.parentElement.scrollIntoView({block: 'nearest'})
        }
        this.modelFilter.select()
    }
    
    selectNextFile() {
        this.selectModelEntry(this.findNextSibling(this.highlightedModelEntry))
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
        this.modelFilter.value = ''
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
        if (this.modelList.style.display != 'block') {
            this.showModelList()
        }
        else
        {
            this.hideModelList()
            this.modelFilter.select()
        }
    }
    
    selectEntry(path) {
        if (path !== undefined) {
            const entries = this.modelList.querySelectorAll('.model-file');

            for (let i = 0; i < entries.length; i++) {
                if (entries[i].dataset.path == path) {
                    this.saveCurrentSelection(entries[i], entries[i].innerText, entries[i].dataset.path)
                    this.highlightedModelEntry = entries[i]
                    entries[i].scrollIntoView({block: 'nearest'})
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
            const elem = this.findNextSibling(this.modelList.querySelector('.model-file'), false)
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
    flattenModelList(models, path) {
        models.forEach(entry => {
            if (Array.isArray(entry)) {
                this.flattenModelList(entry[1], path + '/' + entry[0])
            }
            else
            {
                this.flatModelList.push(path == '' ? entry : path + '/' + entry)
            }
        })
    }

    // sort models
    getFolder(model) {
        return model.substring(0, model.lastIndexOf('/') + 1)
    }
    
    sortModels(models) {
        let found
        do {
            found = false
            for (let i = 0; i < models.length - 1; i++) {
                if (
                    (this.getFolder(models[i]) == this.getFolder(models[i+1]) && models[i].toLowerCase() > models[i+1].toLowerCase()) // same folder, sort by alphabetical order
                    || (this.getFolder(models[i]).toLowerCase() > this.getFolder(models[i+1]).toLowerCase()) // L1 folder > L2 folder
                ) {
                    [models[i], models[i+1]] = [models[i+1], models[i]];
                    found = true
                }
            }
        } while (found)
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
        // prepare to sort the models
        this.flattenModelList(this.inputModels, '')
        this.sortModels(this.flatModelList)

        // create dropdown entries
        this.modelFilter.insertAdjacentHTML('afterend', this.parseModels(this.flatModelList))
        this.modelFilter.classList.add('model-selector')
        this.modelFilterArrow = document.querySelector(`#${this.modelFilter.id}-model-filter-arrow`)
        this.modelList = document.querySelector(`#${this.modelFilter.id}-model-list`)
        this.modelResult = document.querySelector(`#${this.modelFilter.id}-model-result`)
        this.modelNoResult = document.querySelector(`#${this.modelFilter.id}-model-no-result`)
        this.modelList.style.display = 'block'
        this.modelFilter.style.width = this.modelList.offsetWidth + 'px'
        this.modelFilterArrow.style.height = this.modelFilter.offsetHeight + 'px'
        this.modelList.style.display = 'none'
    
        this.modelFilter.addEventListener('input', this.bind(this.filterList, this))
        this.modelFilter.addEventListener('focus', this.bind(this.modelListFocus, this))
        this.modelFilter.addEventListener('blur', this.bind(this.hideModelList, this))
        this.modelFilter.addEventListener('click', this.bind(this.showModelList, this))
        this.modelFilter.addEventListener('keydown', this.bind(this.processKey, this))
        this.modelFilterArrow.addEventListener('mousedown', this.bind(this.toggleModelList, this))
        this.modelList.addEventListener('mousemove', this.bind(this.highlightModelAtPosition, this))
        this.modelList.addEventListener('mousedown', this.bind(this.processClick, this))

        this.selectEntry(this.activeModel)
    }

    parseModels(models) {
        let html = `<i id="${this.modelFilter.id}-model-filter-arrow" class="model-selector-arrow fa-solid fa-angle-down"></i>
            <ul id="${this.modelFilter.id}-model-list" class="model-list">
                <li id="${this.modelFilter.id}-model-no-result" class="model-no-result">No result</li>
                <li id="${this.modelFilter.id}-model-result" class="model-result">
            <ul>
        `
        if (this.noneEntry != '') {
            html += `<li data-path='' class='model-file in-root-folder'>${this.noneEntry}</li>`
        }
        
        let currentFolder = ''
        models.forEach(entry => {
            const folder = entry.substring(0, 1) == '/' ? entry.substring(1, entry.lastIndexOf('/')) : ''
            if (folder !== '' && folder !== currentFolder) {
                if (currentFolder != '') {
                    html += '</ul></li>'
                }
                html += `<li class='model-folder'>/${folder}<ul>`
                currentFolder = folder
            }
            else if (folder == '' && currentFolder !== '') {
                currentFolder = ''
                html += '</ul></li>'
            }
            const modelName = entry.substring(entry.lastIndexOf('/') + 1)
            if (entry.substring(0, 1) == '/') {
                entry = entry.substring(1)
            }
            html += `<li data-path='${entry}' class='model-file${currentFolder == '' ? ' in-root-folder' : ''}'>${modelName}</li>`
        })
        if (currentFolder != '') {
            html += '</ul></li>'
        }
        
        html += `
                    </ul>
                </li>
            </ul>
            `
        return html
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
