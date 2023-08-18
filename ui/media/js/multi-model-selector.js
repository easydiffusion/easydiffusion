/**
 * A component consisting of multiple model dropdowns, along with a "weight" field per model.
 *
 * Behaves like a single input element, giving an object in response to the .value field.
 *
 * Inspired by the design of the ModelDropdown component (searchable-models.js).
 */

class MultiModelSelector {
    root
    modelType
    modelNameFriendly
    defaultWeight
    weightStep

    modelContainer
    addNewButton

    counter = 0

    /* MIMIC A REGULAR INPUT FIELD */
    get id() {
        return this.root.id
    }
    get parentElement() {
        return this.root.parentElement
    }
    get parentNode() {
        return this.root.parentNode
    }
    get value() {
        return { modelNames: this.modelNames, modelWeights: this.modelWeights }
    }
    set value(modelData) {
        if (typeof modelData !== "object") {
            throw new Error("Multi-model selector expects an object containing modelNames and modelWeights as keys!")
        }
        if (!("modelNames" in modelData) || !("modelWeights" in modelData)) {
            throw new Error("modelNames or modelWeights not present in the data passed to the multi-model selector")
        }

        let newModelNames = modelData["modelNames"]
        let newModelWeights = modelData["modelWeights"]
        if (newModelNames.length !== newModelWeights.length) {
            throw new Error("Need to pass an equal number of modelNames and modelWeights!")
        }

        // update weight first, name second.
        // for some unholy reason this order matters for dispatch chains
        // the root of all this unholiness is because searchable-models automatically dispatches an update event
        // as soon as the value is updated via JS, which is against the DOM pattern of not dispatching an event automatically
        // unless the caller explicitly dispatches the event.
        this.modelWeights = newModelWeights
        this.modelNames = newModelNames
    }
    get disabled() {
        return false
    }
    set disabled(state) {
        // do nothing
    }
    getModelElements(ignoreEmpty = false) {
        let entries = this.root.querySelectorAll(".model_entry")
        entries = [...entries]
        let elements = entries.map((e) => {
            let modelName = e.querySelector(".model_name").field
            let modelWeight = e.querySelector(".model_weight")
            if (ignoreEmpty && modelName.value.trim() === "") {
                return null
            }

            return { name: modelName, weight: modelWeight }
        })
        elements = elements.filter((e) => e !== null)
        return elements
    }
    addEventListener(type, listener, options) {
        // do nothing
    }
    dispatchEvent(event) {
        // do nothing
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

    constructor(root, modelType, modelNameFriendly = undefined, defaultWeight = 0.5, weightStep = 0.02) {
        this.root = root
        this.modelType = modelType
        this.modelNameFriendly = modelNameFriendly || modelType
        this.defaultWeight = defaultWeight
        this.weightStep = weightStep

        let self = this
        document.addEventListener("refreshModels", function() {
            setTimeout(self.bind(self.populateModels, self), 1)
        })

        this.createStructure()
        this.populateModels()
    }

    createStructure() {
        this.modelContainer = document.createElement("div")
        this.modelContainer.className = "model_entries"
        this.root.appendChild(this.modelContainer)

        this.addNewButton = document.createElement("button")
        this.addNewButton.className = "add_model_entry"
        this.addNewButton.innerHTML = '<i class="fa-solid fa-plus"></i> add another ' + this.modelNameFriendly
        this.addNewButton.addEventListener("click", this.bind(this.addModelEntry, this))
        this.root.appendChild(this.addNewButton)
    }

    populateModels() {
        if (this.root.dataset.path === "") {
            if (this.length === 0) {
                this.addModelEntry() // create a single blank entry
            }
        } else {
            this.value = JSON.parse(this.root.dataset.path)
        }
    }

    addModelEntry() {
        let idx = this.counter++
        let currLength = this.length

        const modelElement = document.createElement("div")
        modelElement.className = "model_entry"
        modelElement.innerHTML = `
            <input id="${this.modelType}_${idx}" class="model_name model-filter" type="text" spellcheck="false" autocomplete="off" data-path="" />
            <input class="model_weight" type="number" step="${this.weightStep}" style="width: 50pt" value="${this.defaultWeight}" pattern="^-?[0-9]*\.?[0-9]*$" onkeypress="preventNonNumericalInput(event)">
        `
        this.modelContainer.appendChild(modelElement)

        let modelNameEl = modelElement.querySelector(".model_name")
        modelNameEl.field = new ModelDropdown(modelNameEl, this.modelType, "None")
        let modelWeightEl = modelElement.querySelector(".model_weight")

        let self = this

        function makeUpdateEvent(type) {
            return function(e) {
                e.stopPropagation()

                let modelData = self.value
                self.root.dataset.path = JSON.stringify(modelData)

                self.root.dispatchEvent(new Event(type))
            }
        }

        modelNameEl.addEventListener("change", makeUpdateEvent("change"))
        modelNameEl.addEventListener("input", makeUpdateEvent("input"))
        modelWeightEl.addEventListener("change", makeUpdateEvent("change"))
        modelWeightEl.addEventListener("input", makeUpdateEvent("input"))

        let removeBtn = document.createElement("button")
        removeBtn.className = "remove_model_btn"
        removeBtn.setAttribute("title", "Remove model")
        removeBtn.innerHTML = '<i class="fa-solid fa-minus"></i>'

        if (currLength === 0) {
            removeBtn.classList.add("displayNone")
        }

        removeBtn.addEventListener(
            "click",
            this.bind(function(e) {
                this.modelContainer.removeChild(modelElement)

                makeUpdateEvent("change")(e)
            }, this)
        )

        modelElement.appendChild(removeBtn)
    }

    removeModelEntry() {
        if (this.length === 0) {
            return
        }

        let lastEntry = this.modelContainer.lastElementChild
        lastEntry.remove()
    }

    get length() {
        return this.getModelElements().length
    }

    get modelNames() {
        return this.getModelElements(true).map((e) => e.name.value)
    }

    set modelNames(newModelNames) {
        this.resizeEntryList(newModelNames.length)

        if (newModelNames.length === 0) {
            this.getModelElements()[0].name.value = ""
        }

        // assign to the corresponding elements
        let currElements = this.getModelElements()
        for (let i = 0; i < newModelNames.length; i++) {
            let curr = currElements[i]

            curr.name.value = newModelNames[i]
        }
    }

    get modelWeights() {
        return this.getModelElements(true).map((e) => e.weight.value)
    }

    set modelWeights(newModelWeights) {
        this.resizeEntryList(newModelWeights.length)

        if (newModelWeights.length === 0) {
            this.getModelElements()[0].weight.value = this.defaultWeight
        }

        // assign to the corresponding elements
        let currElements = this.getModelElements()
        for (let i = 0; i < newModelWeights.length; i++) {
            let curr = currElements[i]

            curr.weight.value = newModelWeights[i]
        }
    }

    resizeEntryList(newLength) {
        if (newLength === 0) {
            newLength = 1
        }

        let currLength = this.length
        if (currLength < newLength) {
            for (let i = currLength; i < newLength; i++) {
                this.addModelEntry()
            }
        } else {
            for (let i = newLength; i < currLength; i++) {
                this.removeModelEntry()
            }
        }
    }
}
