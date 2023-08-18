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
        let modelNames = []
        let modelWeights = []

        this.modelElements.forEach((e) => {
            modelNames.push(e.name.value)
            modelWeights.push(e.weight.value)
        })

        return { modelNames: modelNames, modelWeights: modelWeights }
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

        // expand or shrink entries
        let currElements = this.modelElements
        if (currElements.length < newModelNames.length) {
            for (let i = currElements.length; i < newModelNames.length; i++) {
                this.addModelEntry()
            }
        } else {
            for (let i = newModelNames.length; i < currElements.length; i++) {
                this.removeModelEntry()
            }
        }

        // assign to the corresponding elements
        currElements = this.modelElements
        for (let i = 0; i < newModelNames.length; i++) {
            let curr = currElements[i]

            // update weight first, name second.
            // for some unholy reason this order matters for dispatch chains
            // the root of all this unholiness is because searchable-models automatically dispatches an update event
            // as soon as the value is updated via JS, which is against the DOM pattern of not dispatching an event automatically
            // unless the caller explicitly dispatches the event.
            curr.weight.value = newModelWeights[i]
            curr.name.value = newModelNames[i]
        }
    }
    get disabled() {
        return false
    }
    set disabled(state) {
        // do nothing
    }
    get modelElements() {
        let entries = this.root.querySelectorAll(".model_entry")
        entries = [...entries]
        let elements = entries.map((e) => {
            return { name: e.querySelector(".model_name").field, weight: e.querySelector(".model_weight") }
        })
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
        return this.modelContainer.childElementCount
    }
}
