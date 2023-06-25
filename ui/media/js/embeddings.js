/*

        <div id="first_embedding_model">
            <input id="embedding_model" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" />
            <span><i class="fa-solid fa-square-caret-up add-prompt help-btn"><span class="simple-tooltip top-left">Add keyword to prompt.<br>Shift-click for negative prompt.</span></i>&nbsp;<i class="fa-solid fa-square-plus add-new-input"></i></span>
        </div>

*/

"use strict"



class embeddingsWidgetClass {
    id
    modelFields


    constructor() {
        this.id=1
        this.modelFields = [{div: document.querySelector("#first_embedding_model"), dropdown: new ModelDropdown(document.querySelector("#embedding_model"), "embeddings", "None")}]
        this.addToPromptButton(document.querySelector("#first_embedding_model .add-prompt"))
        this.addNewModelSelectorButton(document.querySelector("#first_embedding_model .add-new-input"))
    }

    // remember 'this' - http://blog.niftysnippets.org/2008/04/you-must-remember-this.html
    bind(f, obj) {
        return function() {
            return f.apply(obj, arguments)
        }
    }

    addToPromptButton(button) {
        let input = button.closest("div").querySelector("input")

        button.addEventListener("click", (e) => {
            if (input.value == "None") return

            if (e.shiftKey) {
                insertAtCursor(negativePromptField, input.value)
            } else {
                insertAtCursor(promptField, input.value)
            }
        })
        button.style.cursor = "pointer"
    }

    removeModelSelectorButton(button) {
        let parent = button.closest("div")
        button.addEventListener("click", (e) => {

           parent.classList.add("displayNone")
        })

        button.style.cursor = "pointer"
    }

    addNewModelSelectorButton(button) {
        let parent = button.closest("div")

        button.addEventListener("click", this.bind((e) => {
            this.addEmbeddingDropdown(parent)
        }, this))
        button.style.cursor = "pointer"
    }

    addEmbeddingDropdown(parent=null) {
        // recycle deleted model selectors if there are any
        const hiddenSelectors = document.querySelectorAll("#embedding_model_container .displayNone")
        if (hiddenSelectors.length > 0) {
            hiddenSelectors[0].classList.remove("displayNone")
            hiddenSelectors[0].value="None"
            return
        }

        if (parent == null) {
            parent = this.modelFields[0].div
        }
        // No hidden selectors, create a new one
        let d = document.createElement("div")
        d.id = "embedding-"+this.id
        this.id++
        d.innerHTML=`<input id="embedding-field-${this.id}" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" /> `
            + '<span>'
            + '<i class="fa-solid fa-square-caret-up add-prompt help-btn"><span class="simple-tooltip top-left">Add keyword to prompt.<br>Shift-click for negative prompt.</span></i>&nbsp;'
            + '<i class="fa-solid fa-square-plus add-new-input"></i>&nbsp;'
            + '<i class="fa-solid fa-square-minus remove-input"></i>'
            + '</span>'

        parent.insertAdjacentElement("afterEnd", d)
        this.modelFields.push({div: d, dropdown: new ModelDropdown(d.querySelector("input"), "embeddings", "None")})

        this.addToPromptButton(d.querySelector(".add-prompt"))
        this.addNewModelSelectorButton(d.querySelector(".add-new-input"))
        this.removeModelSelectorButton(d.querySelector(".remove-input"))
    }
    
    get value() {
        let result = []
        for (var elem of this.modelFields) {
            if (!elem.div.classList.contains("displayNone") && elem.dropdown.value!=undefined && elem.dropdown.value!="") {
                result.push(elem.dropdown.value)
            }
        }
        return result
    }

    set value(embeddings) {
        // At least the first dropdown needs to be displayed. So if the list of embeddings is empty, add a "None" value
        if (embeddings == []) {
            embeddings = ["None"]
        }

        let selectors = document.querySelectorAll("#embedding_model_container div")
        while (selectors.length < embeddings.length) {
            this.addEmbeddingDropdown()
            selectors = document.querySelectorAll("#embedding_model_container div")
        }
        if (selectors.length > embeddings.length) {
            for (var i = embeddings.length; i < selectors.length; i++) {
                selectorss[i].classList.add("displayNone")
            }
            selectors = document.querySelectorAll("#embedding_model_container div")
        }
        for(var i=0; i<embeddings.length; i++) {
            this.modelFields[i].dropdown.value = embeddings[i]
            selectors[i].classList.remove("displayNone")
        }
    }

}

var embeddingsWidget

function initEmbeddings() {
    embeddingsWidget = new embeddingsWidgetClass()
}


