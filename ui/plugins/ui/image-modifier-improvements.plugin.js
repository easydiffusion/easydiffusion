/*
    Image Modifier Improvements

    1. Allows for multiple custom modifier categories. Use # to name your custom categories, e.g.:
        #Custom category 1
        Custom modifier 1
        Custom modifier 2
        ...    
        #Custom category 2
        Custom modifier n
        ...
        #Custom category n...
        ...
    2. Restores the image modifier cards upon reloading the page.
    3. Adds a Clear All button on top of the task's image modifiers.
    4. Drop images on the modifier cards to set a custom visual (square pictures work best, e.g. 512x512).
    5. Adds import/export capabilities to the custom category editor (also exports/imports custom visuals).
    6. Collapses other categories when selecting a new one.
*/
(function() {
    "use strict"
    
    PLUGINS['MODIFIERS_LOAD'] = []

    PLUGINS['MODIFIERS_LOAD'].push({
        loader: async function() {
            var styleSheet = document.createElement("style")
            styleSheet.textContent = `
                .modifier-separator {
                    border-bottom: 1px solid var(--background-color3);
                    margin-bottom: 15px;
                    padding-bottom: 15px;
                    margin-right: 15px;
                }
        
                #modifierBackupLinks {
                    margin: 4px 0 0 0;
                }
                
                #modifierBackupLinks a {
                    cursor: pointer;
                }
                
                #modifier-settings-config textarea {
                    width: 100%;
                    height: 40vh;
                }
        
                #modifier-settings-config {
                    transition: none;
                }
        
                #image-modifier-filter {
                    box-sizing: border-box;
                    width: 98%;
                    margin-top: 4px;
                    padding: 10px;
                }

                div.modifier-card.hide {
                    display: none;
                }

                div.modifier-category.hide {
                    display: none;
                }
            `;
            document.head.appendChild(styleSheet)

            let customModifiers
            let imageModifierFilter
            let customSection = false

            // pull custom modifiers from legacy storage
            let inputCustomModifiers = localStorage.getItem(CUSTOM_MODIFIERS_KEY)
            if (inputCustomModifiers !== null) {
                customModifiersTextBox.value = inputCustomModifiers
                inputCustomModifiers = inputCustomModifiers.replace(/^\s*$(?:\r\n?|\n)/gm, "") // remove empty lines
            }
            if (inputCustomModifiers !== null && inputCustomModifiers !== '') {
                inputCustomModifiers = importCustomModifiers(inputCustomModifiers)
            }
            else
            {
                inputCustomModifiers = []
            }
            // pull custom modifiers from persistent storage
            customModifiers = await getStorageData(CUSTOM_MODIFIERS_KEY)
            if (customModifiers === undefined) {
                customModifiers = inputCustomModifiers
                saveCustomCategories()
            }
            else
            {
                customModifiers = JSON.parse(customModifiers)
                
                // update existing entries if something changed
                if (updateEntries(inputCustomModifiers, customModifiers)) {
                    saveCustomCategories()
                }
            }
            loadModifierList()
            
            // collapse the first preset section
            let preset = editorModifierEntries.getElementsByClassName('collapsible active')[0]
            if (preset !==  undefined) {
                closeCollapsible(preset.parentElement) // toggle first preset section
            }
            // set up categories auto-collapse
            autoCollapseCategories()

            // add the export and import links to the custom modifiers dialog
            const imageModifierDialog = customModifiersTextBox.parentElement
            if (imageModifierDialog.querySelector('#modifierBackupLinks') === null) {
                imageModifierDialog.insertAdjacentHTML('beforeend', `<p><small>Use the below links to export and import custom image modifiers.<br />
                                                                    (if you have set any visuals, these will be saved/restored too)</small></p><p id="modifierBackupLinks">
                                                                    <small><a id="exportModifiers">Export modifers</a> - <a id="importModifiers">Import modifiers</a></small></p>`)
            
                // export link
                let downloadLink = document.getElementById("exportModifiers")
                downloadLink.addEventListener("click", function(event) {
                    event.preventDefault()
                    downloadJSON(customModifiers, "Image Modifiers.json")
                })
                                              
                function downloadJSON(jsonData, fileName) {
                    var file = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" })
                    var fileUrl = URL.createObjectURL(file)
                    var downloadLink = document.createElement("a")
                    downloadLink.href = fileUrl
                    downloadLink.download = fileName
                    downloadLink.click()
                    URL.revokeObjectURL(fileUrl)
                }
                
                // import link
                let input = document.createElement("input")
                input.style.display = "none"
                input.type = "file"
                document.body.appendChild(input)
                
                let fileSelector = document.querySelector("#importModifiers")        
                fileSelector.addEventListener("click", function(event) {
                    event.preventDefault()
                    input.click()
                })
                
                input.addEventListener("change", function(event) {
                    let selectedFile = event.target.files[0]
                    let reader = new FileReader()
                    
                    reader.onload = function(event) {
                        customModifiers = JSON.parse(event.target.result)
                        // save the updated modifier list to persistent storage
                        saveCustomCategories()
                        // refresh the modifiers list
                        customModifiersTextBox.value = exportCustomModifiers(customModifiers)
                        saveCustomModifiers()
                        //loadModifierList()
                        input.value = ''
                    }
                    reader.readAsText(selectedFile)
                })

                function filterImageModifierList() {
                    let search = imageModifierFilter.value.toLowerCase();
                    for (let category of document.querySelectorAll(".modifier-category")) {
                      let categoryVisible = false;
                      for (let card of category.querySelectorAll(".modifier-card")) {
                        let label = card.querySelector(".modifier-card-label p").innerText.toLowerCase();
                        if (label.indexOf(search) == -1) {
                          card.classList.add("hide");
                        } else {
                          card.classList.remove("hide");
                          categoryVisible = true;
                        }
                      }
                      if (categoryVisible && search !== "") {
                        openCollapsible(category);
                        category.classList.remove("hide");
                      } else {
                        closeCollapsible(category);
                        if (search !== "") {
                            category.classList.add("hide");
                        }
                        else
                        {
                            category.classList.remove("hide");
                        }
                      }
                    }
                }
                // Call debounce function on filterImageModifierList function with 200ms wait time. Thanks JeLuf!
                const debouncedFilterImageModifierList = debounce(filterImageModifierList, 200);
                
                // Add the debounced function to the keyup event listener
                imageModifierFilter.addEventListener('keyup', debouncedFilterImageModifierList);

                // select the text on focus
                imageModifierFilter.addEventListener('focus', function(event) {
                    imageModifierFilter.select()
                });

                // empty the searchbox on escape                
                imageModifierFilter.addEventListener('keydown', function(event) {
                  if (event.keyCode === 'Escape') {
                    imageModifierFilter.value = '';
                    filterImageModifierList();
                  }
                });

                // update the custom modifiers textbox's default string
                customModifiersTextBox.placeholder = 'Enter your custom modifiers, one-per-line. Start a line with # to create custom categories.'
            }

            // refresh modifiers in the UI
            function loadModifierList() {
                let customModifiersGroupElementArray = Array.from(editorModifierEntries.querySelectorAll('.custom-modifier-category'));
                if (Array.isArray(customModifiersGroupElementArray)) {
                    customModifiersGroupElementArray.forEach(div => div.remove())
                }
                if (customModifiersGroupElement !== undefined) {
                    customModifiersGroupElement.remove()
                    customModifiersGroupElement = undefined
                }
                customModifiersGroupElementArray = []

                if (customModifiers && customModifiers.length > 0) {
                    let category = 'Custom Modifiers'
                    let modifiers = []
                    Object.keys(customModifiers).reverse().forEach(item => {
                        // new custom category
                        const elem = createModifierGroup(customModifiers[item], false, false)
                        elem.classList.add('custom-modifier-category')
                        customModifiersGroupElementArray.push(elem)
                        createCollapsibles(elem)
                        makeModifierCardDropAreas(elem)
                        customSection = true
                    })
                    if (Array.isArray(customModifiersGroupElementArray)) {
                        customModifiersGroupElementArray[0].classList.add('modifier-separator')
                    }
                    if (customModifiersGroupElement !== undefined) {
                        customModifiersGroupElement.classList.add('modifier-separator')
                    }

                    // move the searchbox atop of the image modifiers list. create it if needed.
                    imageModifierFilter = document.getElementById("image-modifier-filter") // search box
                    if (imageModifierFilter !== null) {
                        customModifierEntriesToolbar.insertAdjacentElement('afterend', imageModifierFilter);
                    }
                    else
                    {
                        customModifierEntriesToolbar.insertAdjacentHTML('afterend', `<input type="text" id="image-modifier-filter" placeholder="Search for..." autocomplete="off"/>`)
                        imageModifierFilter = document.getElementById("image-modifier-filter") // search box
                    }
                }
            }

            // transform custom modifiers from flat format to structured object
            function importCustomModifiers(input) {
                let res = []
                let lines = input.split("\n")
                let currentCategory = "(Unnamed Section)"
                let currentModifiers = []
                for (let line of lines) {
                    if (line.startsWith("#")) {
                        if (currentModifiers.length > 0) {
                            res.push({ category: currentCategory, modifiers: currentModifiers })
                        }
                        currentCategory = line.substring(1)
                        currentModifiers = []
                    } else {
                        currentModifiers.push({
                            modifier: line,
                            previews: [
                                { name: "portrait", image: "" },
                                { name: "landscape", image: "" }
                            ]
                        })
                    }
                }
                res.push({ category: currentCategory, modifiers: currentModifiers })
                return res
            }
            
            // transform custom modifiers from structured object to flat format
            function exportCustomModifiers(json) {
                let result = '';
            
                json.forEach(item => {
                    result += '#' + item.category + '\n';
                    item.modifiers.forEach(modifier => {
                        result += modifier.modifier + '\n';
                    });
                    result += '\n'; // Add a new line after each category
                });
            
                return result;
            }

            // update entries. add and remove categories/modifiers as needed.
            function updateEntries(newEntries, existingEntries) {
                let updated = false
                
                // loop through each category in existingEntries
                for (let i = 0; i < existingEntries.length; i++) {
                    let existingCategory = existingEntries[i]
                    let newCategory = newEntries.find(entry => entry.category === existingCategory.category)
                
                    if (newCategory) {
                        // if category exists in newEntries, update its modifiers
                        let newModifiers = newCategory.modifiers
                        let existingModifiers = existingCategory.modifiers
                
                        // loop through each modifier in existingModifiers
                        for (let j = 0; j < existingModifiers.length; j++) {
                            let existingModifier = existingModifiers[j]
                            let existingModifierIndex = newModifiers.findIndex(mod => mod.modifier === existingModifier.modifier)
                
                            if (existingModifierIndex === -1) {
                                // if modifier doesn't exist in newModifiers, remove it from existingModifiers
                                existingModifiers.splice(j, 1)
                                j--
                                updated = true
                            }
                        }
                
                        // loop through each modifier in newModifiers
                        for (let j = 0; j < newModifiers.length; j++) {
                            let newModifier = newModifiers[j];
                            let existingIndex = existingModifiers.findIndex(mod => mod.modifier === newModifier.modifier);
                            
                            if (existingIndex === -1) {
                                // Modifier doesn't exist in existingModifiers, so insert it at the same index in existingModifiers
                                existingModifiers.splice(j, 0, newModifier);
                                updated = true;
                            }
                        }
                    } else {
                        // if category doesn't exist in newEntries, remove it from existingEntries
                        existingEntries.splice(i, 1)
                        i--
                        updated = true
                    }
                }
                
                // loop through each category in newEntries
                for (let i = 0; i < newEntries.length; i++) {
                    let newCategory = newEntries[i]
                    let existingCategoryIndex = existingEntries.findIndex(entry => entry.category === newCategory.category)
                
                    if (existingCategoryIndex === -1) {
                        // if category doesn't exist in existingEntries, insert it at the same position
                        existingEntries.splice(i, 0, newCategory)
                        updated = true
                    }
                }
                
                return updated
            }
            
            function makeModifierCardDropAreas(elem) {
                const modifierCards = elem.querySelectorAll('.modifier-card');
                modifierCards.forEach(modifierCard => {
                    const overlay = modifierCard.querySelector('.modifier-card-overlay');
                    overlay.addEventListener('dragover', e => {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'copy';
                    });
                    overlay.addEventListener('drop', e => {
                        e.preventDefault();
                        const image = e.dataTransfer.files[0];
                        const imageContainer = modifierCard.querySelector('.modifier-card-image-container');
                        if (imageContainer.querySelector('.modifier-card-image') === null) {
                            imageContainer.insertAdjacentHTML('beforeend', `<img onerror="this.remove()" alt="Modifier Image" class="modifier-card-image">`)
                        }
                        const imageElement = imageContainer.querySelector('img');
                        const errorLabel = imageContainer.querySelector('.modifier-card-error-label');
                        if (image && image.type.startsWith('image/')) {
                            const reader = new FileReader();
                            reader.onload = () => {
                                // set the modifier card image
                                resizeImage(reader.result, 128, 128)
                                  .then(function(resizedBase64Img) {
                                    imageElement.src = resizedBase64Img
                                    errorLabel.style.display = 'none';
        
                                    // update the active tags if needed
                                    updateActiveTags()
                                    
                                    // save the customer modifiers
                                    const category = imageElement.closest('.modifier-category').querySelector('h5').innerText.slice(1)
                                    const modifier = imageElement.closest('.modifier-card').querySelector('.modifier-card-label > p').dataset.fullName
                                    setPortraitImage(category, modifier, resizedBase64Img)
                                    saveCustomCategories()
                                  })
                                  .catch(function(error) {
                                    // Log the error message to the console
                                    console.log(error);
                                  });
                                };
                            reader.readAsDataURL(image);
                        } else {
                            imageElement.remove();
                            errorLabel.style.display = 'block';
                        }
                    });
                });
            }
            
            function setPortraitImage(category, modifier, image) {
                const categoryObject = customModifiers.find(obj => obj.category === category)
                if (!categoryObject) return
            
                const modifierObject = categoryObject.modifiers.find(obj => obj.modifier === modifier)
                if (!modifierObject) return
            
                const portraitObject = modifierObject.previews.find(obj => obj.name === "portrait")
                if (!portraitObject) return
            
                portraitObject.image = image
            }

            function resizeImage(srcImage, width, height) {
                // Return a new Promise object that will resolve with the resized image data
                return new Promise(function(resolve, reject) {
                    // Create an Image object with the original base64 image data
                    const img = new Image();
                    
                    // Set up a load event listener to ensure the image has finished loading before resizing it
                    img.onload = function() {
                        // Create a canvas element
                        const canvas = document.createElement("canvas");
                        canvas.width = width;
                        canvas.height = height;
                        
                        // Draw the original image on the canvas with bilinear interpolation
                        const ctx = canvas.getContext("2d");
                        ctx.imageSmoothingEnabled = true
                        if (ctx.imageSmoothingQuality !== undefined) {
                            ctx.imageSmoothingQuality = 'high'
                        }
                        ctx.drawImage(img, 0, 0, width, height);
                        
                        // Get the base64-encoded data of the resized image
                        const resizedImage = canvas.toDataURL();
                        
                        // Resolve the Promise with the base64-encoded data of the resized image
                        resolve(resizedImage);
                    };
                        
                    // Set up an error event listener to reject the Promise if there's an error loading the image
                    img.onerror = function() {
                        reject("Error loading image");
                    };
                
                    // Set the source of the Image object to the input base64 image data
                    img.src = srcImage;
                });
            }

            async function saveCustomCategories() {
                setStorageData(CUSTOM_MODIFIERS_KEY, JSON.stringify(customModifiers))                
            }

            // collapsing other categories
            function openCollapsible(element) {
                const collapsibleHeader = element.querySelector(".collapsible");
                const handle = element.querySelector(".collapsible-handle");
                collapsibleHeader.classList.add("active")
                let content = getNextSibling(collapsibleHeader, '.collapsible-content')
                if (collapsibleHeader.classList.contains("active")) {
                    content.style.display = "block"
                    if (handle != null) {  // render results don't have a handle
                        handle.innerHTML = '&#x2796;' // minus
                    }
                }
                document.dispatchEvent(new CustomEvent('collapsibleClick', { detail: collapsibleHeader }))
            }
            
            function closeCollapsible(element) {
                const collapsibleHeader = element.querySelector(".collapsible");
                const handle = element.querySelector(".collapsible-handle");
                collapsibleHeader.classList.remove("active")
                let content = getNextSibling(collapsibleHeader, '.collapsible-content')
                if (!collapsibleHeader.classList.contains("active")) {
                    content.style.display = "none"
                    if (handle != null) {  // render results don't have a handle
                        handle.innerHTML = '&#x2795;' // plus
                    }
                }
                document.dispatchEvent(new CustomEvent('collapsibleClick', { detail: collapsibleHeader }))
            }
            
            function collapseOtherCategories(elem) {
                const modifierCategories = document.querySelectorAll('.modifier-category');
                modifierCategories.forEach(category => {
                    if (category !== elem) {
                        closeCollapsible(category)
                        //elem.scrollIntoView({ block: "nearest" })
                    }
                });
            }
            
            function autoCollapseCategories() {
                const modifierCategories = document.querySelectorAll('.modifier-category');
                modifierCategories.forEach(modifierCategory => {
                    modifierCategory.addEventListener('click', function(e) {
                        collapseOtherCategories(e.target.closest('.modifier-category'))
                    });
                });
            }
            document.dispatchEvent(new Event('loadImageModifiers')) // refresh image modifiers
        }
    })

    //PLUGINS['MODIFIERS_LOAD'].forEach(fn=>fn.loader.call())

    /* RESTORE IMAGE MODIFIERS */
    document.addEventListener("refreshImageModifiers", function(e) {
        localStorage.setItem('image_modifiers', JSON.stringify(activeTags))
        return true
    })

    // reload image modifiers at start
    document.addEventListener("loadImageModifiers", function(e) {
        let savedTags = JSON.parse(localStorage.getItem('image_modifiers'))
        let active_tags = savedTags == null ? [] : savedTags.map(x => x.name)
        
        // reload image modifiers
        refreshModifiersState(active_tags)
        
        // update inactive tags
        if (savedTags !== null) {
            const inactiveTags = savedTags.filter(tag => tag.inactive === true).map(x => x.name)
            refreshInactiveTags(inactiveTags)
        }
        
        // update the active tags if needed
        updateActiveTags()
        
        return true
    })

    function updateActiveTags() {
        activeTags.forEach((tag, index) => {
            if (tag.originElement) { // will be null if the custom tag was removed
                const modifierImage = tag.originElement.querySelector('img')
                let tinyModifierImage = tag.element.querySelector('img')
                if (modifierImage !== null) {
                    if (tinyModifierImage === null) {
                        const tinyImageContainer = tag.element.querySelector('.modifier-card-image-container')
                        tinyImageContainer.insertAdjacentHTML('beforeend', `<img onerror="this.remove()" alt="Modifier Image" class="modifier-card-image">`)
                        tinyModifierImage = tag.element.querySelector('img')
                    }
                    tinyModifierImage.src = modifierImage.src
                }
            }
        })
    }

    /* CLEAR ALL BUTTON */
    var styleSheet = document.createElement("style")
    styleSheet.textContent = `
        #removeImageTags {
            display: block;
            margin: 6px 0 3px 4px;
        }

        .clearAllImageTags {
            font-size: 8pt;
            padding: 2px;
        }
    `
    document.head.appendChild(styleSheet)

    editorModifierTagsList?.insertAdjacentHTML('beforeBegin', `
        <div id="removeImageTags"><button class="secondaryButton clearAllImageTags">Clear all</button></div>
    `)

    document.querySelector('.clearAllImageTags').addEventListener('click', function(e) {
        e.stopPropagation()
        
        // clear existing image tag cards
        editorTagsContainer.style.display = 'none'
        editorModifierTagsList.querySelectorAll('.modifier-card').forEach(modifierCard => {
            modifierCard.remove()
        })

        // reset modifier cards state
        document.querySelector('#editor-modifiers').querySelectorAll('.modifier-card').forEach(modifierCard => {
            const modifierName = modifierCard.querySelector('.modifier-card-label').innerText
            if (activeTags.map(x => x.name).includes(modifierName)) {
                modifierCard.classList.remove(activeCardClass)
                modifierCard.querySelector('.modifier-card-image-overlay').innerText = '+'
            }
        })
        activeTags = []
        document.dispatchEvent(new Event('refreshImageModifiers')) // notify image modifiers have changed
    })

    /* STORAGE MANAGEMENT */
    // Request persistent storage
    async function requestPersistentStorage() {
        if (navigator.storage && navigator.storage.persist) {
          const isPersisted = await navigator.storage.persist();
          console.log(`Persisted storage granted: ${isPersisted}`);
        }
    }
    requestPersistentStorage()

    // Open a database
    function openDB() {
        return new Promise((resolve, reject) => {
            let request = indexedDB.open("EasyDiffusionSettingsDatabase", 1);
            request.addEventListener("upgradeneeded", function() {
                let db = request.result;
                db.createObjectStore("EasyDiffusionSettings", {keyPath: "id"});
            });
            request.addEventListener("success", function() {
                resolve(request.result);
            });
            request.addEventListener("error", function() {
                reject(request.error);
            });
        });
    }
    
    // Function to write data to the object store
    function setStorageData(key, value) {
        return openDB().then(db => {
            let tx = db.transaction("EasyDiffusionSettings", "readwrite");
            let store = tx.objectStore("EasyDiffusionSettings");
            let data = {id: key, value: value};
            return new Promise((resolve, reject) => {
                let request = store.put(data);
                request.addEventListener("success", function() {
                    resolve(request.result);
                });
                request.addEventListener("error", function() {
                    reject(request.error);
                });
            });
        });
    }
    
    // Function to retrieve data from the object store
    function getStorageData(key) {
        return openDB().then(db => {
            let tx = db.transaction("EasyDiffusionSettings", "readonly");
            let store = tx.objectStore("EasyDiffusionSettings");
            return new Promise((resolve, reject) => {
                let request = store.get(key);
                request.addEventListener("success", function() {
                    if (request.result) {
                        resolve(request.result.value);
                    } else {
                        // entry not found
                        resolve();
                    }
                });
                request.addEventListener("error", function() {
                    reject(request.error);
                });
            });
        });
    }

    // indexedDB debug functions
    function getAllKeys() {
        return openDB().then(db => {
            let tx = db.transaction("EasyDiffusionSettings", "readonly");
            let store = tx.objectStore("EasyDiffusionSettings");
            let keys = [];
            return new Promise((resolve, reject) => {
                store.openCursor().onsuccess = function(event) {
                    let cursor = event.target.result;
                    if (cursor) {
                        keys.push(cursor.key);
                        cursor.continue();
                    } else {
                        resolve(keys);
                    }
                };
            });
        });
    }
    
    async function logAllKeys() {
        try {
            let keys = await getAllKeys();
            console.log("All keys:", keys);
            for (const k of keys) {
                console.log(k, await getStorageData(k))
            }
        } catch (error) {
            console.error("Error retrieving keys:", error);
        }
    }

    // USE WITH CARE - THIS MAY DELETE ALL ENTRIES
    function deleteKeys(keyToDelete) {
        let confirmationMessage = keyToDelete
            ? `This will delete the modifier with key "${keyToDelete}". Continue?`
            : "This will delete ALL modifiers. Continue?";
        if (confirm(confirmationMessage)) {
            return openDB().then(db => {
                let tx = db.transaction("EasyDiffusionSettings", "readwrite");
                let store = tx.objectStore("EasyDiffusionSettings");
                return new Promise((resolve, reject) => {
                    store.openCursor().onsuccess = function(event) {
                        let cursor = event.target.result;
                        if (cursor) {
                            if (!keyToDelete || cursor.key === keyToDelete) {
                                cursor.delete();
                            }
                            cursor.continue();
                        } else {
                            // refresh the dropdown and resolve
                            resolve();
                        }
                    };
                });
            });
        }
    }
})()
