/*
    A simple template library manager. Save/restore metadata as templates without having to mess with .txt or .json files. If you found a prompt/model/sampler/etc. combination that you really like, this is an easy way to save it so you can create more of the same later.

    Features:
    - Save/reload task *templates*
    - Search box for easy access
    - Supports template renaming and deletion
    - Export/import function to back up and restore your templates
    - Can restore seeds (optional)
*/
(function() {
   "use strict"

    var styleSheet = document.createElement("style")
    styleSheet.textContent = `
        .templateContainer {
            float: left;
            margin: 0 0 0 4px;
        }
        
        .saveAsTemplate {
            background: var(--accent-color);
            border: 1px solid var(--accent-color);
            color: rgb(255, 221, 255);
            padding: 3pt 6pt;
            margin-right: 6pt;
            float: right;
        }
        
        .saveAsTemplate:hover {
            background: hsl(var(--accent-hue), 100%, calc(var(--accent-lightness) + 6%));
        }
        
        .saveAsTemplateInfo {
            background: var(--accent-color);
            border: 1px solid var(--accent-color);
            color: rgb(255, 221, 255);
            padding: 3pt 6pt;
            float: inline;
        }
        
        .saveAsTemplateInfo:hover {
            background: var(--accent-color);
            cursor: default;
        }
        
        .saveAsTemplateInfo:active {
          top: 0px;
          left: 0px;
        }
        
        .editAllTemplates {
            font-size: 9pt;
            display: inline;
            background-color: var(--accent-color);
            border: none;
        }
        
        #editAllTemplates.secondaryButton:hover {
            background: hsl(var(--accent-hue), 100%, calc(var(--accent-lightness) + 6%));
        }

        #template-list-container::-webkit-scrollbar {
            width: 10px;
        }
        
        #template-list-container::-webkit-scrollbar-track {
            background: var(--background-color1);
            border-radius: 10px;
        }
        
        #template-list-container::-webkit-scrollbar-thumb {
            background: var(--background-color4);
            width: 10px;
            border-radius: 8px;
        }

        #template-list-container {
            width: 100%;
            height: 40vh;
            overflow-x: auto;
            overflow-y: auto;
            margin-bottom: 4px;
            border: 1px solid var(--background-color1);
            border-radius: 8px;
        }
        
        #template-filter {
            box-sizing: border-box;
            width: 100%;
            margin-bottom: 15px;
            padding: 10px;
        }
         
        #template-list {
            list-style: none;
            margin: 0px 0px 0px 0px;
            padding: 0px 0px 0px 0px;
        }
        
        #template-list li {
            padding: 0px 0px 0px 10px;
            border-bottom: 1px solid #444;
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-align: left;
        }
        
        #template-list li.last-item {
            border-bottom: none;
        }
        
        #template-list li:hover {
            background: var(--background-color1);
        }
        
        #template-list li.hide {
            display: none;
        }

        .templateName {
            width: 100%;
            cursor: default;
        }

        .inputTemplateName {
            width: 100%;
        }
        
        .generateFromTemplate {
            background: none;
            border: none;
            padding: 10px 6px 10px 6px;
        }
        
        .generateFromTemplate:hover {
            color: darkolivegreen;
            background: var(--background-color1);
        }

        #generateFromTemplate i:hover {
            transition: 0.1s all;        
        }

        .lastGeneration {
            outline: 1px dotted grey;
            padding: 4px;
            z-index: 2;
        }
        
        .saveTemplate {
            background: none;
            border: none;
            padding: 10px 6px 10px 6px;
        }
        
        .saveTemplate:hover {
            color: darkslateblue;
            background: var(--background-color1);
        }

        #saveTemplate i:hover {
            transition: 0.1s all;        
        }
        
        .editTemplate {
            background: none;
            border: none;
            padding: 10px 6px 10px 6px;
        }
        
        .editTemplate:hover {
            color: darkslateblue;
            background: var(--background-color1);
        }

        #editTemplate i:hover {
            transition: 0.1s all;        
        }
        
        .deleteTemplate {
            background: none;
            border: none;
            padding: 10px 6px 10px 6px;
        }
        
        .deleteTemplate:hover {
            color: darkred;
            background: none;
        }

        #deleteTemplate i:hover {
            transition: 0.1s all;        
        }

        #templateBackupLinks a {
            cursor: pointer;
        }

        .restoreSeeds {
            float: left;
            transition: none;
            margin-left: 8px;
        }

        .slideshowImageCount {
            float: right;
            margin-right: 8px;
        }
        
        .footerTemplateManager {
            margin-top: 48px;
        }
        
        #task-templates-editor {
            transition: none;
        }
    `;
    document.head.appendChild(styleSheet)
    promptsFromFileBtn.insertAdjacentHTML('afterend', `<small> or pick a </small><button id="editAllTemplates" class="secondaryButton editAllTemplates"><i class='fa-solid fa-rectangle-list'></i> Template</button>`)

    let taskTemplates
    let noTemplateView
    let templateView
    let templateFilter
    let templateList
    let entryList
    let templateEditorBtn
    let templateEditorOverlay
    let editingTemplate = false
    let restoreSeeds
    let lastGeneration
    let lastTemplateName
    let slideshowImagecount

    async function loadTemplateList(newTemplateName) {
        // load templates
        taskTemplates = await getStorageData('task templates')
        if (taskTemplates === undefined) {
            taskTemplates = []
        }
        else
        {
            taskTemplates = JSON.parse(taskTemplates)
        }

        // template selector
        if (templateList === undefined) {
            document.querySelector('#modifier-settings-config').insertAdjacentHTML('beforeBegin', `
                <div id="task-templates-editor" class="popup" tabindex="0">
                    <div>
                        <i class="close-button fa-solid fa-xmark"></i>
                        <h1>Template manager</h1>
                        <div id="noTemplateView">
                            You haven't saved any template yet. You can save templates using the <button class="secondaryButton saveAsTemplateInfo"><i class="fa-solid fa-bookmark"></i> Save as template</button> button of the tasks showing up in the right pane.
                            <p id="templateBackupLinks"><small><a id="importTemplates"></a></small></p>
                        </div>
                        <div id="templateView">
                            <p>Manage your saved templates</p>
                            <input type="text" id="template-filter" placeholder="Search For..." autocomplete="off"/>
                            <div id="template-list-container">
                                <ul id="template-list">
                                </ul>
                            </div>
                            <div class="restoreSeeds">
                                <input id="restore_seeds" name="restore_seeds" type="checkbox">
                                <label for="restore_seeds">Restore seeds</label>
                            </div>
                            <div class="slideshowImageCount">
                                <label>Image count for <i class='fa-solid fa-images'></i> <input id="slideshow_image_count" name="slideshow_image_count" type="input" size="4" onkeypress="preventNonNumericalInput(event)"></label>
                            </div>
                            <p class="footerTemplateManager"><small>Click a template to load it. You can also edit existing template names or delete them.</small></p>
                            <p id="templateBackupLinks"><small><a id="exportTemplates">Export templates</a> - <a id="importTemplates">Import templates</a></small></p>
                        </div>
                    </div>
                </div>
            `)
            prettifyInputs(document);
            noTemplateView = document.getElementById("noTemplateView")
            templateView = document.getElementById("templateView")
            templateEditorBtn = document.getElementById("editAllTemplates")
            templateEditorOverlay = document.getElementById("task-templates-editor")
            templateList = document.getElementById("template-list")

            // save/restore the restore seeds toggle state
            restoreSeeds = document.querySelector("#restore_seeds")
            restoreSeeds.addEventListener('click', (e) => {
                localStorage.setItem('restore_seeds', restoreSeeds.checked)
            })
            restoreSeeds.checked = localStorage.getItem('restore_seeds') == "true"

            // save/restore the image count
            slideshowImagecount = document.querySelector("#slideshow_image_count")
            slideshowImagecount.addEventListener('change', (e) => {
                localStorage.setItem('slideshow_image_count', slideshowImagecount.value > 0 ? slideshowImagecount.value : 256)
            })
            slideshowImagecount.value = localStorage.getItem('slideshow_image_count') > 0 ? localStorage.getItem('slideshow_image_count') : 256

            // export link
            let downloadLink = document.getElementById("exportTemplates")
            downloadLink.addEventListener("click", function(event) {
                event.preventDefault()
                downloadJSON(taskTemplates, "Templates backup.json")
            })
            
            // import link
            let input = document.createElement("input")
            input.style.display = "none"
            input.type = "file"
            document.body.appendChild(input)
            
            let fileSelectors = document.querySelectorAll("#importTemplates")
            for (let fileSelector of fileSelectors) {
                fileSelector.innerHTML = "Import templates"
            
                fileSelector.addEventListener("click", function(event) {
                    event.preventDefault()
                    input.click()
                })
            }
            
            input.addEventListener("change", function(event) {
                let selectedFile = event.target.files[0]
                let reader = new FileReader()
                
                reader.onload = function(event) {
                    let fileData = JSON.parse(event.target.result)
                    taskTemplates = mergeArrays(taskTemplates, fileData)
                    // save the updated template list to persistent storage
                    saveTemplates()
                    // refresh the templates list
                    loadTemplateList()
                    input.value = ''
                }
                reader.readAsText(selectedFile)
            })

            function mergeArrays(originalArray, importedArray) {
                const mergedArray = [...originalArray]
                importedArray.forEach(importedTask => {
                    const foundIndex = mergedArray.findIndex(originalTask => originalTask.name === importedTask.name)
                    if (foundIndex === -1) {
                        mergedArray.push(importedTask)
                    } else {
                        mergedArray[foundIndex] = { ...mergedArray[foundIndex], ...importedTask }
                    }
                })
                return mergedArray
            }

            // event handlers
            templateFilter = document.getElementById("template-filter") // search box
            function filterTemplateList() {
                if (entryList !== undefined) {
                    let search = templateFilter.value.toLowerCase()
                    for (let i of entryList) {
                        let item = i.innerText.toLowerCase()
                        if (item.indexOf(search) == -1) {
                            i.parentNode.classList.add("hide")
                        }
                        else {
                            i.parentNode.classList.remove("hide")
                        }
                    }
                }
            }
            templateFilter.addEventListener('keyup', filterTemplateList)
            
            templateEditorOverlay.addEventListener('keydown', function(e) {
                if (editingTemplate === false && e.key === "Escape") {
                    if (templateFilter.value !== '') {
                        templateFilter.value = ''
                    }
                    else
                    {
                        hideTemplateDialog()
                    }
                    e.stopPropagation()
                }
            })
            
            templateEditorBtn.addEventListener('click', function(e) {
                document.getElementById("template-filter").value = ''
                filterTemplateList()
                templateEditorOverlay.classList.add("active")
                templateEditorOverlay.focus()
                templateFilter.focus()
                e.stopPropagation()
            })
            
            // close button
            document.querySelectorAll('#task-templates-editor.popup').forEach(popup => {
                popup.addEventListener('click', event => {
                    if (event.target == popup) {
                        hideTemplateDialog()
                    }
                })
                const closeButton = popup.querySelector(".close-button")
                if (closeButton) {
                    closeButton.addEventListener('click', () => {
                        hideTemplateDialog()
                    })
                }
            })
        }
        else
        {
            // clear the template list
            templateList.innerHTML = ''
            lastTemplateName = undefined
        }

        // close the template dialog
        function hideTemplateDialog() {
            templateEditorOverlay.classList.remove('active')
            if (lastGeneration !== undefined) {
                lastGeneration.classList.remove('lastGeneration')
                lastGeneration = undefined
            }
        }

        // populate the template list
        let newTemplateEntry
        if (taskTemplates.length > 0) {
            noTemplateView.style.display = 'none'
            templateView.style.display = ''
            
            taskTemplates.sort((a, b) => a.name?.toLowerCase().localeCompare(b.name?.toLowerCase()))
            taskTemplates.forEach(function(template, index) {
                const l = document.createElement("li")
                const templateName = document.createElement("div")
                const inputTemplateName = document.createElement("input")
                const generateButton = document.createElement("button")
                const generateMoreButton = document.createElement("button")
                const saveButton = document.createElement("button")
                const editButton = document.createElement("button")
                const deleteButton = document.createElement("button")
                templateName.classList.add("templateName")
                inputTemplateName.style.display = "none"
                inputTemplateName.value = template.name
                inputTemplateName.classList.add("inputTemplateName")
                generateButton.id = "generateFromTemplate"
                generateButton.classList.add("secondaryButton", "generateFromTemplate")
                generateButton.innerHTML = `<i class='fa-solid fa-circle-play'></i>`
                generateMoreButton.id = "generateMoreFromTemplate"
                generateMoreButton.classList.add("secondaryButton", "generateFromTemplate")
                generateMoreButton.innerHTML = `<i class='fa-solid fa-images'></i>`
                saveButton.id = "saveTemplate"
                saveButton.classList.add("secondaryButton", "saveTemplate")
                saveButton.innerHTML = `<i class='fa-solid fa-floppy-disk'></i>`
                editButton.id = "editTemplate"
                editButton.classList.add("secondaryButton", "editTemplate")
                editButton.innerHTML = `<i class='fa-solid fa-pen-to-square'></i>`
                deleteButton.id = "deleteTemplate"
                deleteButton.classList.add("secondaryButton", "deleteTemplate")
                deleteButton.innerHTML = `<i class='fa-solid fa-trash-can'></i>`
                templateName.innerHTML = `${template.name}`
                l.appendChild(templateName)
                l.appendChild(inputTemplateName)
                l.appendChild(generateButton)
                l.appendChild(generateMoreButton)
                l.appendChild(saveButton)
                l.appendChild(editButton)
                l.appendChild(deleteButton)
                templateList.appendChild(l)
    
                // if a template name was passed, go to edit mode
                if (template.name === newTemplateName) {
                    editingTemplate = true
                    templateName.style.display = "none"
                    inputTemplateName.style.display = "block"
                    inputTemplateName.focus()
                    inputTemplateName.select()
                    newTemplateEntry = l
                }
                
                // select template
                l.addEventListener("click", (event) => {
                    if (!editingTemplate) {
                        restoreTask(template.task)
                        document.querySelector('#task-templates-editor.popup').classList.remove("active")
                        event.stopPropagation()
                        lastTemplateName = template.name
                    }
                })

                // generate from template button
                generateButton.addEventListener("click", (event) => {
                    if (lastGeneration !== undefined) {
                        lastGeneration.classList.remove('lastGeneration')
                    }
                    templateName.classList.add('lastGeneration')
                    lastGeneration = templateName
                    restoreTask(template.task)
                    makeImage()
                    event.stopPropagation()
                    lastTemplateName = template.name
                })
    
                // generate MORE from template button
                generateMoreButton.addEventListener("click", (event) => {
                    if (lastGeneration !== undefined) {
                        lastGeneration.classList.remove('lastGeneration')
                    }
                    templateName.classList.add('lastGeneration')
                    lastGeneration = templateName
                    restoreTask(template.task)
                    numOutputsTotalField.value = slideshowImagecount.value > 0 ? slideshowImagecount.value : 256
                    makeImage()
                    document.querySelector('#task-templates-editor.popup').classList.remove("active")
                    event.stopPropagation()
                    lastTemplateName = template.name
                })
    
                // save template button
                saveButton.addEventListener("click", (event) => {
                    const filename = shortenFileName(template.name, 128)
                    if (filename !== '') {
                        downloadJSON([template], template.name + ".json")
                        event.stopPropagation()
                    }
                    else
                    {
                        console.log(`Couldn't save file: ${filename}`)
                    }
                })
    
                // edit template button
                editButton.addEventListener("click", (event) => {
                    editingTemplate = true
                    templateName.style.display = "none"
                    inputTemplateName.style.display = "block"
                    inputTemplateName.focus()
                    inputTemplateName.select()
                    event.stopPropagation()
                    lastTemplateName = undefined
                })
    
                // delete template button
                deleteButton.addEventListener("click", (event) => {
                    shiftOrConfirm(event, "Remove this template?", () => {
                        taskTemplates = taskTemplates.filter(function(el) {
                            return el !== template
                        })
                        l.remove()

                        // save the updated template list
                        saveTemplates()
                        
                        // refresh the dialog as neeed
                        if (taskTemplates.length == 0) {
                            noTemplateView.style.display = ''
                            templateView.style.display = 'none'
                            templateEditorOverlay.focus()
                        }                        
                    })
                    event.stopPropagation()
                    lastTemplateName = undefined
                })
                
                const handleInputBlurOrKeyUp = (event) => {
                    if (event.type === "blur" || event.key === "Enter") {
                        if (inputTemplateName.value.trim() !== '') {
                            let newName = inputTemplateName.value.trim()
                            if (newName !== template.name) {
                                let existingNames = taskTemplates.map(t => t.name)
                
                                // Check if the new name already exists
                                if (existingNames.includes(newName)) {
                                    // If it does, append a number to make it unique
                                    let uniqueNameFound = false
                                    let newNameNumber = 2
                                    while (!uniqueNameFound) {
                                        newName = `${inputTemplateName.value} (${newNameNumber})`
                                        if (newName == template.name || !existingNames.includes(newName)) {
                                            uniqueNameFound = true
                                        } else {
                                            newNameNumber += 1
                                        }
                                    }
                                }
                            }
    
                            // save the new name
                            template.name = newName
                            inputTemplateName.value = newName
                            templateName.innerHTML = newName
                            templateName.style.display = "block"
                            inputTemplateName.style.display = "none"
                            templateFilter.focus()
    
                            // save the updated template list
                            saveTemplates()
                            editingTemplate = false
                        }
                    } else if (event.key === "Escape") {
                        event.stopPropagation()
                        inputTemplateName.value = template.name
                        templateName.style.display = "block"
                        inputTemplateName.style.display = "none"
                        templateFilter.focus()
                        editingTemplate = false
                    }
                }
                inputTemplateName.addEventListener("blur", handleInputBlurOrKeyUp)
                inputTemplateName.addEventListener("keyup", handleInputBlurOrKeyUp)
    
                // if editing a new entry, make it visible
                if (newTemplateEntry !== undefined) {
                    newTemplateEntry.scrollIntoView({ block: "center" })
                }
            })
            const lastItem = templateList.lastElementChild
            lastItem.classList.add("last-item")
            entryList = document.querySelectorAll("#template-list div.templateName") // all list items
        }
        else
        {
            noTemplateView.style.display = ''
            templateView.style.display = 'none'
        }
    }
    loadTemplateList()

    function restoreTask(task) {
        if (!restoreSeeds.checked) {
            delete task.seed
            delete task.reqBody.seed
        }
        restoreTaskToUI(task, TASK_REQ_NO_EXPORT)
    }
                                          
    function downloadJSON(jsonData, fileName) {
        var file = new Blob([JSON.stringify(jsonData, null, 2)], { type: "application/json" })
        var fileUrl = URL.createObjectURL(file)
        var downloadLink = document.createElement("a")
        downloadLink.href = fileUrl
        downloadLink.download = fileName
        downloadLink.click()
        URL.revokeObjectURL(fileUrl)
    }

    function shortenFileName(fileName, maxLength) {
        // Define an object that maps invalid characters to their escaped equivalents
        const escapeChars = {
            '\\': '-',
            '/': '-',
            ':': '-',
            '*': '-',
            '?': '-',
            '"': '',
            '<': '',
            '>': '',
            '|': ''
        };
    
        // Replace any invalid characters with their escaped equivalents
        let escapedFileName = fileName;
        for (const [char, escapedChar] of Object.entries(escapeChars)) {
            escapedFileName = escapedFileName.split(char).join(escapedChar);
        }
    
        // Shorten the escaped file name
        if (escapedFileName.length <= maxLength) {
            return escapedFileName; // File name is already short enough
        }
    
        const extension = escapedFileName.slice(escapedFileName.lastIndexOf('.'));
        const fileNameWithoutExtension = escapedFileName.slice(0, escapedFileName.lastIndexOf('.'));
        const maxFileNameLength = maxLength - extension.length;
    
        if (maxFileNameLength <= 0) {
            return ''; // Max length is too short to include the extension
        }
    
        const truncatedFileName = fileNameWithoutExtension.slice(0, maxFileNameLength).trim();
    
        return truncatedFileName + '...' + extension;
    }

    async function saveTemplates() {
        setStorageData('task templates', JSON.stringify(taskTemplates))
    }

    // observe for changes in the preview pane
    var observer = new MutationObserver(function (mutations) {
        mutations.forEach(async function (mutation) {
            //console.log(mutation.addedNodes[0].className)
            if (mutation.addedNodes[0] !== undefined && mutation.addedNodes[0].className == 'imageTaskContainer') {
                const taskConfig = mutation.addedNodes[0].querySelector(' .useSettings')
                if (taskConfig !== undefined && taskConfig.parentNode.querySelector('.saveAsTemplate') === null) {
                    taskConfig.insertAdjacentHTML("afterend", `<button class="secondaryButton saveAsTemplate"><i class="fa-solid fa-bookmark"></i> Save as template</button>`)
                    mutation.addedNodes[0].querySelector('.saveAsTemplate').addEventListener('click', function(e) {
                        let newName
                        
                        e.stopPropagation()
                        const templateTask = createTaskFromTaskEntry(mutation.addedNodes[0])
                        const templateName = lastTemplateName !== undefined ? lastTemplateName : templateTask.reqBody.prompt.trim()
                        
                        // Check if the template name already exists
                        let existingNames = taskTemplates.map(t => t.name)
                        newName = templateName
                        if (existingNames.includes(templateName)) {
                            // If it does, append a number to make it unique
                            let uniqueNameFound = false
                            let newNameNumber = 2
                            while (!uniqueNameFound) {
                                newName = `${templateName} (${newNameNumber})`
                                if (!existingNames.includes(newName)) {
                                    uniqueNameFound = true
                                } else {
                                    newNameNumber += 1
                                }
                            }
                        }
                        
                        // save the new template
                        let newTemplate = {}
                        newTemplate.name = newName
                        newTemplate.task = templateTask
                        taskTemplates.push(newTemplate)
                        saveTemplates()

                        // refresh the dropdown
                        loadTemplateList(newName)

                        // show the templates dialog
                        templateEditorOverlay.classList.add("active")
                    })
                }                
            }
        })
    })
    observer.observe(document.getElementById('preview-content') || document.getElementById('preview'), { // maintain backward compatibility with current Main version (ED 2.5.15)
            childList: true,
            subtree: (document.getElementById('preview-content') === null),  // no need to scan the subtree if preview-content is present (ED 2.5.16 and later)
            attributes: true
    })

    // create task object from task entry
    function createTaskFromTaskEntry(taskEntry) {
        const task = htmlTaskMap.get(taskEntry)
        
        if (task) {
            let newTask = {}
            
            if ('numOutputsTotal' in task) {
                newTask.numOutputsTotal = task.numOutputsTotal
            }
            if ('seed' in task) {
                newTask.seed = task.seed
            }
            if (!('reqBody' in task)) {
                return
            }
            newTask.reqBody = {}
            for (const key in TASK_MAPPING) {
                if (key in task.reqBody && !TASK_REQ_NO_EXPORT.includes(key)) {
                    newTask.reqBody[key] = task.reqBody[key]
                }
            }
            // original prompt is not part of TASK_MAPPING and needs to be explicitly added for proper restoration
            if ('original_prompt' in task.reqBody) {
                newTask.reqBody.original_prompt = task.reqBody.original_prompt
            }
            return newTask
        }
    }

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
            ? `This will delete the template with key "${keyToDelete}". Continue?`
            : "This will delete ALL templates. Continue?";
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
                            loadTemplateList();
                            resolve();
                        }
                    };
                });
            });
        }
    }
})()
