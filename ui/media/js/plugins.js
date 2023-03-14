const PLUGIN_API_VERSION = "1.0"
const PLUGIN_CATALOG = 'https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/beta/ui/plugins/plugins.json'

const PLUGINS = {
    /**
     * Register new buttons to show on each output image.
     * 
     * Example:
     * PLUGINS['IMAGE_INFO_BUTTONS'].push({
     *   text: 'Make a Similar Image',
     *   on_click: function(origRequest, image) {
     *     let newTaskRequest = getCurrentUserRequest()
     *     newTaskRequest.reqBody = Object.assign({}, origRequest, {
     *       init_image: image.src,
     *       prompt_strength: 0.7,
     *       seed: Math.floor(Math.random() * 10000000)
     *     })
     *     newTaskRequest.seed = newTaskRequest.reqBody.seed
     *     createTask(newTaskRequest)
     *   },
     *   filter: function(origRequest, image) {
     *     // this is an optional function. return true/false to show/hide the button
     *     // if this function isn't set, the button will always be visible
     *     return true
     *   }
     * })
     */
    IMAGE_INFO_BUTTONS: [],
    GET_PROMPTS_HOOK: [],
    MODIFIERS_LOAD: [],
    TASK_CREATE: [],
    OUTPUTS_FORMATS: new ServiceContainer(
        function png() { return (reqBody) => new SD.RenderTask(reqBody) }
        , function jpeg() { return (reqBody) => new SD.RenderTask(reqBody) }
        , function webp() { return (reqBody) => new SD.RenderTask(reqBody) }
    ),
}
PLUGINS.OUTPUTS_FORMATS.register = function(...args) {
    const service = ServiceContainer.prototype.register.apply(this, args)
    if (typeof outputFormatField !== 'undefined') {
        const newOption = document.createElement("option")
        newOption.setAttribute("value", service.name)
        newOption.innerText = service.name
        outputFormatField.appendChild(newOption)
    }
    return service
}

function loadScript(url) {
    const script = document.createElement('script')
    const promiseSrc = new PromiseSource()
    script.addEventListener('error', () => promiseSrc.reject(new Error(`Script "${url}" couldn't be loaded.`)))
    script.addEventListener('load', () => promiseSrc.resolve(url))
    script.src = url + '?t=' + Date.now()

    console.log('loading script', url)
    document.head.appendChild(script)

    return promiseSrc.promise
}

async function loadUIPlugins() {
    try {
        const res = await fetch('/get/ui_plugins')
        if (!res.ok) {
            console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
            return
        }
        const plugins = await res.json()
        const loadingPromises = plugins.map(loadScript)
        return await Promise.allSettled(loadingPromises)
    } catch (e) {
        console.log('error fetching plugin paths', e)
    }
}

/* PLUGIN MANAGER */
/* plugin tab */
//document.querySelector('.tab-container #tab-news')?.insertAdjacentHTML('beforebegin', `
document.querySelector('.tab-container')?.insertAdjacentHTML('beforeend', `
    <span id="tab-plugin" class="tab">
        <span><i class="fa fa-puzzle-piece icon"></i> Plugins</span>
    </span>
`)

document.querySelector('#tab-content-wrapper')?.insertAdjacentHTML('beforeend', `
    <div id="tab-content-plugin" class="tab-content">
        <div id="plugin" class="tab-content-inner">
            Loading...
        </div>
    </div>
`)

const tabPlugin = document.querySelector('#tab-plugin')
if (tabPlugin) {
    linkTabContents(tabPlugin)
}

const plugin = document.querySelector('#plugin')
plugin.innerHTML = `
<div id="plugin-manager" class="tab-content-inner">
    <h1>Plugin Manager</h1>
    <div class="plugin-manager-intro">Changes take effect after reloading the page<br /></div>
    <div class="parameters-table"></div>
</div>`
const pluginsTable = document.querySelector("#plugin-manager .parameters-table")

/* search box */
function filterPlugins() {
    let search = pluginFilter.value.toLowerCase();
    let searchTerms = search.split(' ');
    let labels = pluginsTable.querySelectorAll("label.plugin-name");

    for (let i = 0; i < labels.length; i++) {
        let label = labels[i].innerText.toLowerCase();
        let match = true;

        for (let j = 0; j < searchTerms.length; j++) {
            let term = searchTerms[j].trim();
            if (term && label.indexOf(term) === -1) {
                match = false;
                break;
            }
        }

        if (match) {
            labels[i].parentNode.parentNode.style.display = "flex";
        } else {
            labels[i].parentNode.parentNode.style.display = "none";
        }
    }
}

// Call debounce function on filterImageModifierList function with 200ms wait time. Thanks JeLuf!
const debouncedFilterPlugins = debounce(filterPlugins, 200);

// add the searchbox
pluginsTable.insertAdjacentHTML('beforebegin', `<input type="text" id="plugin-filter" placeholder="Search for..." autocomplete="off"/>`)
const pluginFilter = document.getElementById("plugin-filter") // search box

// Add the debounced function to the keyup event listener
pluginFilter.addEventListener('keyup', debouncedFilterPlugins);

// select the text on focus
pluginFilter.addEventListener('focus', function(event) {
    pluginFilter.select()
});

// empty the searchbox on escape                
pluginFilter.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        pluginFilter.value = '';
        filterPlugins();
    }
});

// focus on the search box upon tab selection
document.addEventListener("tabClick", (e) => {
    if (e.detail.name == 'plugin') {
        pluginFilter.focus()
    }
})

// refresh link
pluginsTable.insertAdjacentHTML('afterend', `<p id="refresh-plugins"><small><a id="refresh-plugins-link">Refresh plugins</a></small></p>
    <p><small>(Plugin developers, add your plugins to <a href='https://github.com/patriceac/Easy-Diffusion-Plugins' target='_blank'>Easy-Diffusion-Plugins</a>)</small></p>`)
const refreshPlugins = document.getElementById("refresh-plugins")
refreshPlugins.addEventListener("click", async function(event) {
    event.preventDefault()
    await initPlugins(true)
    debounce(showToast('Plugins refreshed', 3000), 3000)
})

function showToast(message, duration) {
    const toast = document.createElement("div");
    toast.classList.add("plugin-toast");
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("hide");
        setTimeout(() => {
            toast.remove();
        }, 500);
    }, duration);
}

function matchPluginFileNames(fileName1, fileName2) {
    const regex = /^(.+?)(?:-\d+(\.\d+)*)?\.plugin\.js$/;
    const match1 = fileName1.match(regex);
    const match2 = fileName2.match(regex);

    if (match1 && match2 && match1[1] === match2[1]) {
        return true; // the two file names match
    } else {
        return false; // the two file names do not match
    }
}

function checkFileNameInArray(paths, filePath) {
    // Normalize the path separators to forward slashes and make the file names lowercase
    const normalizedFilePath = filePath.replace(/\\/g, "/").toLowerCase();

    // Strip off the path from the file name
    const fileName = normalizedFilePath.substring(normalizedFilePath.lastIndexOf("/") + 1);

    // Check if the file name exists in the array of paths
    return paths.some(path => {
        // Normalize the path separators to forward slashes and make the file names lowercase
        const normalizedPath = path.replace(/\\/g, "/").toLowerCase();

        // Strip off the path from the file name
        const baseName = normalizedPath.substring(normalizedPath.lastIndexOf("/") + 1);

        // Check if the file names match and return the result as a boolean
        return matchPluginFileNames(fileName, baseName);
    });
}

/* fill in the plugins table */
async function initPluginTable(plugins) {
    plugins.sort((a, b) => a.name.localeCompare(b.name, undefined, {sensitivity: 'base'}))
    plugins.forEach(plugin => {
        const name = plugin.name
        const author = plugin.author ? ', by ' + plugin.author : ''
        const version = plugin.version ? ' (version: ' + plugin.version + ')' : ''
        const note = plugin.description ? `<small>${plugin.description.replaceAll('\n', '<br>')}</small>` : `<small>No description</small>`;
        const icon = plugin.icon ? `<i class="fa ${plugin.icon}"></i>` : '<i class="fa fa-puzzle-piece"></i>';
        const newrow = document.createElement('div')
        const localPluginFound = checkFileNameInArray(localPlugins, plugin.url)
        newrow.innerHTML = `
            <div>${icon}</div>
            <div><label class="plugin-name">${name}${author}${version}</label>${note}</div>
            <div>${localPluginFound ? "<span class='plugin-installed-locally'>Installed locally</span>" : '<input id="plugin-' + plugin.id + '" name="plugin-' + plugin.id + '" type="checkbox"'}</div>`
        pluginsTable.appendChild(newrow)
        const pluginToggle = pluginsTable.querySelector('#plugin-' + plugin.id)
        if (pluginToggle !== null) {
            pluginToggle.checked = plugin.enabled // set initial state of checkbox
            pluginToggle.addEventListener('click', async () => {
                plugin.enabled = pluginToggle.checked
                // if the plugin got enabled, download the plugin's code
                if (plugin.enabled) {
                    const pluginSource = await getDocument(plugin.url);
                    if (pluginSource !== null) {
                        plugin.code = pluginSource
                    }
                    else
                    {
                        plugin.enabled = false
                    }
                }
                setStorageData('plugins', JSON.stringify(plugins))                    
            })
        }
    })
    prettifyInputs(pluginsTable)
}

/* version management. Thanks Madrang! */
const parseVersion = function(versionString, options = {}) {
    if (typeof versionString === "undefined") {
        throw new Error("versionString is undefined.");
    }
    if (typeof versionString !== "string") {
        throw new Error("versionString is not a string.");
    }
    const lexicographical = options && options.lexicographical;
    const zeroExtend = options && options.zeroExtend;
    let versionParts = versionString.split('.');
    function isValidPart(x) {
        const re = (lexicographical ? /^\d+[A-Za-z]*$/ : /^\d+$/);
        return re.test(x);
    }

    if (!versionParts.every(isValidPart)) {
        throw new Error("Version string is invalid.");
    }

    if (zeroExtend) {
        while (versionParts.length < 4) {
            versionParts.push("0");
        }
    }
    if (!lexicographical) {
        versionParts = versionParts.map(Number);
    }
    return versionParts;
};

const versionCompare = function(v1, v2, options = {}) {
    if (typeof v1 == "undefined") {
        throw new Error("vi is undefined.");
    }
    if (typeof v2 === "undefined") {
        throw new Error("v2 is undefined.");
    }

    let v1parts;
    if (typeof v1 === "string") {
        v1parts = parseVersion(v1, options);
    } else if (Array.isArray(v1)) {
        v1parts = [...v1];
        if (!v1parts.every (p => typeof p === "number" && p !== NaN)) {
            throw new Error("v1 part array does not only contains numbers.");
        }
    } else {
        throw new Error("v1 is of an unexpected type: " + typeof v1);
    }

    let v2parts;
    if (typeof v2 === "string") {
        v2parts = parseVersion(v2, options);
    } else if (Array.isArray(v2)) {
        v2parts = [...v2];
        if (!v2parts.every(p => typeof p === "number" && p !== NaN)) {
            throw new Error("v2 part array does not only contains numbers.");
        }
    } else {
        throw new Error("v2 is of an unexpected type: " + typeof v2);
    }

    while (v1parts.length < v2parts.length) {
        v1parts.push("0");
    }
    while (v2parts.length < v1parts.length) {
        v2parts.push("0");
    }

    for (let i = 0; i < v1parts.length; ++i) {
        if (v2parts.length == i) {
            return 1;
        }
        if (v1parts[i] == v2parts[i]) {
            continue;
        } else if (v1parts[i] > v2parts[i]) {
            return 1;
        } else {
            return -1;
        }
    }
    return 0;
};

function filterPluginsByMinEDVersion(plugins, EDVersion) {
    const filteredPlugins = plugins.filter(plugin => {
        if (plugin.minEDVersion) {
            return versionCompare(plugin.minEDVersion, EDVersion) <= 0;
        }
        return true;
    });
    
    return filteredPlugins;
}

function extractVersionNumber(elem) {
    const versionStr = elem.innerHTML;
    const regex = /v(\d+\.\d+\.\d+)/;
    const matches = regex.exec(versionStr);
    if (matches && matches.length > 1) {
        return matches[1];
    } else {
        return null;
    }
}
const EasyDiffusionVersion = extractVersionNumber(document.querySelector('#top-nav > #logo'))

/* PLUGIN MANAGEMENT */
let plugins
let localPlugins

async function initPlugins(refreshPlugins = false) {
    let pluginsLoaded
    
    const res = await fetch('/get/ui_plugins')
    if (!res.ok) {
        console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
    }
    else
    {
        localPlugins = await res.json()
    }
    
    if (refreshPlugins === false) {
        // try and load plugins from local cache
        plugins = await getStorageData('plugins')
        if (plugins !== undefined) {
            plugins = JSON.parse(await getStorageData('plugins'))
            plugins = filterPluginsByMinEDVersion(plugins, EasyDiffusionVersion) // remove plugins that don't meet the min ED version requirement
            pluginsTable.innerHTML = ''
            initPluginTable(plugins)
            await loadPlugins(plugins)
            pluginsLoaded = true
        }
        else
        {
            plugins = []
            pluginsLoaded = false
        }
    }

    // update plugins asynchronously (updated versions will be available next time the UI is loaded)
    let pluginCatalog = await getDocument(PLUGIN_CATALOG)
    if (pluginCatalog !== null) {
        console.log('Plugin catalog successfully downloaded')
        pluginCatalog = JSON.parse(pluginCatalog)
        
        // remove from plugins the entries that don't have mandatory fields (id, name, url)
        plugins = plugins.filter((plugin) => { return plugin.id && plugin.name && plugin.url; });
        await downloadPlugins(pluginCatalog, plugins, refreshPlugins)
        
        // remove from plugins the entries that no longer exist in the catalog
        plugins = plugins.filter((plugin) => { return pluginCatalog.find((p) => p.id === plugin.id) });
        
        // remove plugins that don't meet the min ED version requirement
        plugins = filterPluginsByMinEDVersion(plugins, EasyDiffusionVersion)
        
        // save the remaining plugins            
        setStorageData('plugins', JSON.stringify(plugins))

        // refresh the display of the plugins table
        pluginsTable.innerHTML = ''
        initPluginTable(plugins)
        if (pluginsLoaded && pluginsLoaded === false) {
            loadPlugins(plugins)
        }
    }
    else
    {
        console.log('Could not download the plugin catalog')
    }
}
initPlugins()

async function loadPlugins(plugins) {
    plugins.forEach((plugin) => {
        if (plugin.enabled === true) {
            const localPluginFound = checkFileNameInArray(localPlugins, plugin.url);
            if (!localPluginFound) {
                try {
                    eval(plugin.code);
                    console.log("Plugin " + plugin.name + " loaded");
                } catch (err) {
                    console.error("Error loading plugin " + plugin.name + ": " + err.message);
                }
            } else {
                console.log("Skipping plugin " + plugin.name + " (installed locally)");
            }
        }
    });
}

async function getFileHash(url) {
    const regex = /^https:\/\/raw\.githubusercontent\.com\/(?<owner>[^/]+)\/(?<repo>[^/]+)\/(?<branch>[^/]+)\/(?<filePath>.+)$/;
    const match = url.match(regex);
    if (!match) {
        console.error('Invalid GitHub repository URL.');
        return Promise.resolve(null);
    }
    const owner = match.groups.owner;
    const repo = match.groups.repo;
    const branch = match.groups.branch;
    const filePath = match.groups.filePath;
    const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${filePath}?ref=${branch}`;
    
    const response = await fetch(apiUrl);
    const data = await response.json();
    
    // Store the new sha value for future reference
    return data.sha;
}

async function downloadPlugins(pluginCatalog, plugins, refreshPlugins) {
    // download the plugins as needed
    for (const plugin of pluginCatalog) {
        const existingPlugin = plugins.find(p => p.id === plugin.id);
        // get the file hash in the GitHub repo
        let sha
        if (existingPlugin?.enabled === true) {
            sha = await getFileHash(plugin.url)
        }
        if ((existingPlugin?.enabled === true && (refreshPlugins || existingPlugin?.sha !== sha || existingPlugin?.code === undefined))) {
            const pluginSource = await getDocument(plugin.url);
            if (pluginSource !== null) {
                console.log(`Plugin ${plugin.name} downloaded`);
                // Update the corresponding plugin
                const updatedPlugin = {
                    ...existingPlugin,
                    icon: plugin.icon ? plugin.icon : "fa-puzzle-piece",
                    id: plugin.id,
                    name: plugin.name,
                    description: plugin.description,
                    url: plugin.url,
                    version: plugin.version,
                    code: pluginSource,
                    author: plugin.author,
                    sha: sha
                };
                // Replace the old plugin in the plugins array
                const pluginIndex = plugins.indexOf(existingPlugin);
                if (pluginIndex >= 0) {
                    plugins.splice(pluginIndex, 1, updatedPlugin);
                } else {
                    plugins.push(updatedPlugin);
                }
            }
        }
        else if (existingPlugin !== undefined) {
            // Update the corresponding plugin's metadata
            const updatedPlugin = {
                ...existingPlugin,
                icon: plugin.icon ? plugin.icon : "fa-puzzle-piece",
                id: plugin.id,
                name: plugin.name,
                description: plugin.description,
                url: plugin.url,
                version: plugin.version,
                author: plugin.author
            };
            // Replace the old plugin in the plugins array
            const pluginIndex = plugins.indexOf(existingPlugin);
            plugins.splice(pluginIndex, 1, updatedPlugin);
        }
        else
        {
            plugins.push(plugin);
        }
    }
}

async function getDocument(url) {
    try {
        let response = await fetch(url === PLUGIN_CATALOG ? PLUGIN_CATALOG : url, { cache: "no-cache" });
        if (!response.ok) {
            throw new Error(`Response error: ${response.status} ${response.statusText}`);
        }
        let document = await response.text();
        return document;
    } catch (error) {
        console.error(error);
        return null;
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
async function openDB() {
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
async function setStorageData(key, value) {
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
async function getStorageData(key) {
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
async function getAllKeys() {
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
async function deleteKeys(keyToDelete) {
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
                        resolve();
                    }
                };
            });
        });
    }
}
