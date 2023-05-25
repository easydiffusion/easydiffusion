const PLUGIN_API_VERSION = "1.0"

const PLUGIN_CATALOG = 'https://raw.githubusercontent.com/patriceac/Easy-Diffusion-Plugins/main/plugins.json'
const PLUGIN_CATALOG_GITHUB = 'https://github.com/patriceac/Easy-Diffusion-Plugins/blob/main/plugins.json'

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
        function png() {
            return (reqBody) => new SD.RenderTask(reqBody)
        },
        function jpeg() {
            return (reqBody) => new SD.RenderTask(reqBody)
        },
        function webp() {
            return (reqBody) => new SD.RenderTask(reqBody)
        }
    ),
}
PLUGINS.OUTPUTS_FORMATS.register = function(...args) {
    const service = ServiceContainer.prototype.register.apply(this, args)
    if (typeof outputFormatField !== "undefined") {
        const newOption = document.createElement("option")
        newOption.setAttribute("value", service.name)
        newOption.innerText = service.name
        outputFormatField.appendChild(newOption)
    }
    return service
}

function loadScript(url) {
    const script = document.createElement("script")
    const promiseSrc = new PromiseSource()
    script.addEventListener("error", () => promiseSrc.reject(new Error(`Script "${url}" couldn't be loaded.`)))
    script.addEventListener("load", () => promiseSrc.resolve(url))
    script.src = url + "?t=" + Date.now()

    console.log("loading script", url)
    document.head.appendChild(script)

    return promiseSrc.promise
}

async function loadUIPlugins() {
    try {
        const res = await fetch("/get/ui_plugins")
        if (!res.ok) {
            console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
            return
        }
        const plugins = await res.json()
        const loadingPromises = plugins.map(loadScript)
        return await Promise.allSettled(loadingPromises)
    } catch (e) {
        console.log("error fetching plugin paths", e)
    }
}


/* PLUGIN MANAGER */
/* plugin tab */
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
    <i id="plugin-notification-button" class="fa-solid fa-message">
        <span class="plugin-notification-pill" id="notification-pill" style="display: none"></span>
    </i>
    <div id="plugin-notification-list" style="display: none">
        <h1>Notifications</h1>
        <div class="plugin-manager-intro">The latest plugin updates are listed below</div>
        <div class="notifications-table"></div>
        <div class="no-notification">No new notifications</div>
    </div>
    <div id="plugin-manager-section">
        <h1>Plugin Manager</h1>
        <div class="plugin-manager-intro">Changes take effect after reloading the page</div>
        <div class="plugins-table"></div>
    </div>
</div>`
const pluginsTable = document.querySelector("#plugin-manager-section .plugins-table")
const pluginNotificationTable = document.querySelector("#plugin-notification-list .notifications-table")
const pluginNoNotification = document.querySelector("#plugin-notification-list .no-notification")

/* notification center */
const pluginNotificationButton = document.getElementById("plugin-notification-button");
const pluginNotificationList = document.getElementById("plugin-notification-list");
const notificationPill = document.getElementById("notification-pill");
const pluginManagerSection = document.getElementById("plugin-manager-section");
let pluginNotifications;

// Add event listener to show/hide the action center
pluginNotificationButton.addEventListener("click", function () {
    // Hide the notification pill when the action center is opened
    notificationPill.style.display = "none"
    pluginNotifications.lastUpdated = Date.now()

    // save the notifications
    setStorageData('notifications', JSON.stringify(pluginNotifications))

    renderPluginNotifications()

    if (pluginNotificationList.style.display === "none") {
        pluginNotificationList.style.display = "block"
        pluginManagerSection.style.display = "none"
    } else {
        pluginNotificationList.style.display = "none"
        pluginManagerSection.style.display = "block"
    }
})

document.addEventListener("tabClick", (e) => {
    if (e.detail.name == 'plugin') {
        pluginNotificationList.style.display = "none"
        pluginManagerSection.style.display = "block"
    }
})

async function addPluginNotification(pluginNotifications, messageText, error) {
    const now = Date.now()
    pluginNotifications.entries.unshift({ date: now, text: messageText, error: error }); // add new entry to the beginning of the array
    if (pluginNotifications.entries.length > 50) {
        pluginNotifications.entries.length = 50 // limit array length to 50 entries
    }
    pluginNotifications.lastUpdated = now
    notificationPill.style.display = "block"
    // save the notifications
    await setStorageData('notifications', JSON.stringify(pluginNotifications))
}

function timeAgo(inputDate) {
    const now = new Date();
    const date = new Date(inputDate);
    const diffInSeconds = Math.floor((now - date) / 1000);
    const units = [
        { name: 'year', seconds: 31536000 },
        { name: 'month', seconds: 2592000 },
        { name: 'week', seconds: 604800 },
        { name: 'day', seconds: 86400 },
        { name: 'hour', seconds: 3600 },
        { name: 'minute', seconds: 60 },
        { name: 'second', seconds: 1 }
    ];

    for (const unit of units) {
        const unitValue = Math.floor(diffInSeconds / unit.seconds);
        if (unitValue > 0) {
            return `${unitValue} ${unit.name}${unitValue > 1 ? 's' : ''} ago`;
        }
    }

    return 'just now';
}

function convertSeconds(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;

    let timeParts = [];
    if (hours === 1) {
        timeParts.push(`${hours} hour`);
    } else if (hours > 1) {
        timeParts.push(`${hours} hours`);
    }
    if (minutes === 1) {
        timeParts.push(`${minutes} minute`);
    } else if (minutes > 1) {
        timeParts.push(`${minutes} minutes`);
    }
    if (remainingSeconds === 1) {
        timeParts.push(`${remainingSeconds} second`);
    } else if (remainingSeconds > 1) {
        timeParts.push(`${remainingSeconds} seconds`);
    }

    return timeParts.join(', and ');
}

function renderPluginNotifications() {
    pluginNotificationTable.innerHTML = ''

    if (pluginNotifications.entries?.length > 0) {
        pluginNoNotification.style.display = "none"
        pluginNotificationTable.style.display = "block"
    }
    else {
        pluginNoNotification.style.display = "block"
        pluginNotificationTable.style.display = "none"
    }
    for (let i = 0; i < pluginNotifications.entries?.length; i++) {
        const date = pluginNotifications.entries[i].date
        const text = pluginNotifications.entries[i].text
        const error = pluginNotifications.entries[i].error
        const newRow = document.createElement('div')

        newRow.innerHTML = `
            <div${error === true ? ' class="notification-error"' : ''}>${text}</div>
            <div><small>${timeAgo(date)}</small></div>
        `;
        pluginNotificationTable.appendChild(newRow)
    }
}

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
            labels[i].closest('.plugin-container').style.display = "flex";
        } else {
            labels[i].closest('.plugin-container').style.display = "none";
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
pluginFilter.addEventListener('focus', function (event) {
    pluginFilter.select()
});

// empty the searchbox on escape                
pluginFilter.addEventListener('keydown', function (event) {
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
    <p><small>(Plugin developers, add your plugins to <a href='${PLUGIN_CATALOG_GITHUB}' target='_blank'>plugins.json</a>)</small></p>`)
const refreshPlugins = document.getElementById("refresh-plugins")
refreshPlugins.addEventListener("click", async function (event) {
    event.preventDefault()
    await initPlugins(true)
})

function showPluginToast(message, duration = 5000, error = false, addNotification = true) {
    if (addNotification === true) {
        addPluginNotification(pluginNotifications, message, error)
    }
    try {
        showToast(message, duration, error)
    } catch (error) {
        console.error('Error while trying to show toast:', error);
    }
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

function extractFilename(filepath) {
    // Normalize the path separators to forward slashes and make the file names lowercase
    const normalizedFilePath = filepath.replace(/\\/g, "/").toLowerCase();

    // Strip off the path from the file name
    const fileName = normalizedFilePath.substring(normalizedFilePath.lastIndexOf("/") + 1);

    return fileName
}

function checkFileNameInArray(paths, filePath) {
    // Strip off the path from the file name
    const fileName = extractFilename(filePath);

    // Check if the file name exists in the array of paths
    return paths.some(path => {
        // Strip off the path from the file name
        const baseName = extractFilename(path);

        // Check if the file names match and return the result as a boolean
        return matchPluginFileNames(fileName, baseName);
    });
}

function isGitHub(url) {
    return url.startsWith("https://raw.githubusercontent.com/") === true
}

/* fill in the plugins table */
function getIncompatiblePlugins(pluginId) {
    const enabledPlugins = plugins.filter(plugin => plugin.enabled && plugin.id !== pluginId);
    const incompatiblePlugins = enabledPlugins.filter(plugin => plugin.compatIssueIds?.includes(pluginId));
    const pluginNames = incompatiblePlugins.map(plugin => plugin.name);
    if (pluginNames.length === 0) {
        return null;
    }
    const pluginNamesList = pluginNames.map(name => `<li>${name}</li>`).join('');
    return `<ul>${pluginNamesList}</ul>`;
}

async function initPluginTable(plugins) {
    pluginsTable.innerHTML = ''
    plugins.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
    plugins.forEach(plugin => {
        const name = plugin.name
        const author = plugin.author ? ', by ' + plugin.author : ''
        const version = plugin.version ? ' (version: ' + plugin.version + ')' : ''
        const warning = getIncompatiblePlugins(plugin.id) ? `<span class="plugin-warning${plugin.enabled ? '' : ' hide'}">This plugin might conflict with:${getIncompatiblePlugins(plugin.id)}</span>` : ''
        const note = plugin.description ? `<small>${plugin.description.replaceAll('\n', '<br>')}</small>` : `<small>No description</small>`;
        const icon = plugin.icon ? `<i class="fa ${plugin.icon}"></i>` : '<i class="fa fa-puzzle-piece"></i>';
        const newRow = document.createElement('div')
        const localPluginFound = checkFileNameInArray(localPlugins, plugin.url)

        newRow.innerHTML = `
            <div>${icon}</div>
            <div><label class="plugin-name">${name}${author}${version}</label>${warning}${note}<span class='plugin-source'>Source: <a href="${plugin.url}" target="_blank">${extractFilename(plugin.url)}</a><span></div>
            <div>
                ${localPluginFound ? "<span class='plugin-installed-locally'>Installed locally</span>" :
                (plugin.localInstallOnly ? '<span class="plugin-installed-locally">Download and<br />install manually</span>' :
                    (isGitHub(plugin.url) ?
                        '<input id="plugin-' + plugin.id + '" name="plugin-' + plugin.id + '" type="checkbox">' :
                        '<button id="plugin-' + plugin.id + '-install" class="tertiaryButton"></button>'
                    )
                )
            }
            </div>`;
        newRow.classList.add('plugin-container')
        //console.log(plugin.id, plugin.localInstallOnly)
        pluginsTable.appendChild(newRow)
        const pluginManualInstall = pluginsTable.querySelector('#plugin-' + plugin.id + '-install')
        updateManualInstallButtonCaption()

        // checkbox event handler
        const pluginToggle = pluginsTable.querySelector('#plugin-' + plugin.id)
        if (pluginToggle !== null) {
            pluginToggle.checked = plugin.enabled // set initial state of checkbox
            pluginToggle.addEventListener('change', async () => {
                const container = pluginToggle.closest(".plugin-container");
                const warningElement = container.querySelector(".plugin-warning");

                // if the plugin got enabled, download the plugin's code
                plugin.enabled = pluginToggle.checked
                if (plugin.enabled) {
                    const pluginSource = await getDocument(plugin.url);
                    if (pluginSource !== null) {
                        // Store the current scroll position before navigating away
                        const currentPosition = window.pageYOffset;
                        initPluginTable(plugins)
                        // When returning to the page, set the scroll position to the stored value
                        window.scrollTo(0, currentPosition);
                        warningElement?.classList.remove("hide");
                        plugin.code = pluginSource
                        loadPlugins([plugin])
                        console.log(`Plugin ${plugin.name} installed`);
                        showPluginToast("Plugin " + plugin.name + " installed");
                    }
                    else {
                        plugin.enabled = false
                        pluginToggle.checked = false
                        console.error(`Couldn't download plugin ${plugin.name}`);
                        showPluginToast("Failed to install " + plugin.name + " (Couldn't fetch " + extractFilename(plugin.url) + ")", 5000, true);
                    }
                } else {
                    warningElement?.classList.add("hide");
                    // Store the current scroll position before navigating away
                    const currentPosition = window.pageYOffset;
                    initPluginTable(plugins)
                    // When returning to the page, set the scroll position to the stored value
                    window.scrollTo(0, currentPosition);
                    console.log(`Plugin ${plugin.name} uninstalled`);
                    showPluginToast("Plugin " + plugin.name + " uninstalled");
                }
                await setStorageData('plugins', JSON.stringify(plugins))
            })
        }

        // manual install event handler
        if (pluginManualInstall !== null) {
            pluginManualInstall.addEventListener('click', async () => {
                pluginDialogOpenDialog(inputOK, inputCancel)
                pluginDialogTextarea.value = plugin.code ? plugin.code : ''
                pluginDialogTextarea.select()
                pluginDialogTextarea.focus()
            })
        }
        // Dialog OK
        async function inputOK() {
            let pluginSource = pluginDialogTextarea.value
            // remove empty lines and trim existing lines
            plugin.code = pluginSource
            if (pluginSource.trim() !== '') {
                plugin.enabled = true
                console.log(`Plugin ${plugin.name} installed`);
                showPluginToast("Plugin " + plugin.name + " installed");
            }
            else {
                plugin.enabled = false
                console.log(`No code provided for plugin ${plugin.name}, disabling the plugin`);
                showPluginToast("No code provided for plugin " + plugin.name + ", disabling the plugin");
            }
            updateManualInstallButtonCaption()
            await setStorageData('plugins', JSON.stringify(plugins))
        }
        // Dialog Cancel
        async function inputCancel() {
            plugin.enabled = false
            console.log(`Installation of plugin ${plugin.name} cancelled`);
            showPluginToast("Cancelled installation of " + plugin.name);
        }
        // update button caption
        function updateManualInstallButtonCaption() {
            if (pluginManualInstall !== null) {
                pluginManualInstall.innerHTML = plugin.code === undefined || plugin.code.trim() === '' ? 'Install' : 'Edit'
            }
        }
    })
    prettifyInputs(pluginsTable)
    filterPlugins()
}

/* version management. Thanks Madrang! */
const parseVersion = function (versionString, options = {}) {
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

const versionCompare = function (v1, v2, options = {}) {
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
        if (!v1parts.every(p => typeof p === "number" && p !== NaN)) {
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
let initPluginsInProgress = false

async function initPlugins(refreshPlugins = false) {
    let pluginsLoaded
    if (initPluginsInProgress === true) {
        return
    }
    initPluginsInProgress = true

    const res = await fetch('/get/ui_plugins')
    if (!res.ok) {
        console.error(`Error HTTP${res.status} while loading plugins list. - ${res.statusText}`)
    }
    else {
        localPlugins = await res.json()
    }

    if (refreshPlugins === false) {
        // load the notifications
        pluginNotifications = await getStorageData('notifications')            
        if (typeof pluginNotifications === "string") {
            try {
                pluginNotifications = JSON.parse(pluginNotifications)
            } catch (e) {
                console.error("Failed to parse pluginNotifications", e);
                pluginNotifications = {};
                pluginNotifications.entries = [];
            }
        }
        if (pluginNotifications !== undefined) {
            if (pluginNotifications.entries && pluginNotifications.entries.length > 0 && pluginNotifications.entries[0].date && pluginNotifications.lastUpdated <= pluginNotifications.entries[0].date) {
                notificationPill.style.display = "block";
            }
        } else {
            pluginNotifications = {};
            pluginNotifications.entries = [];
        }

        // try and load plugins from local cache
        plugins = await getStorageData('plugins')
        if (plugins !== undefined) {
            plugins = JSON.parse(plugins)

            // remove duplicate entries if any (should not happen)
            plugins = deduplicatePluginsById(plugins)

            // remove plugins that don't meet the min ED version requirement
            plugins = filterPluginsByMinEDVersion(plugins, EasyDiffusionVersion)

            // remove from plugins the entries that don't have mandatory fields (id, name, url)
            plugins = plugins.filter((plugin) => { return plugin.id !== '' && plugin.name !== '' && plugin.url !== ''; });

            // populate the table
            initPluginTable(plugins)
            await loadPlugins(plugins)
            pluginsLoaded = true
        }
        else {
            plugins = []
            pluginsLoaded = false
        }
    }

    // update plugins asynchronously (updated versions will be available next time the UI is loaded)
    if (refreshAllowed()) {
        let pluginCatalog = await getDocument(PLUGIN_CATALOG)
        if (pluginCatalog !== null) {
            let parseError = false;
            try {
                pluginCatalog = JSON.parse(pluginCatalog);
                console.log('Plugin catalog successfully downloaded');
            } catch (error) {
                console.error('Error parsing plugin catalog:', error);
                parseError = true;
            }

            if (!parseError) {
                await downloadPlugins(pluginCatalog, plugins, refreshPlugins)

                // update compatIssueIds
                updateCompatIssueIds()

                // remove plugins that don't meet the min ED version requirement
                plugins = filterPluginsByMinEDVersion(plugins, EasyDiffusionVersion)

                // remove from plugins the entries that don't have mandatory fields (id, name, url)
                plugins = plugins.filter((plugin) => { return plugin.id !== '' && plugin.name !== '' && plugin.url !== ''; });

                // remove from plugins the entries that no longer exist in the catalog
                plugins = plugins.filter((plugin) => { return pluginCatalog.some((p) => p.id === plugin.id) });

                if (pluginCatalog.length > plugins.length) {
                    const newPlugins = pluginCatalog.filter((plugin) => {
                        return !plugins.some((p) => p.id === plugin.id);
                    });

                    newPlugins.forEach((plugin, index) => {
                        setTimeout(() => {
                            showPluginToast(`New plugin "${plugin.name}" is available.`);
                        }, (index + 1) * 1000);
                    });
                }

                let pluginsJson;
                try {
                    pluginsJson = JSON.stringify(plugins); // attempt to parse plugins to JSON
                } catch (error) {
                    console.error('Error converting plugins to JSON:', error);
                }

                if (pluginsJson) { // only store the data if pluginsJson is not null or undefined
                    await setStorageData('plugins', pluginsJson)
                }

                // refresh the display of the plugins table
                initPluginTable(plugins)
                if (pluginsLoaded && pluginsLoaded === false) {
                    loadPlugins(plugins)
                }
            }
        }
    }
    else {
        if (refreshPlugins) {
            showPluginToast('Plugins have been refreshed recently, refresh will be available in ' + convertSeconds(getTimeUntilNextRefresh()), 5000, true, false)
        }
    }
    initPluginsInProgress = false
}

function updateMetaTagPlugins(plugin) {
    // Update the meta tag with the list of loaded plugins
    let metaTag = document.querySelector('meta[name="plugins"]');
    if (metaTag === null) {
        metaTag = document.createElement('meta');
        metaTag.name = 'plugins';
        document.head.appendChild(metaTag);
    }
    const pluginArray = [...(metaTag.content ? metaTag.content.split(',') : []), plugin.id];
    metaTag.content = pluginArray.join(',');
}

function updateCompatIssueIds() {
    // Loop through each plugin
    plugins.forEach(plugin => {
        // Check if the plugin has `compatIssueIds` property
        if (plugin.compatIssueIds !== undefined) {
            // Loop through each of the `compatIssueIds`
            plugin.compatIssueIds.forEach(issueId => {
                // Find the plugin with the corresponding `issueId`
                const issuePlugin = plugins.find(p => p.id === issueId);
                // If the corresponding plugin is found, initialize its `compatIssueIds` property with an empty array if it's undefined
                if (issuePlugin) {
                    if (issuePlugin.compatIssueIds === undefined) {
                        issuePlugin.compatIssueIds = [];
                    }
                    // If the current plugin's ID is not already in the `compatIssueIds` array, add it
                    if (!issuePlugin.compatIssueIds.includes(plugin.id)) {
                        issuePlugin.compatIssueIds.push(plugin.id);
                    }
                }
            });
        } else {
            // If the plugin doesn't have `compatIssueIds` property, initialize it with an empty array
            plugin.compatIssueIds = [];
        }
    });
}

function deduplicatePluginsById(plugins) {
    const seenIds = new Set();
    const deduplicatedPlugins = [];

    for (const plugin of plugins) {
        if (!seenIds.has(plugin.id)) {
            seenIds.add(plugin.id);
            deduplicatedPlugins.push(plugin);
        } else {
            // favor dupes that have enabled == true
            const index = deduplicatedPlugins.findIndex(p => p.id === plugin.id);
            if (index >= 0) {
                if (plugin.enabled) {
                    deduplicatedPlugins[index] = plugin;
                }
            }
        }
    }

    return deduplicatedPlugins;
}

async function loadPlugins(plugins) {
    for (let i = 0; i < plugins.length; i++) {
        const plugin = plugins[i];
        if (plugin.enabled === true && plugin.localInstallOnly !== true) {
            const localPluginFound = checkFileNameInArray(localPlugins, plugin.url);
            if (!localPluginFound) {
                try {
                    // Indirect eval to work around sloppy plugin implementations
                    const indirectEval = { eval };
                    console.log("Loading plugin " + plugin.name);
                    indirectEval.eval(plugin.code);
                    console.log("Plugin " + plugin.name + " loaded");
                    await updateMetaTagPlugins(plugin); // add plugin to the meta tag
                } catch (err) {
                    showPluginToast("Error loading plugin " + plugin.name + " (" + err.message + ")", null, true);
                    console.error("Error loading plugin " + plugin.name + ": " + err.message);
                }
            } else {
                console.log("Skipping plugin " + plugin.name + " (installed locally)");
            }
        }
    }
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

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}, url: ${apiUrl}`);
        }
        const data = await response.json();
        return data.sha;
    } catch (error) {
        console.error('Error fetching data from url:', apiUrl, 'Error:', error);
        return null;
    }
}

// only allow two refresh per hour
function getTimeUntilNextRefresh() {
    const lastRuns = JSON.parse(localStorage.getItem('lastRuns') || '[]');
    const currentTime = new Date().getTime();
    const numRunsLast60Min = lastRuns.filter(run => currentTime - run <= 60 * 60 * 1000).length;

    if (numRunsLast60Min >= 2) {
        return 3600 - Math.round((currentTime - lastRuns[lastRuns.length - 1]) / 1000);
    }

    return 0;
}

function refreshAllowed() {
    const timeUntilNextRefresh = getTimeUntilNextRefresh();

    if (timeUntilNextRefresh > 0) {
        console.log(`Next refresh available in ${timeUntilNextRefresh} seconds`);
        return false;
    }

    const lastRuns = JSON.parse(localStorage.getItem('lastRuns') || '[]');
    const currentTime = new Date().getTime();
    lastRuns.push(currentTime);
    localStorage.setItem('lastRuns', JSON.stringify(lastRuns));
    return true;
}

async function downloadPlugins(pluginCatalog, plugins, refreshPlugins) {
    // download the plugins as needed
    for (const plugin of pluginCatalog) {
        //console.log(plugin.id, plugin.url)
        const existingPlugin = plugins.find(p => p.id === plugin.id);
        // get the file hash in the GitHub repo
        let sha
        if (isGitHub(plugin.url) && existingPlugin?.enabled === true) {
            sha = await getFileHash(plugin.url)
        }
        if (plugin.localInstallOnly !== true && isGitHub(plugin.url) && existingPlugin?.enabled === true && (refreshPlugins || (existingPlugin.sha !== undefined && existingPlugin.sha !== null && existingPlugin.sha !== sha) || existingPlugin?.code === undefined)) {
            const pluginSource = await getDocument(plugin.url);
            if (pluginSource !== null && pluginSource !== existingPlugin.code) {
                console.log(`Plugin ${plugin.name} updated`);
                showPluginToast("Plugin " + plugin.name + " updated", 5000);
                // Update the corresponding plugin
                const updatedPlugin = {
                    ...existingPlugin,
                    icon: plugin.icon ? plugin.icon : "fa-puzzle-piece",
                    id: plugin.id,
                    name: plugin.name,
                    description: plugin.description,
                    url: plugin.url,
                    localInstallOnly: Boolean(plugin.localInstallOnly),
                    version: plugin.version,
                    code: pluginSource,
                    author: plugin.author,
                    sha: sha,
                    compatIssueIds: plugin.compatIssueIds
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
                localInstallOnly: Boolean(plugin.localInstallOnly),
                version: plugin.version,
                author: plugin.author,
                compatIssueIds: plugin.compatIssueIds
            };
            // Replace the old plugin in the plugins array
            const pluginIndex = plugins.indexOf(existingPlugin);
            plugins.splice(pluginIndex, 1, updatedPlugin);
        }
        else {
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
        showPluginToast("Couldn't fetch " + extractFilename(url) + " (" + error + ")", null, true);
        console.error(error);
        return null;
    }
}

/* MODAL DIALOG */
const pluginDialogDialog = document.createElement("div");
pluginDialogDialog.id = "pluginDialog-input-dialog";
pluginDialogDialog.style.display = "none";

pluginDialogDialog.innerHTML = `
    <div class="pluginDialog-dialog-overlay"></div>
    <div class="pluginDialog-dialog-box">
        <div class="pluginDialog-dialog-header">
            <h2>Paste the plugin's code here</h2>
            <button class="pluginDialog-dialog-close-button">&times;</button>
        </div>
        <div class="pluginDialog-dialog-content">
            <textarea id="pluginDialog-input-textarea" spellcheck="false" autocomplete="off"></textarea>
        </div>
        <div class="pluginDialog-dialog-buttons">
            <button id="pluginDialog-input-ok">OK</button>
            <button id="pluginDialog-input-cancel">Cancel</button>
        </div>
    </div>
`;

document.body.appendChild(pluginDialogDialog);

const pluginDialogOverlay = document.querySelector(".pluginDialog-dialog-overlay");
const pluginDialogOkButton = document.getElementById("pluginDialog-input-ok");
const pluginDialogCancelButton = document.getElementById("pluginDialog-input-cancel");
const pluginDialogCloseButton = document.querySelector(".pluginDialog-dialog-close-button");
const pluginDialogTextarea = document.getElementById("pluginDialog-input-textarea");
let callbackOK
let callbackCancel

function pluginDialogOpenDialog(inputOK, inputCancel) {
    pluginDialogDialog.style.display = "block";
    callbackOK = inputOK
    callbackCancel = inputCancel
}

function pluginDialogCloseDialog() {
    pluginDialogDialog.style.display = "none";
}

function pluginDialogHandleOkClick() {
    const userInput = pluginDialogTextarea.value;
    // Do something with the user input
    callbackOK()
    pluginDialogCloseDialog();
}

function pluginDialogHandleCancelClick() {
    callbackCancel()
    pluginDialogCloseDialog();
}

function pluginDialogHandleOverlayClick(event) {
    if (event.target === pluginDialogOverlay) {
        pluginDialogCloseDialog();
    }
}

function pluginDialogHandleKeyDown(event) {
    if ((event.key === "Enter" && event.ctrlKey) || event.key === "Escape") {
        event.preventDefault();
        if (event.key === "Enter" && event.ctrlKey) {
            pluginDialogHandleOkClick();
        } else {
            pluginDialogCloseDialog();
        }
    }
}

pluginDialogTextarea.addEventListener("keydown", pluginDialogHandleKeyDown);
pluginDialogOkButton.addEventListener("click", pluginDialogHandleOkClick);
pluginDialogCancelButton.addEventListener("click", pluginDialogHandleCancelClick);
pluginDialogCloseButton.addEventListener("click", pluginDialogCloseDialog);
pluginDialogOverlay.addEventListener("click", pluginDialogHandleOverlayClick);
