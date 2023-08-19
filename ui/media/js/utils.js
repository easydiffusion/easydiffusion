"use strict"

// https://gomakethings.com/finding-the-next-and-previous-sibling-elements-that-match-a-selector-with-vanilla-js/
function getNextSibling(elem, selector) {
    // Get the next sibling element
    let sibling = elem.nextElementSibling

    // If there's no selector, return the first sibling
    if (!selector) {
        return sibling
    }

    // If the sibling matches our selector, use it
    // If not, jump to the next sibling and continue the loop
    while (sibling) {
        if (sibling.matches(selector)) {
            return sibling
        }
        sibling = sibling.nextElementSibling
    }
}

/* Panel Stuff */

// true = open
let COLLAPSIBLES_INITIALIZED = false
const COLLAPSIBLES_KEY = "collapsibles"
const COLLAPSIBLE_PANELS = [] // filled in by createCollapsibles with all the elements matching .collapsible

// on-init call this for any panels that are marked open
function toggleCollapsible(element) {
    const collapsibleHeader = element.querySelector(".collapsible")
    const handle = element.querySelector(".collapsible-handle")
    collapsibleHeader.classList.toggle("active")
    let content = getNextSibling(collapsibleHeader, ".collapsible-content")
    if (!collapsibleHeader.classList.contains("active")) {
        content.style.display = "none"
        if (handle != null) {
            // render results don't have a handle
            handle.innerHTML = "&#x2795;" // plus
        }
    } else {
        content.style.display = "block"
        if (handle != null) {
            // render results don't have a handle
            handle.innerHTML = "&#x2796;" // minus
        }
    }
    document.dispatchEvent(new CustomEvent("collapsibleClick", { detail: collapsibleHeader }))

    if (COLLAPSIBLES_INITIALIZED && COLLAPSIBLE_PANELS.includes(element)) {
        saveCollapsibles()
    }
}

function saveCollapsibles() {
    let values = {}
    COLLAPSIBLE_PANELS.forEach((element) => {
        let value = element.querySelector(".collapsible").className.indexOf("active") !== -1
        values[element.id] = value
    })
    localStorage.setItem(COLLAPSIBLES_KEY, JSON.stringify(values))
}

function createCollapsibles(node) {
    let save = false
    if (!node) {
        node = document
        save = true
    }
    let collapsibles = node.querySelectorAll(".collapsible")
    collapsibles.forEach(function(c) {
        if (save && c.parentElement.id) {
            COLLAPSIBLE_PANELS.push(c.parentElement)
        }
        let handle = document.createElement("span")
        handle.className = "collapsible-handle"

        if (c.classList.contains("active")) {
            handle.innerHTML = "&#x2796;" // minus
        } else {
            handle.innerHTML = "&#x2795;" // plus
        }
        c.insertBefore(handle, c.firstChild)

        c.addEventListener("click", function() {
            toggleCollapsible(c.parentElement)
        })
    })
    if (save) {
        let saved = localStorage.getItem(COLLAPSIBLES_KEY)
        if (!saved) {
            saved = tryLoadOldCollapsibles()
        }
        if (!saved) {
            saveCollapsibles()
            saved = localStorage.getItem(COLLAPSIBLES_KEY)
        }
        let values = JSON.parse(saved)
        COLLAPSIBLE_PANELS.forEach((element) => {
            let value = element.querySelector(".collapsible").className.indexOf("active") !== -1
            if (values[element.id] != value) {
                toggleCollapsible(element)
            }
        })
        COLLAPSIBLES_INITIALIZED = true
    }
}

function tryLoadOldCollapsibles() {
    const old_map = {
        advancedPanelOpen: "editor-settings",
        modifiersPanelOpen: "editor-modifiers",
        negativePromptPanelOpen: "editor-inputs-prompt",
    }
    if (localStorage.getItem(Object.keys(old_map)[0])) {
        let result = {}
        Object.keys(old_map).forEach((key) => {
            const value = localStorage.getItem(key)
            if (value !== null) {
                result[old_map[key]] = value == true || value == "true"
                localStorage.removeItem(key)
            }
        })
        result = JSON.stringify(result)
        localStorage.setItem(COLLAPSIBLES_KEY, result)
        return result
    }
    return null
}

function collapseAll(selector) {
    const collapsibleElems = document.querySelectorAll(selector); // needs to have ";"

    [...collapsibleElems].forEach((elem) => {
        const isActive =  elem.classList.contains("active")

        if(isActive) {
            elem?.click()
        }
    })
}

function expandAll(selector) {
    const collapsibleElems = document.querySelectorAll(selector); // needs to have ";"

    [...collapsibleElems].forEach((elem) => {
        const isActive =  elem.classList.contains("active")

        if (!isActive) {
            elem?.click()
        }
    })
}


function permute(arr) {
    let permutations = []
    let n = arr.length
    let n_permutations = Math.pow(2, n)
    for (let i = 0; i < n_permutations; i++) {
        let perm = []
        let mask = Number(i)
            .toString(2)
            .padStart(n, "0")

        for (let idx = 0; idx < mask.length; idx++) {
            if (mask[idx] === "1" && arr[idx].trim() !== "") {
                perm.push(arr[idx])
            }
        }

        if (perm.length > 0) {
            permutations.push(perm)
        }
    }

    return permutations
}

function permuteNumber(arr) {
    return Math.pow(2, arr.length)
}

// https://stackoverflow.com/a/8212878
function millisecondsToStr(milliseconds) {
    function numberEnding(number) {
        return number > 1 ? "s" : ""
    }

    let temp = Math.floor(milliseconds / 1000)
    let hours = Math.floor((temp %= 86400) / 3600)
    let s = ""
    if (hours) {
        s += hours + " hour" + numberEnding(hours) + " "
    }
    let minutes = Math.floor((temp %= 3600) / 60)
    if (minutes) {
        s += minutes + " minute" + numberEnding(minutes) + " "
    }
    let seconds = temp % 60
    if (!hours && minutes < 4 && seconds) {
        s += seconds + " second" + numberEnding(seconds)
    }

    return s
}

// https://rosettacode.org/wiki/Brace_expansion#JavaScript
function BraceExpander() {
    "use strict"

    // Index of any closing brace matching the opening
    // brace at iPosn,
    // with the indices of any immediately-enclosed commas.
    function bracePair(tkns, iPosn, iNest, lstCommas) {
        if (iPosn >= tkns.length || iPosn < 0) return null

        let t = tkns[iPosn],
            n = t === "{" ? iNest + 1 : t === "}" ? iNest - 1 : iNest,
            lst = t === "," && iNest === 1 ? lstCommas.concat(iPosn) : lstCommas

        return n
            ? bracePair(tkns, iPosn + 1, n, lst)
            : {
                  close: iPosn,
                  commas: lst,
              }
    }

    // Parse of a SYNTAGM subtree
    function andTree(dctSofar, tkns) {
        if (!tkns.length) return [dctSofar, []]

        let dctParse = dctSofar
                ? dctSofar
                : {
                      fn: and,
                      args: [],
                  },
            head = tkns[0],
            tail = head ? tkns.slice(1) : [],
            dctBrace = head === "{" ? bracePair(tkns, 0, 0, []) : null,
            lstOR = dctBrace && dctBrace.close && dctBrace.commas.length ? splitAt(dctBrace.close + 1, tkns) : null

        return andTree(
            {
                fn: and,
                args: dctParse.args.concat(lstOR ? orTree(dctParse, lstOR[0], dctBrace.commas) : head),
            },
            lstOR ? lstOR[1] : tail
        )
    }

    // Parse of a PARADIGM subtree
    function orTree(dctSofar, tkns, lstCommas) {
        if (!tkns.length) return [dctSofar, []]
        let iLast = lstCommas.length

        return {
            fn: or,
            args: splitsAt(lstCommas, tkns)
                .map(function(x, i) {
                    let ts = x.slice(1, i === iLast ? -1 : void 0)

                    return ts.length ? ts : [""]
                })
                .map(function(ts) {
                    return ts.length > 1 ? andTree(null, ts)[0] : ts[0]
                }),
        }
    }

    // List of unescaped braces and commas, and remaining strings
    function tokens(str) {
        // Filter function excludes empty splitting artefacts
        let toS = function(x) {
            return x.toString()
        }

        return str
            .split(/(\\\\)/)
            .filter(toS)
            .reduce(function(a, s) {
                return a.concat(s.charAt(0) === "\\" ? s : s.split(/(\\*[{,}])/).filter(toS))
            }, [])
    }

    // PARSE TREE OPERATOR (1 of 2)
    // Each possible head * each possible tail
    function and(args) {
        let lng = args.length,
            head = lng ? args[0] : null,
            lstHead = "string" === typeof head ? [head] : head

        return lng
            ? 1 < lng
                ? lstHead.reduce(function(a, h) {
                      return a.concat(
                          and(args.slice(1)).map(function(t) {
                              return h + t
                          })
                      )
                  }, [])
                : lstHead
            : []
    }

    // PARSE TREE OPERATOR (2 of 2)
    // Each option flattened
    function or(args) {
        return args.reduce(function(a, b) {
            return a.concat(b)
        }, [])
    }

    // One list split into two (first sublist length n)
    function splitAt(n, lst) {
        return n < lst.length + 1 ? [lst.slice(0, n), lst.slice(n)] : [lst, []]
    }

    // One list split into several (sublist lengths [n])
    function splitsAt(lstN, lst) {
        return lstN.reduceRight(
            function(a, x) {
                return splitAt(x, a[0]).concat(a.slice(1))
            },
            [lst]
        )
    }

    // Value of the parse tree
    function evaluated(e) {
        return typeof e === "string" ? e : e.fn(e.args.map(evaluated))
    }

    // JSON prettyprint (for parse tree, token list etc)
    function pp(e) {
        return JSON.stringify(
            e,
            function(k, v) {
                return typeof v === "function" ? "[function " + v.name + "]" : v
            },
            2
        )
    }

    // ----------------------- MAIN ------------------------

    // s -> [s]
    this.expand = function(s) {
        // BRACE EXPRESSION PARSED
        let dctParse = andTree(null, tokens(s))[0]

        // ABSTRACT SYNTAX TREE LOGGED
        // console.log(pp(dctParse));

        // AST EVALUATED TO LIST OF STRINGS
        return evaluated(dctParse)
    }
}

/** Pause the execution of an async function until timer elapse.
 * @Returns a promise that will resolve after the specified timeout.
 */
function asyncDelay(timeout) {
    return new Promise(function(resolve, reject) {
        setTimeout(resolve, timeout, true)
    })
}

function PromiseSource() {
    const srcPromise = new Promise((resolve, reject) => {
        Object.defineProperties(this, {
            resolve: { value: resolve, writable: false },
            reject: { value: reject, writable: false },
        })
    })
    Object.defineProperties(this, {
        promise: { value: makeQuerablePromise(srcPromise), writable: false },
    })
}

/** A debounce is a higher-order function, which is a function that returns another function
 * that, as long as it continues to be invoked, will not be triggered.
 * The function will be called after it stops being called for N milliseconds.
 * If `immediate` is passed, trigger the function on the leading edge, instead of the trailing.
 * @Returns a promise that will resolve to func return value.
 */
function debounce(func, wait, immediate) {
    if (typeof wait === "undefined") {
        wait = 40
    }
    if (typeof wait !== "number") {
        throw new Error("wait is not an number.")
    }
    let timeout = null
    let lastPromiseSrc = new PromiseSource()
    const applyFn = function(context, args) {
        let result = undefined
        try {
            result = func.apply(context, args)
        } catch (err) {
            lastPromiseSrc.reject(err)
        }
        if (result instanceof Promise) {
            result.then(lastPromiseSrc.resolve, lastPromiseSrc.reject)
        } else {
            lastPromiseSrc.resolve(result)
        }
    }
    return function(...args) {
        const callNow = Boolean(immediate && !timeout)
        const context = this
        if (timeout) {
            clearTimeout(timeout)
        }
        timeout = setTimeout(function() {
            if (!immediate) {
                applyFn(context, args)
            }
            lastPromiseSrc = new PromiseSource()
            timeout = null
        }, wait)
        if (callNow) {
            applyFn(context, args)
        }
        return lastPromiseSrc.promise
    }
}

function preventNonNumericalInput(e) {
    e = e || window.event
    const charCode = typeof e.which == "undefined" ? e.keyCode : e.which
    const charStr = String.fromCharCode(charCode)
    const newInputValue = `${e.target.value}${charStr}`
    const re = new RegExp(e.target.getAttribute("pattern") || "^[0-9]+$")

    if (!re.test(charStr) && !re.test(newInputValue)) {
        e.preventDefault()
    }
}

/** Returns the global object for the current execution environement.
 * @Returns window in a browser, global in node and self in a ServiceWorker.
 * @Notes Allows unit testing and use of the engine outside of a browser.
 */
function getGlobal() {
    if (typeof globalThis === "object") {
        return globalThis
    } else if (typeof global === "object") {
        return global
    } else if (typeof self === "object") {
        return self
    }
    try {
        return Function("return this")()
    } catch {
        // If the Function constructor fails, we're in a browser with eval disabled by CSP headers.
        return window
    } // Returns undefined if global can't be found.
}

/** Check if x is an Array or a TypedArray.
 * @Returns true if x is an Array or a TypedArray, false otherwise.
 */
function isArrayOrTypedArray(x) {
    return Boolean(typeof x === "object" && (Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView))))
}

function makeQuerablePromise(promise) {
    if (typeof promise !== "object") {
        throw new Error("promise is not an object.")
    }
    if (!(promise instanceof Promise)) {
        throw new Error("Argument is not a promise.")
    }
    // Don't modify a promise that's been already modified.
    if ("isResolved" in promise || "isRejected" in promise || "isPending" in promise) {
        return promise
    }
    let isPending = true
    let isRejected = false
    let rejectReason = undefined
    let isResolved = false
    let resolvedValue = undefined
    const qurPro = promise.then(
        function(val) {
            isResolved = true
            isPending = false
            resolvedValue = val
            return val
        },
        function(reason) {
            rejectReason = reason
            isRejected = true
            isPending = false
            throw reason
        }
    )
    Object.defineProperties(qurPro, {
        isResolved: {
            get: () => isResolved,
        },
        resolvedValue: {
            get: () => resolvedValue,
        },
        isPending: {
            get: () => isPending,
        },
        isRejected: {
            get: () => isRejected,
        },
        rejectReason: {
            get: () => rejectReason,
        },
    })
    return qurPro
}

/* inserts custom html to allow prettifying of inputs */
function prettifyInputs(root_element) {
    root_element.querySelectorAll(`input[type="checkbox"]`).forEach((element) => {
        if (element.style.display === "none") {
            return
        }
        var parent = element.parentNode
        if (!parent.classList.contains("input-toggle")) {
            var wrapper = document.createElement("div")
            wrapper.classList.add("input-toggle")
            parent.replaceChild(wrapper, element)
            wrapper.appendChild(element)
            var label = document.createElement("label")
            label.htmlFor = element.id
            wrapper.appendChild(label)
        }
    })
}

class GenericEventSource {
    #events = {}
    #types = []
    constructor(...eventsTypes) {
        if (Array.isArray(eventsTypes) && eventsTypes.length === 1 && Array.isArray(eventsTypes[0])) {
            eventsTypes = eventsTypes[0]
        }
        this.#types.push(...eventsTypes)
    }
    get eventTypes() {
        return this.#types
    }
    /** Add a new event listener
     */
    addEventListener(name, handler) {
        if (!this.#types.includes(name)) {
            throw new Error("Invalid event name.")
        }
        if (this.#events.hasOwnProperty(name)) {
            this.#events[name].push(handler)
        } else {
            this.#events[name] = [handler]
        }
    }
    /** Remove the event listener
     */
    removeEventListener(name, handler) {
        if (!this.#events.hasOwnProperty(name)) {
            return
        }
        const index = this.#events[name].indexOf(handler)
        if (index != -1) {
            this.#events[name].splice(index, 1)
        }
    }
    fireEvent(name, ...args) {
        if (!this.#types.includes(name)) {
            throw new Error(`Event ${String(name)} missing from Events.types`)
        }
        if (!this.#events.hasOwnProperty(name)) {
            return Promise.resolve()
        }
        if (!args || !args.length) {
            args = []
        }
        const evs = this.#events[name]
        if (evs.length <= 0) {
            return Promise.resolve()
        }
        return Promise.allSettled(
            evs.map((callback) => {
                try {
                    return Promise.resolve(callback.apply(SD, args))
                } catch (ex) {
                    return Promise.reject(ex)
                }
            })
        )
    }
}

class ServiceContainer {
    #services = new Map()
    #singletons = new Map()
    constructor(...servicesParams) {
        servicesParams.forEach(this.register.bind(this))
    }
    get services() {
        return this.#services
    }
    get singletons() {
        return this.#singletons
    }
    register(params) {
        if (ServiceContainer.isConstructor(params)) {
            if (typeof params.name !== "string") {
                throw new Error("params.name is not a string.")
            }
            params = { name: params.name, definition: params }
        }
        if (typeof params !== "object") {
            throw new Error("params is not an object.")
        }
        ;["name", "definition"].forEach((key) => {
            if (!(key in params)) {
                console.error("Invalid service %o registration.", params)
                throw new Error(`params.${key} is not defined.`)
            }
        })
        const opts = { definition: params.definition }
        if ("dependencies" in params) {
            if (Array.isArray(params.dependencies)) {
                params.dependencies.forEach((dep) => {
                    if (typeof dep !== "string") {
                        throw new Error("dependency name is not a string.")
                    }
                })
                opts.dependencies = params.dependencies
            } else {
                throw new Error("params.dependencies is not an array.")
            }
        }
        if (params.singleton) {
            opts.singleton = true
        }
        this.#services.set(params.name, opts)
        return Object.assign({ name: params.name }, opts)
    }
    get(name) {
        const ctorInfos = this.#services.get(name)
        if (!ctorInfos) {
            return
        }
        if (!ServiceContainer.isConstructor(ctorInfos.definition)) {
            return ctorInfos.definition
        }
        if (!ctorInfos.singleton) {
            return this._createInstance(ctorInfos)
        }
        const singletonInstance = this.#singletons.get(name)
        if (singletonInstance) {
            return singletonInstance
        }
        const newSingletonInstance = this._createInstance(ctorInfos)
        this.#singletons.set(name, newSingletonInstance)
        return newSingletonInstance
    }

    _getResolvedDependencies(service) {
        let classDependencies = []
        if (service.dependencies) {
            classDependencies = service.dependencies.map(this.get.bind(this))
        }
        return classDependencies
    }

    _createInstance(service) {
        if (!ServiceContainer.isClass(service.definition)) {
            // Call as normal function.
            return service.definition(...this._getResolvedDependencies(service))
        }
        // Use new
        return new service.definition(...this._getResolvedDependencies(service))
    }

    static isClass(definition) {
        return (
            typeof definition === "function" &&
            Boolean(definition.prototype) &&
            definition.prototype.constructor === definition
        )
    }
    static isConstructor(definition) {
        return typeof definition === "function"
    }
}

/**
 *
 * @param {string} tag
 * @param {object} attributes
 * @param {string | Array<string>} classes
 * @param {string | Node | Array<string | Node>}
 * @returns {HTMLElement}
 */
function createElement(tagName, attributes, classes, textOrElements) {
    const element = document.createElement(tagName)
    if (attributes) {
        Object.entries(attributes).forEach(([key, value]) => {
            if (value !== undefined && value !== null) {
                element.setAttribute(key, value)
            }
        })
    }
    if (classes) {
        ;(Array.isArray(classes) ? classes : [classes]).forEach((className) => element.classList.add(className))
    }
    if (textOrElements) {
        const children = Array.isArray(textOrElements) ? textOrElements : [textOrElements]
        children.forEach((textOrElem) => {
            if (textOrElem instanceof Node) {
                element.appendChild(textOrElem)
            } else {
                element.appendChild(document.createTextNode(textOrElem))
            }
        })
    }
    return element
}

/**
 * Add a listener for arrays
 * @param {keyof Array} method
 * @param {(args) => {}} callback
 */
Array.prototype.addEventListener = function(method, callback) {
    const originalFunction = this[method]
    if (originalFunction) {
        this[method] = function() {
            originalFunction.apply(this, arguments)
            callback.apply(this, arguments)
        }
    }
}

/**
 * @typedef {object} TabOpenDetails
 * @property {HTMLElement} contentElement
 * @property {HTMLElement} labelElement
 * @property {number} timesOpened
 * @property {boolean} firstOpen
 */

/**
 * @typedef {object} CreateTabRequest
 * @property {string} id
 * @property {string | Node | (() => (string | Node))} label
 * Label text or an HTML element
 * @property {string} icon
 * @property {string | Node | Promise<string | Node> | (() => (string | Node | Promise<string | Node>)) | undefined} content
 * HTML string or HTML element
 * @property {((TabOpenDetails, Event) => (undefined | string | Node | Promise<string | Node>)) | undefined} onOpen
 * If an HTML string or HTML element is returned, then that will replace the tab content
 * @property {string | undefined} css
 */

/**
 * @param {CreateTabRequest} request
 */
function createTab(request) {
    if (!request?.id) {
        console.error("createTab() error - id is required", Error().stack)
        return
    }

    if (!request.label) {
        console.error("createTab() error - label is required", Error().stack)
        return
    }

    if (!request.icon) {
        console.error("createTab() error - icon is required", Error().stack)
        return
    }

    if (!request.content && !request.onOpen) {
        console.error("createTab() error - content or onOpen required", Error().stack)
        return
    }

    const tabsContainer = document.querySelector(".tab-container")
    if (!tabsContainer) {
        return
    }

    const tabsContentWrapper = document.querySelector("#tab-content-wrapper")
    if (!tabsContentWrapper) {
        return
    }

    // console.debug('creating tab: ', request)

    if (request.css) {
        document
            .querySelector("body")
            .insertAdjacentElement(
                "beforeend",
                createElement("style", { id: `tab-${request.id}-css` }, undefined, request.css)
            )
    }

    const label = typeof request.label === "function" ? request.label() : request.label
    const labelElement = label instanceof Node ? label : createElement("span", undefined, undefined, label)

    const tab = createElement(
        "span",
        { id: `tab-${request.id}`, "data-times-opened": 0 },
        ["tab"],
        createElement("span", undefined, undefined, [
            createElement("i", { style: "margin-right: 0.25em" }, [
                "fa-solid",
                `${request.icon.startsWith("fa-") ? "" : "fa-"}${request.icon}`,
                "icon",
            ]),
            labelElement,
        ])
    )

    tabsContainer.insertAdjacentElement("beforeend", tab)

    const wrapper = createElement("div", { id: request.id }, ["tab-content-inner"], "Loading..")

    const tabContent = createElement("div", { id: `tab-content-${request.id}` }, ["tab-content"], wrapper)
    tabsContentWrapper.insertAdjacentElement("beforeend", tabContent)

    linkTabContents(tab)

    function replaceContent(resultFactory) {
        if (resultFactory === undefined || resultFactory === null) {
            return
        }
        const result = typeof resultFactory === "function" ? resultFactory() : resultFactory
        if (result instanceof Promise) {
            result.then(replaceContent)
        } else if (result instanceof Node) {
            wrapper.replaceChildren(result)
        } else {
            wrapper.innerHTML = result
        }
    }

    replaceContent(request.content)

    tab.addEventListener("click", (e) => {
        const timesOpened = +(tab.dataset.timesOpened || 0) + 1
        tab.dataset.timesOpened = timesOpened

        if (request.onOpen) {
            const result = request.onOpen(
                {
                    contentElement: wrapper,
                    labelElement,
                    timesOpened,
                    firstOpen: timesOpened === 1,
                },
                e
            )

            replaceContent(result)
        }
    })
}


/* TOAST NOTIFICATIONS */
function showToast(message, duration = 5000, error = false) {
    const toast = document.createElement("div")
    toast.classList.add("toast-notification")
    if (error === true) {
        toast.classList.add("toast-notification-error")
    }
    toast.innerHTML = message
    document.body.appendChild(toast)

    // Set the position of the toast on the screen
    const toastCount = document.querySelectorAll(".toast-notification").length
    const toastHeight = toast.offsetHeight
    const previousToastsHeight = Array.from(document.querySelectorAll(".toast-notification"))
        .slice(0, -1) // exclude current toast
        .reduce((totalHeight, toast) => totalHeight + toast.offsetHeight + 10, 0) // add 10 pixels for spacing
    toast.style.bottom = `${10 + previousToastsHeight}px`
    toast.style.right = "10px"

    // Delay the removal of the toast until animation has completed
    const removeToast = () => {
        toast.classList.add("hide")
        const removeTimeoutId = setTimeout(() => {
            toast.remove()
            // Adjust the position of remaining toasts
            const remainingToasts = document.querySelectorAll(".toast-notification")
            const removedToastBottom = toast.getBoundingClientRect().bottom

            remainingToasts.forEach((toast) => {
                if (toast.getBoundingClientRect().bottom < removedToastBottom) {
                    toast.classList.add("slide-down")
                }
            })

            // Wait for the slide-down animation to complete
            setTimeout(() => {
                // Remove the slide-down class after the animation has completed
                const slidingToasts = document.querySelectorAll(".slide-down")
                slidingToasts.forEach((toast) => {
                    toast.classList.remove("slide-down")
                })

                // Adjust the position of remaining toasts again, in case there are multiple toasts being removed at once
                const remainingToastsDown = document.querySelectorAll(".toast-notification")
                let heightSoFar = 0
                remainingToastsDown.forEach((toast) => {
                    toast.style.bottom = `${10 + heightSoFar}px`
                    heightSoFar += toast.offsetHeight + 10 // add 10 pixels for spacing
                })
            }, 0) // The duration of the slide-down animation (in milliseconds)
        }, 500)
    }

    // Remove the toast after specified duration
    setTimeout(removeToast, duration)
}

function alert(msg, title) {
    title = title || ""
    $.alert({
        theme: "modern",
        title: title,
        useBootstrap: false,
        animateFromElement: false,
        content: msg,
    })
}

function confirm(msg, title, fn) {
    title = title || ""
    $.confirm({
        theme: "modern",
        title: title,
        useBootstrap: false,
        animateFromElement: false,
        content: msg,
        buttons: {
            yes: fn,
            cancel: () => {},
        },
    })
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
        request.addEventListener("upgradeneeded", function () {
            let db = request.result;
            db.createObjectStore("EasyDiffusionSettings", { keyPath: "id" });
        });
        request.addEventListener("success", function () {
            resolve(request.result);
        });
        request.addEventListener("error", function () {
            reject(request.error);
        });
    });
}

// Function to write data to the object store
async function setStorageData(key, value) {
    return openDB().then(db => {
        let tx = db.transaction("EasyDiffusionSettings", "readwrite");
        let store = tx.objectStore("EasyDiffusionSettings");
        let data = { id: key, value: value };
        return new Promise((resolve, reject) => {
            let request = store.put(data);
            request.addEventListener("success", function () {
                resolve(request.result);
            });
            request.addEventListener("error", function () {
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
            request.addEventListener("success", function () {
                if (request.result) {
                    resolve(request.result.value);
                } else {
                    // entry not found
                    resolve();
                }
            });
            request.addEventListener("error", function () {
                reject(request.error);
            });
        });
    });
}

function insertAtCursor(field, text) {
    if (field.selectionStart || field.selectionStart == "0") {
        var startPos = field.selectionStart
        var endPos = field.selectionEnd
        var before = field.value.substring(0, startPos)
        var after = field.value.substring(endPos, field.value.length)

        if (!before.endsWith(" ")) { before += " " }
        if (!after.startsWith(" ")) { after = " "+after }

        field.value = before + text + after
    } else {
        field.value += text
    }
}

// indexedDB debug functions
async function getAllKeys() {
    return openDB().then(db => {
        let tx = db.transaction("EasyDiffusionSettings", "readonly");
        let store = tx.objectStore("EasyDiffusionSettings");
        let keys = [];
        return new Promise((resolve, reject) => {
            store.openCursor().onsuccess = function (event) {
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

async function logAllStorageKeys() {
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
                store.openCursor().onsuccess = function (event) {
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

/**
 * @param {String} Data URL of the image
 * @param {Integer} Top left X-coordinate of the crop area
 * @param {Integer} Top left Y-coordinate of the crop area
 * @param {Integer} Width of the crop area
 * @param {Integer} Height of the crop area
 * @return {String}
 */
function cropImageDataUrl(dataUrl, x, y, width, height) {
    return new Promise((resolve, reject) => {
        const image = new Image()
        image.src = dataUrl

        image.onload = () => {
            const canvas = document.createElement('canvas')
            canvas.width = width
            canvas.height = height

            const ctx = canvas.getContext('2d')
            ctx.drawImage(image, x, y, width, height, 0, 0, width, height)

            const croppedDataUrl = canvas.toDataURL('image/png')
            resolve(croppedDataUrl)
        }

        image.onerror = (error) => {
            reject(error)
        }
    })
}

/**
 * @param {String} HTML representing a single element
 * @return {Element}
 */
function htmlToElement(html) {
    var template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}

function modalDialogCloseOnBackdropClick(dialog) {
    dialog.addEventListener('mousedown', function (event) {
        // Firefox creates an event with clientX|Y = 0|0 when choosing an <option>.
        // Test whether the element interacted with is a child of the dialog, but not the
        // dialog itself (the backdrop would be a part of the dialog)
        if (dialog.contains(event.target) && dialog != event.target) {
            return
        }
        var rect = dialog.getBoundingClientRect()
        var isInDialog=(rect.top <= event.clientY && event.clientY <= rect.top + rect.height
          && rect.left <= event.clientX && event.clientX <= rect.left + rect.width)
        if (!isInDialog) {
            dialog.close()
        }
    })
}

function makeDialogDraggable(element) {
    element.querySelector(".dialog-header").addEventListener('mousedown', (function() {
        let deltaX=0
        let deltaY=0
        let dragStartX=0
        let dragStartY=0
        let oldTop=0
        let oldLeft=0

        function dlgDragStart(e) {
            e = e || window.event;
            const d = e.target.closest("dialog")
            e.preventDefault();
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            oldTop = parseInt(d.style.top)
            oldLeft = parseInt(d.style.left)
            if (isNaN(oldTop)) { oldTop=0 }
            if (isNaN(oldLeft)) { oldLeft=0 }
            document.addEventListener('mouseup', dlgDragClose);
            document.addEventListener('mousemove', dlgDrag);
        }

        function dlgDragClose(e) {
            document.removeEventListener('mouseup', dlgDragClose);
            document.removeEventListener('mousemove', dlgDrag);
        }

        function dlgDrag(e) {
            e = e || window.event;
            const d = e.target.closest("dialog")
            e.preventDefault();
            deltaX = dragStartX - e.clientX;
            deltaY = dragStartY - e.clientY;
            d.style.left = `${oldLeft-2*deltaX}px`
            d.style.top  = `${oldTop-2*deltaY}px`
        }

        return dlgDragStart
    })() )
}

function logMsg(msg, level, outputMsg) {
    if (outputMsg.hasChildNodes()) {
        outputMsg.appendChild(document.createElement("br"))
    }
    if (level === "error") {
        outputMsg.innerHTML += '<span style="color: red">Error: ' + msg + "</span>"
    } else if (level === "warn") {
        outputMsg.innerHTML += '<span style="color: orange">Warning: ' + msg + "</span>"
    } else {
        outputMsg.innerText += msg
    }
    console.log(level, msg)
}

function logError(msg, res, outputMsg) {
    logMsg(msg, "error", outputMsg)

    console.log("request error", res)
    console.trace()
    // setStatus("request", "error", "error")
}

function playSound() {
    const audio = new Audio("/media/ding.mp3")
    audio.volume = 0.2
    var promise = audio.play()
    if (promise !== undefined) {
        promise
            .then((_) => {})
            .catch((error) => {
                console.warn("browser blocked autoplay")
            })
    }
}
