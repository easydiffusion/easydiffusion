// https://gomakethings.com/finding-the-next-and-previous-sibling-elements-that-match-a-selector-with-vanilla-js/
function getNextSibling(elem, selector) {
	// Get the next sibling element
	var sibling = elem.nextElementSibling

	// If there's no selector, return the first sibling
	if (!selector) return sibling

	// If the sibling matches our selector, use it
	// If not, jump to the next sibling and continue the loop
	while (sibling) {
		if (sibling.matches(selector)) return sibling
		sibling = sibling.nextElementSibling
	}
}



/* Panel Stuff */

// true = open
var COLLAPSIBLES_INITIALIZED = false;
const COLLAPSIBLES_KEY = "collapsibles";
const COLLAPSIBLE_PANELS = []; // filled in by createCollapsibles with all the elements matching .collapsible

// on-init call this for any panels that are marked open
function toggleCollapsible(element) {
    var collapsibleHeader = element.querySelector(".collapsible");
    var handle = element.querySelector(".collapsible-handle");
    collapsibleHeader.classList.toggle("active")
    let content = getNextSibling(collapsibleHeader, '.collapsible-content')
    if (!collapsibleHeader.classList.contains("active")) {
        content.style.display = "none"
        if (handle != null) {  // render results don't have a handle
            handle.innerHTML = '&#x2795;' // plus
        }
    } else {
        content.style.display = "block"
        if (handle != null) {  // render results don't have a handle
            handle.innerHTML = '&#x2796;' // minus
        }
    }
    
    if (COLLAPSIBLES_INITIALIZED && COLLAPSIBLE_PANELS.includes(element)) {
        saveCollapsibles()
    }
}

function saveCollapsibles() {
    var values = {}
    COLLAPSIBLE_PANELS.forEach(element => {
        var value = element.querySelector(".collapsible").className.indexOf("active") !== -1
        values[element.id] = value
    })
    localStorage.setItem(COLLAPSIBLES_KEY, JSON.stringify(values))
}

function createCollapsibles(node) {
    var save = false
    if (!node) {
        node = document
        save = true
    }
    let collapsibles = node.querySelectorAll(".collapsible")
    collapsibles.forEach(function(c) {
        if (save && c.parentElement.id) {
            COLLAPSIBLE_PANELS.push(c.parentElement)
        }
        let handle = document.createElement('span')
        handle.className = 'collapsible-handle'

        if (c.classList.contains("active")) {
            handle.innerHTML = '&#x2796;' // minus
        } else {
            handle.innerHTML = '&#x2795;' // plus
        }
        c.insertBefore(handle, c.firstChild)

        c.addEventListener('click', function() {
            toggleCollapsible(c.parentElement)
        })
    })
    if (save) {
        var saved = localStorage.getItem(COLLAPSIBLES_KEY)
        if (!saved) { 
            saved = tryLoadOldCollapsibles();
        }
        if (!saved) {
            saveCollapsibles()
            saved = localStorage.getItem(COLLAPSIBLES_KEY)
        }
        var values = JSON.parse(saved)
        COLLAPSIBLE_PANELS.forEach(element => {
            var value = element.querySelector(".collapsible").className.indexOf("active") !== -1
            if (values[element.id] != value) {
                toggleCollapsible(element)
            }
        })
        COLLAPSIBLES_INITIALIZED = true
    }
}

function tryLoadOldCollapsibles() {
    var old_map = {
        "advancedPanelOpen": "editor-settings",
        "modifiersPanelOpen": "editor-modifiers",
        "negativePromptPanelOpen": "editor-inputs-prompt"
    };
    if (localStorage.getItem(Object.keys(old_map)[0])) {
        var result = {};
        Object.keys(old_map).forEach(key => {
            var value = localStorage.getItem(key);
            if (value !== null) {
                result[old_map[key]] = value == true || value == "true"
                localStorage.removeItem(key)
            }
        });
        result = JSON.stringify(result)
        localStorage.setItem(COLLAPSIBLES_KEY, result)
        return result
    }
    return null;
}

function permute(arr) {
    let permutations = []
    let n = arr.length
    let n_permutations = Math.pow(2, n)
    for (let i = 0; i < n_permutations; i++) {
        let perm = []
        let mask = Number(i).toString(2).padStart(n, '0')

        for (let idx = 0; idx < mask.length; idx++) {
            if (mask[idx] === '1' && arr[idx].trim() !== '') {
                perm.push(arr[idx])
            }
        }

        if (perm.length > 0) {
            permutations.push(perm)
        }
    }

    return permutations
}

// https://stackoverflow.com/a/8212878
function millisecondsToStr(milliseconds) {
    function numberEnding (number) {
        return (number > 1) ? 's' : ''
    }

    var temp = Math.floor(milliseconds / 1000)
    var hours = Math.floor((temp %= 86400) / 3600)
    var s = ''
    if (hours) {
        s += hours + ' hour' + numberEnding(hours) + ' '
    }
    var minutes = Math.floor((temp %= 3600) / 60)
    if (minutes) {
        s += minutes + ' minute' + numberEnding(minutes) + ' '
    }
    var seconds = temp % 60
    if (!hours && minutes < 4 && seconds) {
        s += seconds + ' second' + numberEnding(seconds)
    }

    return s
}

// https://rosettacode.org/wiki/Brace_expansion#JavaScript
function BraceExpander() {
    'use strict'

    // Index of any closing brace matching the opening
    // brace at iPosn,
    // with the indices of any immediately-enclosed commas.
    function bracePair(tkns, iPosn, iNest, lstCommas) {
        if (iPosn >= tkns.length || iPosn < 0) return null;

        var t = tkns[iPosn],
            n = (t === '{') ? (
                iNest + 1
            ) : (t === '}' ? (
                iNest - 1
            ) : iNest),
            lst = (t === ',' && iNest === 1) ? (
                lstCommas.concat(iPosn)
            ) : lstCommas;

        return n ? bracePair(tkns, iPosn + 1, n, lst) : {
            close: iPosn,
            commas: lst
        };
    }

    // Parse of a SYNTAGM subtree
    function andTree(dctSofar, tkns) {
        if (!tkns.length) return [dctSofar, []];

        var dctParse = dctSofar ? dctSofar : {
                fn: and,
                args: []
            },

            head = tkns[0],
            tail = head ? tkns.slice(1) : [],

            dctBrace = head === '{' ? bracePair(
                tkns, 0, 0, []
            ) : null,

            lstOR = dctBrace && (
                dctBrace.close
            ) && dctBrace.commas.length ? (
                splitAt(dctBrace.close + 1, tkns)
            ) : null;

        return andTree({
            fn: and,
            args: dctParse.args.concat(
                lstOR ? (
                    orTree(dctParse, lstOR[0], dctBrace.commas)
                ) : head
            )
        }, lstOR ? (
            lstOR[1]
        ) : tail);
    }

    // Parse of a PARADIGM subtree
    function orTree(dctSofar, tkns, lstCommas) {
        if (!tkns.length) return [dctSofar, []];
        var iLast = lstCommas.length;

        return {
            fn: or,
            args: splitsAt(
                lstCommas, tkns
            ).map(function (x, i) {
                var ts = x.slice(
                    1, i === iLast ? (
                        -1
                    ) : void 0
                );

                return ts.length ? ts : [''];
            }).map(function (ts) {
                return ts.length > 1 ? (
                    andTree(null, ts)[0]
                ) : ts[0];
            })
        };
    }

    // List of unescaped braces and commas, and remaining strings
    function tokens(str) {
        // Filter function excludes empty splitting artefacts
        var toS = function (x) {
            return x.toString();
        };

        return str.split(/(\\\\)/).filter(toS).reduce(function (a, s) {
            return a.concat(s.charAt(0) === '\\' ? s : s.split(
                /(\\*[{,}])/
            ).filter(toS));
        }, []);
    }

    // PARSE TREE OPERATOR (1 of 2)
    // Each possible head * each possible tail
    function and(args) {
        var lng = args.length,
            head = lng ? args[0] : null,
            lstHead = "string" === typeof head ? (
                [head]
            ) : head;

        return lng ? (
            1 < lng ? lstHead.reduce(function (a, h) {
                return a.concat(
                    and(args.slice(1)).map(function (t) {
                        return h + t;
                    })
                );
            }, []) : lstHead
        ) : [];
    }

    // PARSE TREE OPERATOR (2 of 2)
    // Each option flattened
    function or(args) {
        return args.reduce(function (a, b) {
            return a.concat(b);
        }, []);
    }

    // One list split into two (first sublist length n)
    function splitAt(n, lst) {
        return n < lst.length + 1 ? [
            lst.slice(0, n), lst.slice(n)
        ] : [lst, []];
    }

    // One list split into several (sublist lengths [n])
    function splitsAt(lstN, lst) {
        return lstN.reduceRight(function (a, x) {
            return splitAt(x, a[0]).concat(a.slice(1));
        }, [lst]);
    }

    // Value of the parse tree
    function evaluated(e) {
        return typeof e === 'string' ? e :
            e.fn(e.args.map(evaluated));
    }

    // JSON prettyprint (for parse tree, token list etc)
    function pp(e) {
        return JSON.stringify(e, function (k, v) {
            return typeof v === 'function' ? (
                '[function ' + v.name + ']'
            ) : v;
        }, 2)
    }


    // ----------------------- MAIN ------------------------

    // s -> [s]
    this.expand = function(s) {
        // BRACE EXPRESSION PARSED
        var dctParse = andTree(null, tokens(s))[0];

        // ABSTRACT SYNTAX TREE LOGGED
        // console.log(pp(dctParse));

        // AST EVALUATED TO LIST OF STRINGS
        return evaluated(dctParse);
    }

}

function asyncDelay(timeout) {
    return new Promise(function(resolve, reject) {
        setTimeout(resolve, timeout, true)
    })
}

function preventNonNumericalInput(e) {
    e = e || window.event;
    let charCode = (typeof e.which == "undefined") ? e.keyCode : e.which;
    let charStr = String.fromCharCode(charCode);
    let re = e.target.getAttribute('pattern') || '^[0-9]+$'
    re = new RegExp(re)

    if (!charStr.match(re)) {
        e.preventDefault();
    }
}

/* inserts custom html to allow prettifying of inputs */
function prettifyInputs(root_element) {
    root_element.querySelectorAll(`input[type="checkbox"]`).forEach(element => {
        var parent = element.parentNode;
        if (!parent.classList.contains("input-toggle")) {
            var wrapper = document.createElement("div");
            wrapper.classList.add("input-toggle");
            parent.replaceChild(wrapper, element);
            wrapper.appendChild(element);
            var label = document.createElement("label");
            label.htmlFor = element.id;
            wrapper.appendChild(label);
        }
    })
}
