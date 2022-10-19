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
    var handle = element.firstChild;
    collapsibleHeader.classList.toggle("active")
    let content = getNextSibling(collapsibleHeader, '.collapsible-content')
    if (content.style.display === "block") {
        content.style.display = "none"
        handle.innerHTML = '&#x2795;' // plus
    } else {
        content.style.display = "block"
        handle.innerHTML = '&#x2796;' // minus
    }
    
    if (COLLAPSIBLES_INITIALIZED && COLLAPSIBLE_PANELS.includes(element.parentElement)) {
        saveCollapsibles();
    }
}

function saveCollapsibles() {
    var values = {};
    console.log(COLLAPSIBLE_PANELS);
    COLLAPSIBLE_PANELS.forEach(element => {
        values[element.id] = element.querySelector(".collapsible").className.indexOf("active") !== -1;
    });
    localStorage.setItem(COLLAPSIBLES_KEY, JSON.stringify(values));
}

function createCollapsibles(node) {
    var save = false;
    if (!node) {
        node = document;
        save = true;
    }
    // FIX: problem here is that the elements we're getting in c are the buttons, and they are the children of the things with collapsible stuff
    // FIX: gotta get parent

    // default closed.
    let collapsibles = node.querySelectorAll(".collapsible")
    collapsibles.forEach(function(c) {
        if (save && c.parentElement.id) {
            COLLAPSIBLE_PANELS.push(c.parentElement);
        }
        let handle = document.createElement('span')
        handle.className = 'collapsible-handle'

        if (c.className.indexOf('active') !== -1) {
            handle.innerHTML = '&#x2796;' // minus
        } else {
            handle.innerHTML = '&#x2795;' // plus
        }
        c.insertBefore(handle, c.firstChild)

        c.addEventListener('click', function() {
            toggleCollapsible(c.parentElement);
        });
    });
    if (save) {
        var saved = localStorage.getItem(COLLAPSIBLES_KEY);
        if (!saved) {
            saveCollapsibles();
            saved = localStorage.getItem(COLLAPSIBLES_KEY);
        }
        var values = JSON.parse(saved);
        Object.keys(values).forEach(element_id => {
            if (values[element_id]) {
                var element = COLLAPSIBLE_PANELS.find(e => e.id == element_id);
                toggleCollapsible(element);
            }
        });
        COLLAPSIBLES_INITIALIZED = true;
    }
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