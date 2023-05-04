/* SD-UI Selftest Plugin.js
 */
;(function() {
    "use strict"
    const ID_PREFIX = "selftest-plugin"

    const links = document.getElementById("community-links")
    if (!links) {
        console.error('%s the ID "community-links" cannot be found.', ID_PREFIX)
        return
    }

    // Add link to Jasmine SpecRunner
    const pluginLink = document.createElement("li")
    const options = {
        stopSpecOnExpectationFailure: "true",
        stopOnSpecFailure: "false",
        random: "false",
        hideDisabled: "false",
    }
    const optStr = Object.entries(options)
        .map(([key, val]) => `${key}=${val}`)
        .join("&")
    pluginLink.innerHTML = `<a id="${ID_PREFIX}-starttest" href="${location.protocol}/plugins/core/SpecRunner.html?${optStr}" target="_blank"><i class="fa-solid fa-vial-circle-check"></i> Start SelfTest</a>`
    links.appendChild(pluginLink)

    console.log("%s loaded!", ID_PREFIX)
})()
