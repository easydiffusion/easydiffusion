const themeField = document.getElementById("theme");
var DEFAULT_THEME = {};
var THEMES = []; // initialized in initTheme from data in css

function getThemeName(theme) {
    theme = theme.replace("theme-", "");
    theme = theme.split("-").map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
    return theme;
}
// init themefield
function initTheme() {
    Array.from(document.styleSheets)
        .filter(sheet => sheet.href?.startsWith(window.location.origin))
        .flatMap(sheet => Array.from(sheet.cssRules))
        .forEach(rule => {
            var selector = rule.selectorText; // TODO: also do selector == ":root", re-run un-set props
            if (selector && selector.startsWith(".theme-")) {
                var theme_key = selector.substring(1);
                THEMES.push({
                    key: theme_key,
                    name: getThemeName(theme_key),
                    rule: rule
                })
            }
            if (selector && selector == ":root") {
                DEFAULT_THEME = {
                    key: "theme-default",
                    name: "Default",
                    rule: rule
                };
            }
    });
    
    THEMES.forEach(theme => {
        var new_option = document.createElement("option");
        new_option.setAttribute("value", theme.key);
        new_option.innerText = theme.name;
        themeField.appendChild(new_option);
    });

    
    // setup the style transitions a second after app initializes, so initial style is instant
    setTimeout(() => {
        var body = document.querySelector("body");
        var style = document.createElement('style');
        style.innerHTML = "* { transition: background 0.5s, color 0.5s, background-color 0.5s; }";
        body.appendChild(style);
    }, 1000);
}
initTheme();

function themeFieldChanged() {
    var theme_key = themeField.value;

    var body = document.querySelector("body");
    body.classList.remove(...THEMES.map(theme => theme.key));
    body.classList.add(theme_key);
    
    // 

    body.style = "";
    var theme = THEMES.find(t => t.key == theme_key);
    if (theme) {
        // refresh variables incase they are back referencing
        Array.from(DEFAULT_THEME.rule.style)
            .filter(cssVariable => !Array.from(theme.rule.style).includes(cssVariable))
            .forEach(cssVariable => {
            body.style.setProperty(cssVariable, DEFAULT_THEME.rule.style.getPropertyValue(cssVariable));
        });
    }
}

themeField.addEventListener('change', themeFieldChanged);
