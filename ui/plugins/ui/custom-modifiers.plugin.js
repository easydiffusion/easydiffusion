(function() {
    PLUGINS['MODIFIERS_LOAD'].push({
        loader: function() {
            let customModifiers = localStorage.getItem(CUSTOM_MODIFIERS_KEY, '')
            customModifiersTextBox.value = customModifiers

            if (customModifiersGroupElement !== undefined) {
                customModifiersGroupElement.remove()
            }

            if (customModifiers && customModifiers.trim() !== '') {
                customModifiers = customModifiers.split('\n')
                customModifiers = customModifiers.filter(m => m.trim() !== '')
                customModifiers = customModifiers.map(function(m) {
                    return {
                        "modifier": m
                    }
                })

                let customGroup = {
                    'category': 'Custom Modifiers',
                    'modifiers': customModifiers
                }

                customModifiersGroupElement = createModifierGroup(customGroup, true)

                createCollapsibles(customModifiersGroupElement)
            }
        }
    })
})()
