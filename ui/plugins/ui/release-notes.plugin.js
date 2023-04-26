(function() {
    // Register selftests when loaded by jasmine.
    if (typeof PLUGINS?.SELFTEST === 'object') {
        PLUGINS.SELFTEST["release-notes"] = function() {
            it('should be able to fetch CHANGES.md', async function() {
                let releaseNotes = await fetch(`https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/main/CHANGES.md`)
                expect(releaseNotes.status).toBe(200)
            })
        }
    }

    createTab({
        id: 'news',
        icon: 'fa-bolt',
        label: "What's new",
        css: `
        #tab-content-news .tab-content-inner {
            max-width: 100%;
            text-align: left;
            padding: 10pt;
        }
        `,
        onOpen: async ({ firstOpen }) => {
            if (firstOpen) {
                const loadMarkedScriptPromise = loadScript('/media/js/marked.min.js')

                let appConfig = await fetch('/get/app_config')
                if (!appConfig.ok) {
                    console.error('[release-notes] Failed to get app_config.')
                    return
                }
                appConfig = await appConfig.json()
        
                const updateBranch = appConfig.update_branch || 'main'
        
                let releaseNotes = await fetch(`https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/${updateBranch}/CHANGES.md`)
                if (!releaseNotes.ok) {
                    console.error('[release-notes] Failed to get CHANGES.md.')
                    return
                }
                releaseNotes = await releaseNotes.text()

                await loadMarkedScriptPromise

                return marked.parse(releaseNotes)
            }
        },
    })
})()