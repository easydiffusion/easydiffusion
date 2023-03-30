;(function() {
    // Register selftests when loaded by jasmine.
    if (typeof PLUGINS?.SELFTEST === "object") {
        PLUGINS.SELFTEST["release-notes"] = function() {
            it("should be able to fetch CHANGES.md", async function() {
                let releaseNotes = await fetch(
                    `https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/main/CHANGES.md`
                )
                expect(releaseNotes.status).toBe(200)
            })
        }
    }

    document.querySelector(".tab-container")?.insertAdjacentHTML(
        "beforeend",
        `
        <span id="tab-news" class="tab">
            <span><i class="fa fa-bolt icon"></i> What's new?</span>
        </span>
    `
    )

    document.querySelector("#tab-content-wrapper")?.insertAdjacentHTML(
        "beforeend",
        `
        <div id="tab-content-news" class="tab-content">
            <div id="news" class="tab-content-inner">
                Loading..
            </div>
        </div>
    `
    )

    const tabNews = document.querySelector("#tab-news")
    if (tabNews) {
        linkTabContents(tabNews)
    }
    const news = document.querySelector("#news")
    if (!news) {
        // news tab not found, dont exec plugin code.
        return
    }

    document.querySelector("body").insertAdjacentHTML(
        "beforeend",
        `
        <style>
        #tab-content-news .tab-content-inner {
            max-width: 100%;
            text-align: left;
            padding: 10pt;
        }
        </style>
    `
    )

    loadScript("/media/js/marked.min.js").then(async function() {
        let appConfig = await fetch("/get/app_config")
        if (!appConfig.ok) {
            console.error("[release-notes] Failed to get app_config.")
            return
        }
        appConfig = await appConfig.json()

        const updateBranch = appConfig.update_branch || "main"

        let releaseNotes = await fetch(
            `https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/${updateBranch}/CHANGES.md`
        )
        if (!releaseNotes.ok) {
            console.error("[release-notes] Failed to get CHANGES.md.")
            return
        }
        releaseNotes = await releaseNotes.text()
        news.innerHTML = marked.parse(releaseNotes)
    })
})()
