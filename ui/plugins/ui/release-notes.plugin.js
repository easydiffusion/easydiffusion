(function() {
    document.querySelector('#tab-container').insertAdjacentHTML('beforeend', `
        <span id="tab-news" class="tab">
            <span><i class="fa fa-bolt icon"></i> What's new?</span>
        </span>
    `)

    document.querySelector('#tab-content-wrapper').insertAdjacentHTML('beforeend', `
        <div id="tab-content-news" class="tab-content">
            <div id="news" class="tab-content-inner">
                Loading..
            </div>
        </div>
    `)

    document.querySelector('body').insertAdjacentHTML('beforeend', `
        <style>
        #tab-content-news .tab-content-inner {
            max-width: 100%;
            text-align: left;
            padding: 10pt;
        }
        </style>
    `)

    linkTabContents(document.querySelector('#tab-news'))

    let markedScript = document.createElement('script')
    markedScript.src = '/media/js/marked.min.js'

    markedScript.onload = async function() {
        let appConfig = await fetch('/get/app_config')
        appConfig = await appConfig.json()

        let updateBranch = appConfig.update_branch || 'main'

        let news = document.querySelector('#news')
        let releaseNotes = await fetch(`https://raw.githubusercontent.com/cmdr2/stable-diffusion-ui/${updateBranch}/CHANGES.md`)
        if (releaseNotes.status != 200) {
            return
        }
        releaseNotes = await releaseNotes.text()
        news.innerHTML = marked.parse(releaseNotes)
    }

    document.querySelector('body').appendChild(markedScript)
})()