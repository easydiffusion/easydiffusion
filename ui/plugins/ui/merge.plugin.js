;(function() {
    "use strict"

    let mergeCSS = `
        /*********** Main tab ***********/
        .tab-centered {
            justify-content: center;
        }

        #model-tool-tab-content {
            background-color: var(--background-color3);
        }

        #model-tool-tab-content .tab-content-inner {
            text-align: initial;
        }

        #model-tool-tab-bar .tab {
            margin-bottom: 0px;
            border-top-left-radius: var(--input-border-radius);
            background-color: var(--background-color3);
            padding: 6px 6px 0.8em 6px;
        }
        #tab-content-merge .tab-content-inner {
            max-width: 100%;
            padding: 10pt;
        }

        /*********** Merge UI ***********/
        .merge-model-container {  
            margin-left: 15%;
            margin-right: 15%;
            text-align: left;
            display: inline-grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 0px 0px;
            grid-auto-flow: row;
            grid-template-areas:
              "merge-input merge-config"
              "merge-buttons merge-buttons";
        }
        .merge-model-container p {
            margin-top: 3pt;
            margin-bottom: 3pt;
        }
        .merge-config .tab-content {
            background: var(--background-color1);
            border-radius: 3pt;
        }
        .merge-config .tab-content-inner {
            text-align: left;
        }

        .merge-input { 
            grid-area: merge-input; 
            padding-left:1em;
        }
        .merge-config { 
            grid-area: merge-config; 
            padding:1em;
        }
        .merge-config input { 
            margin-bottom: 3px; 
        }
        .merge-config select { 
            margin-bottom: 3px; 
        }
        .merge-buttons { 
            grid-area: merge-buttons; 
            padding:1em; 
            text-align: center;
        }
        #merge-button  { 
            padding: 8px; 
            width:20em; 
        }
        div#merge-log {
            height:150px; 
            overflow-x:hidden;
            overflow-y:scroll;
            background:var(--background-color1);
            border-radius: 3pt;
        }
        div#merge-log i {
            color: hsl(var(--accent-hue), 100%, calc(2*var(--accent-lightness))); 
            font-family: monospace;
        }
        .disabled { 
            background: var(--background-color4); 
            color: var(--text-color); 
        }
        #merge-type-tabs {
            border-bottom: 1px solid black;
        }
        #merge-log-container {
            display: none;
        }
        .merge-model-container #merge-warning {
            color: var(--small-label-color);
        }

        /*********** LORA UI ***********/
        .lora-manager-grid {  
            display: grid;
            gap: 0px 8px;
            grid-auto-flow: row;
        }

        @media screen and (min-width: 1501px) {
            .lora-manager-grid textarea {
                height:350px;
            }

            .lora-manager-grid {  
                grid-template-columns: auto 1fr 1fr;
                grid-template-rows: auto 1fr;
                grid-template-areas:
                    "selector  selector selector"
                    "thumbnail keywords notes";
            }
        }

        @media screen and (min-width: 1001px) and (max-width: 1500px) {
            .lora-manager-grid textarea {
                height:250px;
            }

            .lora-manager-grid {  
                grid-template-columns: auto auto;
                grid-template-rows: auto auto auto;
                grid-template-areas:
                    "selector  selector"
                    "thumbnail keywords"
                    "thumbnail notes";
            }

        }

        @media screen and (max-width: 1000px) {
            .lora-manager-grid textarea {
                height:200px;
            }

            .lora-manager-grid {  
                grid-template-columns: auto;
                grid-template-rows: auto auto auto auto;
                grid-template-areas:
                    "selector"
                    "keywords"
                    "thumbnail"
                    "notes";
            }

        }

        .lora-manager-grid-selector { 
            grid-area: selector; 
            justify-self: start;
        }

        .lora-manager-grid-thumbnail { 
            grid-area: thumbnail; 
            justify-self: center;
        }

        .lora-manager-grid-keywords { 
            grid-area: keywords; 
        }

        .lora-manager-grid-notes { 
            grid-area: notes; 
        }
        
        .lora-manager-grid p {
            margin-bottom: 2px;
        }


        `

    let mergeUI = `
        <div class="merge-model-container panel-box">
          <div class="merge-input">
             <p><label for="#mergeModelA">Select Model A:</label></p>
             <input id="mergeModelA" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" />
             <p><label for="#mergeModelB">Select Model B:</label></p>
             <input id="mergeModelB" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" />
             <br/><br/>
             <p id="merge-warning"><small><b>Important:</b> Please merge models of similar type.<br/>For e.g. <code>SD 1.4</code> models with only <code>SD 1.4/1.5</code> models,<br/><code>SD 2.0</code> with <code>SD 2.0</code>-type, and <code>SD 2.1</code> with <code>SD 2.1</code>-type models.</small></p>
             <br/>
             <table>
                <tr>
                    <td><label for="#merge-filename">Output file name:</label></td>
                    <td><input id="merge-filename" size=24> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Base name of the output file.<br>Mix ratio and file suffix will be appended to this.</span></i></td>
                </tr>
                <tr>
                    <td><label for="#merge-fp">Output precision:</label></td>
                    <td><select id="merge-fp">
                        <option value="fp16">fp16 (smaller file size)</option>
                        <option value="fp32">fp32 (larger file size)</option>
                    </select>
                    <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Image generation uses fp16, so it's a good choice.<br>Use fp32 if you want to use the result models for more mixes</span></i>
                    </td>
                </tr>
                <tr>
                    <td><label for="#merge-format">Output file format:</label></td>
                    <td><select id="merge-format">
                        <option value="safetensors">Safetensors (recommended)</option>
                        <option value="ckpt">CKPT/Pickle (legacy format)</option>
                    </select>
                    </td>
                </tr>
             </table>
             <br/>
             <div id="merge-log-container">
                <p><label for="#merge-log">Log messages:</label></p>
                <div id="merge-log"></div>
             </div>
          </div>
          <div class="merge-config">
             <div class="tab-container">
                <span id="tab-merge-opts-single" class="tab active">
                    <span>Make a single file</small></span>
                </span>
                <span id="tab-merge-opts-batch" class="tab">
                    <span>Make multiple variations</small></span>
                </span>
             </div>
             <div>
                <div id="tab-content-merge-opts-single" class="tab-content active">
                    <div class="tab-content-inner">
                        <small>Saves a single merged model file, at the specified merge ratio.</small><br/><br/>
                        <label for="#single-merge-ratio-slider">Merge ratio:</label>
                        <input id="single-merge-ratio-slider" name="single-merge-ratio-slider" class="editor-slider" value="50" type="range" min="1" max="1000">
                        <input id="single-merge-ratio" size=2 value="5">%
                        <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Model A's contribution to the mix. The rest will be from Model B.</span></i>
                    </div>
                </div>
                <div id="tab-content-merge-opts-batch" class="tab-content">
                    <div class="tab-content-inner">
                        <small>Saves multiple variations of the model, at different merge ratios.<br/>Each variation will be saved as a separate file.</small><br/><br/>
                        <table>
                            <tr><td><label for="#merge-count">Number of variations:</label></td>
                                <td> <input id="merge-count" size=2 value="5"></td>
                                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Number of models to create</span></i></td></tr>
                            <tr><td><label for="#merge-start">Starting merge ratio:</label></td>
                                <td> <input id="merge-start" size=2 value="5">%</td>
                                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Smallest share of model A in the mix</span></i></td></tr>
                            <tr><td><label for="#merge-step">Increment each step:</label></td>
                                <td> <input id="merge-step" size=2 value="10">%</td>
                                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Share of model A added into the mix per step</span></i></td></tr>
                            <tr><td><label for="#merge-interpolation">Interpolation model:</label></td>
                                <td> <select id="merge-interpolation">
                                    <option>Exact</option>
                                    <option>SmoothStep</option>
                                    <option>SmootherStep</option>
                                    <option>SmoothestStep</option>
                                </select></td>
                                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Sigmoid function to be applied to the model share before mixing</span></i></td></tr>
                        </table>
                        <br/>
                        <small>Preview of variation ratios:</small><br/>
                        <canvas id="merge-canvas" width="400" height="400"></canvas>
                    </div>
                </div>
             </div>
          </div>
          <div class="merge-buttons">
             <button id="merge-button" class="primaryButton">Merge models</button>
          </div>
        </div>`

        
    let loraUI=`
        <div class="panel-box lora-manager-grid">
            <div class="lora-manager-grid-selector">
                <label for="#loraModel">Select Lora:</label>
                <input id="loraModel" type="text" spellcheck="false" autocomplete="off" class="model-filter" data-path="" />
            </div>
            <div class="lora-manager-grid-thumbnail">
                <p style="height:2em;">Thumbnail:</p>
                <div style="position:relative; height:256px; width:256px;background-color:#222;border-radius:1em;margin-bottom:1em;">
                    <i id="lora-manager-image-placeholder" class="fa-regular fa-image" style="font-size:500%;color:#555;position:absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);"></i>
                    <img id="lora-manager-image" class="displayNone" style="border-radius:6px;max-height:256px;max-width:256px;"/>
                </div>
                <div style="text-align:center;">
                    <button class="tertiaryButton"><i class="fa-solid fa-upload"></i> Upload new thumbnail</button>
                    <button class="tertiaryButton"><i class="fa-solid fa-trash-can"></i> Remove</button>
                </div>
            </div>
            <div class="lora-manager-grid-keywords">
                <p style="height:2em;">Keywords:
                <span style="float:right;margin-bottom:4px;"><button id="lora-keyword-from-civitai" class="tertiaryButton smallButton">Import from Civitai</button></span></p>
                <textarea style="width:100%;resize:vertical;" id="lora-manager-keywords" placeholder="Put LORA specific keywords here..."></textarea>
                <p style="color:var(--small-label-color);">
                    <b>LORA model keywords</b> can be used via the <code>+&nbsp;Embeddings</code> button. They get added to the embedding 
                    keyword menu when the LORA has been selected in the image settings.
                </p>
            </div>
            <div class="lora-manager-grid-notes">
                <p style="height:2em;">Notes:</p>
                <textarea style="width:100%;resize:vertical;"  id="lora-manager-notes" placeholder="Place for things you want to remember..."></textarea>
                <p id="civitai-section" class="displayNone">
                    <b>Civitai model page:</b>
                    <a id="civitai-model-page" target="_blank"></a>
                </p>
            </div>
        </div>`

    let tabHTML=`
        <div id="model-tool-tab-bar" class="tab-container tab-centered">
            <span id="tab-model-loraUI" class="tab active">
                <span><i class="fa-solid fa-key"></i> Lora Keywords</small></span>
            </span>
            <span id="tab-model-mergeUI" class="tab">
                <span><i class="fa-solid fa-code-merge"></i> Merge Models</small></span>
            </span>
        </div>
        <div id="model-tool-tab-content" class="panel-box">
            <div id="tab-content-model-loraUI" class="tab-content active">
                <div class="tab-content-inner">
                    ${loraUI}
                </div>
            </div>

            <div id="tab-content-model-mergeUI" class="tab-content">
                <div class="tab-content-inner">
                    ${mergeUI}
                </div>
            </div>
        </div>`


    ///////////////////// Function section
    function smoothstep(x) {
        return x * x * (3 - 2 * x)
    }

    function smootherstep(x) {
        return x * x * x * (x * (x * 6 - 15) + 10)
    }

    function smootheststep(x) {
        let y = -20 * Math.pow(x, 7)
        y += 70 * Math.pow(x, 6)
        y -= 84 * Math.pow(x, 5)
        y += 35 * Math.pow(x, 4)
        return y
    }
    function getCurrentTime() {
        const now = new Date()
        let hours = now.getHours()
        let minutes = now.getMinutes()
        let seconds = now.getSeconds()

        hours = hours < 10 ? `0${hours}` : hours
        minutes = minutes < 10 ? `0${minutes}` : minutes
        seconds = seconds < 10 ? `0${seconds}` : seconds

        return `${hours}:${minutes}:${seconds}`
    }

    function addLogMessage(message) {
        const logContainer = document.getElementById("merge-log")
        logContainer.innerHTML += `<i>${getCurrentTime()}</i> ${message}<br>`

        // Scroll to the bottom of the log
        logContainer.scrollTop = logContainer.scrollHeight

        document.querySelector("#merge-log-container").style.display = "block"
    }

    function addLogSeparator() {
        const logContainer = document.getElementById("merge-log")
        logContainer.innerHTML += "<hr>"

        logContainer.scrollTop = logContainer.scrollHeight
    }

    function drawDiagram(fn) {
        const SIZE = 300
        const canvas = document.getElementById("merge-canvas")
        canvas.height = canvas.width = SIZE
        const ctx = canvas.getContext("2d")

        // Draw coordinate system
        ctx.scale(1, -1)
        ctx.translate(0, -canvas.height)
        ctx.lineWidth = 1
        ctx.beginPath()

        ctx.strokeStyle = "white"
        ctx.moveTo(0, 0)
        ctx.lineTo(0, SIZE)
        ctx.lineTo(SIZE, SIZE)
        ctx.lineTo(SIZE, 0)
        ctx.lineTo(0, 0)
        ctx.lineTo(SIZE, SIZE)
        ctx.stroke()
        ctx.beginPath()
        ctx.setLineDash([1, 2])
        const n = SIZE / 10
        for (let i = n; i < SIZE; i += n) {
            ctx.moveTo(0, i)
            ctx.lineTo(SIZE, i)
            ctx.moveTo(i, 0)
            ctx.lineTo(i, SIZE)
        }
        ctx.stroke()
        ctx.beginPath()
        ctx.setLineDash([])
        ctx.beginPath()
        ctx.strokeStyle = "black"
        ctx.lineWidth = 3
        // Plot function
        const numSamples = 20
        for (let i = 0; i <= numSamples; i++) {
            const x = i / numSamples
            const y = fn(x)

            const canvasX = x * SIZE
            const canvasY = y * SIZE

            if (i === 0) {
                ctx.moveTo(canvasX, canvasY)
            } else {
                ctx.lineTo(canvasX, canvasY)
            }
        }
        ctx.stroke()
        // Plot alpha values (yellow boxes)
        let start = parseFloat(document.querySelector("#merge-start").value)
        let step = parseFloat(document.querySelector("#merge-step").value)
        let iterations = document.querySelector("#merge-count").value >> 0
        ctx.beginPath()
        ctx.fillStyle = "yellow"
        for (let i = 0; i < iterations; i++) {
            const alpha = (start + i * step) / 100
            const x = alpha * SIZE
            const y = fn(alpha) * SIZE
            if (x <= SIZE) {
                ctx.rect(x - 3, y - 3, 6, 6)
                ctx.fill()
            } else {
                ctx.strokeStyle = "red"
                ctx.moveTo(0, 0)
                ctx.lineTo(0, SIZE)
                ctx.lineTo(SIZE, SIZE)
                ctx.lineTo(SIZE, 0)
                ctx.lineTo(0, 0)
                ctx.lineTo(SIZE, SIZE)
                ctx.stroke()
                addLogMessage("<i>Warning: maximum ratio is &#8805; 100%</i>")
            }
        }
    }

    function updateChart() {
        let fn = (x) => x
        switch (document.querySelector("#merge-interpolation").value) {
            case "SmoothStep":
                fn = smoothstep
                break
            case "SmootherStep":
                fn = smootherstep
                break
            case "SmoothestStep":
                fn = smootheststep
                break
        }
        drawDiagram(fn)
    }

    function initMergeUI() {
        const tabSettingsSingle = document.querySelector("#tab-merge-opts-single")
        const tabSettingsBatch = document.querySelector("#tab-merge-opts-batch")
        linkTabContents(tabSettingsSingle)
        linkTabContents(tabSettingsBatch)

        let mergeModelAField = new ModelDropdown(document.querySelector("#mergeModelA"), "stable-diffusion")
        let mergeModelBField = new ModelDropdown(document.querySelector("#mergeModelB"), "stable-diffusion")
        updateChart()

        // slider
        const singleMergeRatioField = document.querySelector("#single-merge-ratio")
        const singleMergeRatioSlider = document.querySelector("#single-merge-ratio-slider")

        function updateSingleMergeRatio() {
            singleMergeRatioField.value = singleMergeRatioSlider.value / 10
            singleMergeRatioField.dispatchEvent(new Event("change"))
        }

        function updateSingleMergeRatioSlider() {
            if (singleMergeRatioField.value < 0) {
                singleMergeRatioField.value = 0
            } else if (singleMergeRatioField.value > 100) {
                singleMergeRatioField.value = 100
            }

            singleMergeRatioSlider.value = singleMergeRatioField.value * 10
            singleMergeRatioSlider.dispatchEvent(new Event("change"))
        }

        singleMergeRatioSlider.addEventListener("input", updateSingleMergeRatio)
        singleMergeRatioField.addEventListener("input", updateSingleMergeRatioSlider)
        updateSingleMergeRatio()

        document.querySelector(".merge-config").addEventListener("change", updateChart)

        document.querySelector("#merge-button").addEventListener("click", async function(e) {
            // Build request template
            let model0 = mergeModelAField.value
            let model1 = mergeModelBField.value
            let request = { model0: model0, model1: model1 }
            request["use_fp16"] = document.querySelector("#merge-fp").value == "fp16"
            let iterations = document.querySelector("#merge-count").value >> 0
            let start = parseFloat(document.querySelector("#merge-start").value)
            let step = parseFloat(document.querySelector("#merge-step").value)

            if (isTabActive(tabSettingsSingle)) {
                start = parseFloat(singleMergeRatioField.value)
                step = 0
                iterations = 1
                addLogMessage(`merge ratio = ${start}%`)
            } else {
                addLogMessage(`start = ${start}%`)
                addLogMessage(`step  = ${step}%`)
            }

            if (start + (iterations - 1) * step >= 100) {
                addLogMessage("<i>Aborting: maximum ratio is &#8805; 100%</i>")
                addLogMessage("Reduce the number of variations or the step size")
                addLogSeparator()
                document.querySelector("#merge-count").focus()
                return
            }

            if (document.querySelector("#merge-filename").value == "") {
                addLogMessage("<i>Aborting: No output file name specified</i>")
                addLogSeparator()
                document.querySelector("#merge-filename").focus()
                return
            }

            // Disable merge button
            e.target.disabled = true
            e.target.classList.add("disabled")
            let cursor = $("body").css("cursor")
            let label = document.querySelector("#merge-button").innerHTML
            $("body").css("cursor", "progress")
            document.querySelector("#merge-button").innerHTML = "Merging models ..."

            addLogMessage("Merging models")
            addLogMessage("Model A: " + model0)
            addLogMessage("Model B: " + model1)

            // Batch main loop
            for (let i = 0; i < iterations; i++) {
                let alpha = (start + i * step) / 100

                if (isTabActive(tabSettingsBatch)) {
                    switch (document.querySelector("#merge-interpolation").value) {
                        case "SmoothStep":
                            alpha = smoothstep(alpha)
                            break
                        case "SmootherStep":
                            alpha = smootherstep(alpha)
                            break
                        case "SmoothestStep":
                            alpha = smootheststep(alpha)
                            break
                    }
                }
                addLogMessage(`merging batch job ${i + 1}/${iterations}, alpha = ${alpha.toFixed(5)}...`)

                request["out_path"] = document.querySelector("#merge-filename").value
                request["out_path"] += "-" + alpha.toFixed(5) + "." + document.querySelector("#merge-format").value
                addLogMessage(`&nbsp;&nbsp;filename: ${request["out_path"]}`)

                // sdkit documentation: "ratio - the ratio of the second model. 1 means only the second model will be used."
                request["ratio"] = 1-alpha 
                let res = await fetch("/model/merge", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(request),
                })
                const data = await res.json()
                addLogMessage(JSON.stringify(data))
            }
            addLogMessage(
                "<b>Done.</b> The models have been saved to your <tt>models/stable-diffusion</tt> folder."
            )
            addLogSeparator()
            // Re-enable merge button
            $("body").css("cursor", cursor)
            document.querySelector("#merge-button").innerHTML = label
            e.target.disabled = false
            e.target.classList.remove("disabled")

            // Update model list
            stableDiffusionModelField.innerHTML = ""
            vaeModelField.innerHTML = ""
            hypernetworkModelField.innerHTML = ""
            await getModels()
        })
    }

    const LoraUI = {
        modelField: undefined,
        keywordsField: undefined,
        notesField: undefined,
        civitaiImportBtn: undefined,
        civitaiSecion: undefined,
        civitaiAnchor: undefined,
        image: undefined,
        imagePlaceholder: undefined,

        init() {
            LoraUI.modelField = new ModelDropdown(document.querySelector("#loraModel"), "lora")
            LoraUI.keywordsField = document.querySelector("#lora-manager-keywords")
            LoraUI.notesField = document.querySelector("#lora-manager-notes")
            LoraUI.civitaiImportBtn = document.querySelector("#lora-keyword-from-civitai")
            LoraUI.civitaiSection = document.querySelector("#civitai-section")
            LoraUI.civitaiAnchor = document.querySelector("#civitai-model-page")
            LoraUI.image = document.querySelector("#lora-manager-image")
            LoraUI.imagePlaceholder = document.querySelector("#lora-manager-image-placeholder")

            LoraUI.modelField.addEventListener("change", LoraUI.updateFields)
            LoraUI.keywordsField.addEventListener("focusout", LoraUI.saveInfos)
            LoraUI.notesField.addEventListener("focusout", LoraUI.saveInfos)
            LoraUI.civitaiImportBtn.addEventListener("click", LoraUI.importFromCivitai)

            LoraUI.updateFields()
        },

        updateFields() {
            document.getElementById("civitai-section").classList.add("displayNone")
            Bucket.retrieve(`modelinfo/lora/${LoraUI.modelField.value}`)
                .then((info) => {
                    if (info == null) {
                        LoraUI.keywordsField.value = ""
                        LoraUI.notesField.value = ""
                        LoraUI.hideCivitaiLink()
                    } else {
                        LoraUI.keywordsField.value = info.keywords.join("\n")
                        LoraUI.notesField.value = info.notes
                        if ("civitai" in info) {
                            LoraUI.showCivitaiLink(info.civitai)
                        }
                    }
                })
            Bucket.getImageAsDataURL(`${profileNameField.value}/lora/${LoraUI.modelField.value}.png`)
                .then((data) => {
                    LoraUI.image.src=data
                    LoraUI.image.classList.remove("displayNone")
                    LoraUI.imagePlaceholder.classList.add("displayNone")
                })
                .catch((error) => {
                    console.error("Caught error:", error)
                    LoraUI.image.classList.add("displayNone")
                    LoraUI.imagePlaceholder.classList.remove("displayNone")
                })
        },

        saveInfos() {
            let info = {
                keywords: LoraUI.keywordsField.value
                            .split("\n")
                            .filter((x) => (x != "")),
                notes: LoraUI.notesField.value,
                civitai: LoraUI.civitaiSection.checkVisibility() ? LoraUI.civitaiAnchor.href : null, 
            }
            Bucket.store(`modelinfo/lora/${LoraUI.modelField.value}`, info)
        },

        importFromCivitai() {
            document.body.style["cursor"] = "progress"
            fetch("/sha256/lora/"+LoraUI.modelField.value)
                .then((result) => result.json())
                .then((json) => fetch("https://civitai.com/api/v1/model-versions/by-hash/" + json.digest))
                .then((result) => result.json())
                .then((json) => {
                    document.body.style["cursor"] = "default"
                    if (json == null) {
                        return
                    }
                    if ("trainedWords" in json) {
                        LoraUI.keywordsField.value = json["trainedWords"].join("\n")
                    } else {
                        showToast("No keyword info found.")
                    }
                    if ("modelId" in json) {
                        LoraUI.showCivitaiLink("https://civitai.com/models/" + json.modelId)
                    } else {
                        LoraUI.hideCivitaiLink()
                    }

                    LoraUI.saveInfos()
                })
        },

        showCivitaiLink(href) {
            LoraUI.civitaiSection.classList.remove("displayNone")
            LoraUI.civitaiAnchor.href = href
            LoraUI.civitaiAnchor.innerHTML = LoraUI.civitaiAnchor.href
        },

        hideCivitaiLink() {
            LoraUI.civitaiSection.classList.add("displayNone")
        }
    }

    createTab({
        id: "merge",
        icon: "fa-toolbox",
        label: "Model tools",
        css: mergeCSS,
        content: tabHTML, 
        onOpen: ({ firstOpen }) => {
            if (!firstOpen) {
                return
            }
            initMergeUI()
            LoraUI.init()
            const tabMergeUI = document.querySelector("#tab-model-mergeUI")
            const tabLoraUI = document.querySelector("#tab-model-loraUI")
            linkTabContents(tabMergeUI)
            linkTabContents(tabLoraUI)
        },
    })
})()
async function getLoraKeywords(model) {
    return Bucket.retrieve(`modelinfo/lora/${model}`)
        .then((info) => info.keywords)
}
