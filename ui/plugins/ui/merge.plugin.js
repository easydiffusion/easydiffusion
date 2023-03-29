(function() {
    "use strict"

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
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        let seconds = now.getSeconds();

        hours = hours < 10 ? `0${hours}` : hours;
        minutes = minutes < 10 ? `0${minutes}` : minutes;
        seconds = seconds < 10 ? `0${seconds}` : seconds;

        return `${hours}:${minutes}:${seconds}`;
    }

    function addLogMessage(message) {
        const logContainer = document.getElementById('merge-log');
        logContainer.innerHTML += `<i>${getCurrentTime()}</i> ${message}<br>`;

        // Scroll to the bottom of the log
        logContainer.scrollTop = logContainer.scrollHeight;

        document.querySelector('#merge-log-container').style.display = 'block'
    }    

    function addLogSeparator() {
        const logContainer = document.getElementById('merge-log');
        logContainer.innerHTML += '<hr>'

        logContainer.scrollTop = logContainer.scrollHeight;
    }

    function drawDiagram(fn) {
        const SIZE = 300
        const canvas = document.getElementById('merge-canvas');
        canvas.height = canvas.width = SIZE
        const ctx = canvas.getContext('2d');

        // Draw coordinate system
        ctx.scale(1, -1);
        ctx.translate(0, -canvas.height);
        ctx.lineWidth = 1;
        ctx.beginPath();

        ctx.strokeStyle = 'white'
        ctx.moveTo(0,0); ctx.lineTo(0,SIZE); ctx.lineTo(SIZE,SIZE); ctx.lineTo(SIZE,0); ctx.lineTo(0,0); ctx.lineTo(SIZE,SIZE);
        ctx.stroke()
        ctx.beginPath()
        ctx.setLineDash([1,2])
        const n = SIZE / 10
        for (let i=n; i<SIZE; i+=n) {
            ctx.moveTo(0,i)
            ctx.lineTo(SIZE,i)
            ctx.moveTo(i,0)
            ctx.lineTo(i,SIZE)
        }
        ctx.stroke()
        ctx.beginPath()
        ctx.setLineDash([])
        ctx.beginPath();
        ctx.strokeStyle = 'black'
        ctx.lineWidth = 3;
        // Plot function
        const numSamples = 20;
        for (let i = 0; i <= numSamples; i++) {
            const x = i / numSamples;
            const y = fn(x);
        
            const canvasX = x * SIZE;
            const canvasY = y * SIZE;

            if (i === 0) {
                ctx.moveTo(canvasX, canvasY);
            } else {
                ctx.lineTo(canvasX, canvasY);
            }
        }
        ctx.stroke()
        // Plot alpha values (yellow boxes)
        let start = parseFloat( document.querySelector('#merge-start').value )
        let step = parseFloat( document.querySelector('#merge-step').value )
        let iterations = document.querySelector('#merge-count').value>>0
        ctx.beginPath()
        ctx.fillStyle = "yellow"
        for (let i=0; i< iterations; i++) {
            const alpha = ( start + i * step ) / 100
            const x = alpha*SIZE
            const y = fn(alpha) * SIZE
            if (x <= SIZE) {
                ctx.rect(x-3,y-3,6,6)
                ctx.fill()
            } else {
                ctx.strokeStyle = 'red'
                ctx.moveTo(0,0); ctx.lineTo(0,SIZE); ctx.lineTo(SIZE,SIZE); ctx.lineTo(SIZE,0); ctx.lineTo(0,0); ctx.lineTo(SIZE,SIZE);
                ctx.stroke()
                addLogMessage('<i>Warning: maximum ratio is &#8805; 100%</i>')
            }
        }
    }

    function updateChart() {
        let fn = (x) => x
        switch (document.querySelector('#merge-interpolation').value) {
            case 'SmoothStep':
                fn = smoothstep
                break
            case 'SmootherStep':
                fn = smootherstep
                break
            case 'SmoothestStep':
                fn = smootheststep
                break
        }
        drawDiagram(fn)
    }

    /////////////////////// Tab implementation
    document.querySelector('.tab-container')?.insertAdjacentHTML('beforeend', `
        <span id="tab-merge" class="tab">
            <span><i class="fa fa-code-merge icon"></i> Merge models</span>
        </span>
    `)

    document.querySelector('#tab-content-wrapper')?.insertAdjacentHTML('beforeend', `
        <div id="tab-content-merge" class="tab-content">
            <div id="merge" class="tab-content-inner">
                Loading..
            </div>
        </div>
    `)

    const tabMerge = document.querySelector('#tab-merge')
    if (tabMerge) {
        linkTabContents(tabMerge)
    }
    const merge = document.querySelector('#merge')
    if (!merge) {
        // merge tab not found, dont exec plugin code.
        return
    }

    document.querySelector('body').insertAdjacentHTML('beforeend', `
        <style>
        #tab-content-merge .tab-content-inner {
            max-width: 100%;
            padding: 10pt;
        }
        .merge-container {  
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
        .merge-container p {
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
        .merge-container #merge-warning {
            color: rgb(153, 153, 153);
        }
        </style>
    `)

    merge.innerHTML = `
    <div class="merge-container panel-box">
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

    const tabSettingsSingle = document.querySelector('#tab-merge-opts-single')
    const tabSettingsBatch = document.querySelector('#tab-merge-opts-batch')
    linkTabContents(tabSettingsSingle)
    linkTabContents(tabSettingsBatch)

    console.log('Activate')
    let mergeModelAField = new ModelDropdown(document.querySelector('#mergeModelA'), 'stable-diffusion')
    let mergeModelBField = new ModelDropdown(document.querySelector('#mergeModelB'), 'stable-diffusion')
    updateChart()

    // slider
    const singleMergeRatioField = document.querySelector('#single-merge-ratio')
    const singleMergeRatioSlider = document.querySelector('#single-merge-ratio-slider')

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

    singleMergeRatioSlider.addEventListener('input', updateSingleMergeRatio)
    singleMergeRatioField.addEventListener('input', updateSingleMergeRatioSlider)
    updateSingleMergeRatio()

    document.querySelector('.merge-config').addEventListener('change', updateChart)

    document.querySelector('#merge-button').addEventListener('click', async function(e) {
        // Build request template
        let model0 = mergeModelAField.value
        let model1 = mergeModelBField.value
        let request = { model0: model0, model1: model1 }
        request['use_fp16'] = document.querySelector('#merge-fp').value == 'fp16'
        let iterations = document.querySelector('#merge-count').value>>0
        let start = parseFloat( document.querySelector('#merge-start').value )
        let step = parseFloat( document.querySelector('#merge-step').value )

        if (isTabActive(tabSettingsSingle)) {
            start = parseFloat(singleMergeRatioField.value)
            step = 0
            iterations = 1
            addLogMessage(`merge ratio = ${start}%`)
        } else {
            addLogMessage(`start = ${start}%`)
            addLogMessage(`step  = ${step}%`)
        }

        if (start + (iterations-1) * step >= 100) {
            addLogMessage('<i>Aborting: maximum ratio is &#8805; 100%</i>')
            addLogMessage('Reduce the number of variations or the step size')
            addLogSeparator()
            document.querySelector('#merge-count').focus()
            return
        }

        if (document.querySelector('#merge-filename').value == "") {
            addLogMessage('<i>Aborting: No output file name specified</i>')
            addLogSeparator()
            document.querySelector('#merge-filename').focus()
            return
        }
        
        // Disable merge button
        e.target.disabled=true
        e.target.classList.add('disabled')
        let cursor = $("body").css("cursor");
        let label = document.querySelector('#merge-button').innerHTML
        $("body").css("cursor", "progress");
        document.querySelector('#merge-button').innerHTML = 'Merging models ...'

        addLogMessage("Merging models")
        addLogMessage("Model A: "+model0)
        addLogMessage("Model B: "+model1)

        // Batch main loop
        for (let i=0; i<iterations; i++) {
            let alpha = ( start + i * step ) / 100
            switch (document.querySelector('#merge-interpolation').value) {
                case 'SmoothStep':
                    alpha = smoothstep(alpha)
                    break
                case 'SmootherStep':
                    alpha = smootherstep(alpha)
                    break
                case 'SmoothestStep':
                    alpha = smootheststep(alpha)
                    break
            }
            addLogMessage(`merging batch job ${i+1}/${iterations}, alpha = ${alpha.toFixed(5)}...`)

            request['out_path'] = document.querySelector('#merge-filename').value
            request['out_path'] += '-' + alpha.toFixed(5) + '.' + document.querySelector('#merge-format').value
            addLogMessage(`&nbsp;&nbsp;filename: ${request['out_path']}`)

            request['ratio'] = alpha
            let res = await fetch('/model/merge', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(request) })
            const data = await res.json();
            addLogMessage(JSON.stringify(data))
        }
        addLogMessage("<b>Done.</b> The models have been saved to your <tt>models/stable-diffusion</tt> folder.")
        addLogSeparator()
        // Re-enable merge button
        $("body").css("cursor", cursor);
        document.querySelector('#merge-button').innerHTML = label
        e.target.disabled=false
        e.target.classList.remove('disabled')

        // Update model list
        stableDiffusionModelField.innerHTML = ''
        vaeModelField.innerHTML = ''
        hypernetworkModelField.innerHTML = ''
        await getModels()
    })

})()
