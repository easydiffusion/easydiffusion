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
    }    

    function addLogSeparator() {
        const logContainer = document.getElementById('merge-log');
        logContainer.innerHTML += '<hr>'

        logContainer.scrollTop = logContainer.scrollHeight;
    }

    function drawDiagram(fn) {
        const canvas = document.getElementById('merge-canvas');
        canvas.width=canvas.width
        const ctx = canvas.getContext('2d');

        // Draw coordinate system
        ctx.scale(1, -1);
        ctx.translate(0, -canvas.height);
        ctx.lineWidth = 1;
        ctx.beginPath();

        ctx.strokeStyle = 'white'
        ctx.moveTo(0,0); ctx.lineTo(0,400); ctx.lineTo(400,400); ctx.lineTo(400,0); ctx.lineTo(0,0); ctx.lineTo(400,400);
        ctx.stroke()
        ctx.beginPath()
        ctx.setLineDash([1,2])
        for (let i=40; i<400; i+=40) {
            ctx.moveTo(0,i)
            ctx.lineTo(400,i)
            ctx.moveTo(i,0)
            ctx.lineTo(i,400)
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
        
            const canvasX = x * 400;
            const canvasY = y * 400;

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
            const x = alpha*400
            const y = fn(alpha) * 400
            if (x <= 400) {
                ctx.rect(x-3,y-3,6,6)
                ctx.fill()
            } else {
                ctx.strokeStyle = 'red'
                ctx.moveTo(0,0); ctx.lineTo(0,400); ctx.lineTo(400,400); ctx.lineTo(400,0); ctx.lineTo(0,0); ctx.lineTo(400,400);
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
    document.querySelector('#tab-container')?.insertAdjacentHTML('beforeend', `
        <span id="tab-merge" class="tab">
            <span><i class="fa fa-code-merge icon"></i> Merge models <small>(beta)</small></span>
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
            text-align: center;
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
        }
        div#merge-log i {
            color: hsl(var(--accent-hue), 100%, calc(2*var(--accent-lightness))); 
            font-family: monospace;
        }
        .disabled { 
            background: var(--background-color4); 
            color: var(--text-color); 
        }
        </style>
    `)

    merge.innerHTML = `
    <div class="merge-container panel-box">
      <div class="merge-input">
         <h1>Batch merger</h1>
         <p><label for="#mergeModelA">Select Model A:</label></p>
         <select id="mergeModelA">
             <option>A</option>
         </select>
         <p><label for="#mergeModelB">Select Model B:</label></p>
         <select id="mergeModelB">
             <option>A</option>
         </select>
         <p><small><b>Important:</b> Please merge models of similar type.<br/>For e.g. <code>SD 1.4</code> models with only <code>SD 1.4/1.5</code> models,<br/> <code>SD 2.0</code> with <code>SD 2.0</code>-type, and <code>SD 2.1</code> with <code>SD 2.1</code>-type models.</small></p>
         <p><label for="#merge-log">Log messages:</label></p>
         <div id="merge-log"></div>
      </div>
      <div class="merge-config">
         <table>
            <tr><td><label for="#merge-filename">Output file name:</label></td>
                <td> <input id="merge-filename" size=24></td>
                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Base name of the output file.<br>Mix ratio and file suffix will be appended to this.</span></i></td></tr>
            <tr><td><label for="#merge-format">File format:</label></td><td> <select id="merge-format">
                <option value="safetensors">Safetensors (recommended)</option>
                <option value="ckpt">CKPT (legacy format)</option>
            </select></td>
            <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Use safetensors. It's the better format. Only use ckpt if you want to use<br>the models in legacy software not supporting saftensors.</span></i></td></tr>
            <tr><td>&nbsp;</td></tr>
            <tr><td><label for="#merge-count">Step count:</label></td>
                <td> <input id="merge-count" size=2 value="5"></td>
                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Number of models to create</span></i></td></tr>
            <tr><td><label for="#merge-start">Start batch from:</label></td>
                <td> <input id="merge-start" size=2 value="5">%</td>
                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Smallest share of model A in the mix</span></i></td></tr>
            <tr><td><label for="#merge-step">Step size:</label></td>
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
            <tr><td><label for="#merge-fp">Float precision:</label></td><td> <select id="merge-fp">
                <option value="fp16">fp16 - Half precision (compact file size)</option>
                <option value="fp32">fp32 - Full precision (larger file size)</option>
            </select></td>
                <td> <i class="fa-solid fa-circle-question help-btn"><span class="simple-tooltip top-left">Image generation uses fp16, so it's a good choice.<br>Use fp32 if you want to use the result models for more mixes</span></i></td></tr>
         </table>
         <br/>
         <canvas id="merge-canvas" width="400" height="400"></canvas>
      </div>
      <div class="merge-buttons">
         <button id="merge-button" class="primaryButton">Merge models</button>
      </div>
    </div>`


    /////////////////////// Event Listener
    document.addEventListener('tabClick', (e) => { 
        if (e.detail.name == 'merge') {
	    console.log('Activate')
            let modelList = stableDiffusionModelField.cloneNode(true)
            modelList.id = "mergeModelA"
            modelList.size = 8
            document.querySelector("#mergeModelA").replaceWith(modelList)
            modelList = stableDiffusionModelField.cloneNode(true)
            modelList.id = "mergeModelB"
            modelList.size = 8
            document.querySelector("#mergeModelB").replaceWith(modelList)
            updateChart()
	}
    })

    document.querySelector('.merge-config').addEventListener('change', updateChart)

    document.querySelector('#merge-button').addEventListener('click', async function(e) {
        // Build request template
        let model0 = document.querySelector('#mergeModelA').value
        let model1 = document.querySelector('#mergeModelB').value
        let request = { model0: model0, model1: model1 }
        request['use_fp16'] = document.querySelector('#merge-fp').value == 'fp16'
        let iterations = document.querySelector('#merge-count').value>>0
        let start = parseFloat( document.querySelector('#merge-start').value )
        let step = parseFloat( document.querySelector('#merge-step').value )
        addLogMessage(`start = ${start}%`)
        addLogMessage(`step  = ${step}%`)

        if (start + iterations * (step-1) >= 100) {
            addLogMessage('<i>Aborting: maximum ratio is &#8805; 100%</i>')
            addLogMessage('Reduce the number of steps or the step size')
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
            addLogMessage(`merging batch job ${i+1}/${iterations}, alpha = ${alpha}...`)

            request['out_path'] = document.querySelector('#merge-filename').value
            request['out_path'] += '-' +alpha+ '.' + document.querySelector('#merge-format').value
            addLogMessage(`&nbsp;&nbsp;filename: ${request['out_path']}`)

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
