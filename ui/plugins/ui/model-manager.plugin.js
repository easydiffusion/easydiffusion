"use strict"


class sduiTab {
    constructor(name, label, icon) {
        this.name = name
        this.label = label
        this.icon = icon
        this.render()
    }

    setContent(content) {
        this.content.innerHTML = content
    }

    render() {
        document.querySelector('#tab-container').insertAdjacentHTML('beforeend', `
            <span id="tab-${this.name}" class="tab">
                <span><i class="fa ${this.icon} icon"></i> ${this.label}</span>
            </span>
        `)

        document.querySelector('#tab-content-wrapper').insertAdjacentHTML('beforeend', `
            <div id="tab-content-${this.name}" class="tab-content">
            </div>
        `)
        this.content = document.querySelector(`#tab-content-${this.name}`)
        linkTabContents(document.querySelector(`#tab-${this.name}`))
    }
}

let models=null;

(async function() {

    let tab = new sduiTab('modelmgr', "Model Manager", "fa-folder-tree")
    let token = {prefix:{},suffix:{}} 
    let t = localStorage.getItem('modelToken')
    if (t!=null) {
        token = JSON.parse(t)
    }

    async function updateModels() {
        let res = await fetch('/get/models')
        let models = await res.json()
        console.log(models)

        let content = `<div id="modelmgr" class="tab-content-inner" style="text-align:left;">`

	// Stable Diffusion
	content += `<div id="modelmgr-sd">`

        content += '<h4>Stable Diffusion Models</h4>'
        content += '<button class="primaryButton">Add model</button>'
        models["options"]["stable-diffusion"].forEach( (m) => { 
            content += `
	       <div class="panel-box">
	         <div><i class="fa fa-file-code icon"></i> ${m}</div>
	         <div style="padding-left:2em;">
	           <table>
                     <tr><th><i class="fa fa-backward-step icon"></i> Prefix token:</th><td><input size="25" data-model="${m}" data-type="prefix" value="${token["prefix"][m] || ""}"><td></tr>
                     <tr><th><i class="fa fa-forward-step icon"></i> Suffix token:</th><td><input size="25" data-model="${m}" data-type="suffix"i value="${token["suffix"][m] || ""}"><td></tr>
	           </table>
	         </div>
	       </div>
	    `
        })

        content += `</div>`

        // TODO Other models
        content += `</div>`
        tab.setContent(content)
    }

    function addModelToken(prompts) { 
        let model = stableDiffusionModelField.value
	let prefix = ""
	let suffix = ""
	if( token["prefix"][model] != "") {
	    prefix = token["prefix"][model] + ", "
	}
	if( token["suffix"][model] != "") {
	    suffix = ", " + token["suffix"][model]
	}

        return prompts.map( x => prefix + x + suffix )
    }

    await updateModels()

    document.querySelector('#modelmgr-sd').onchange=function(){
         document.querySelectorAll('#modelmgr-sd input').forEach( i => {
             token[i.dataset.type][i.dataset.model] = i.value
         })
         localStorage.setItem('modelToken', JSON.stringify(token))
    }

    PLUGINS['GET_PROMPTS_HOOK'].push(addModelToken)

})()


