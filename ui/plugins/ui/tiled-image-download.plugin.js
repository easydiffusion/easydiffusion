;(function(){
    "use strict";
    const PAPERSIZE = [
        {id: "a3p",  width: 297, height: 420, unit: "mm"},
        {id: "a3l",  width: 420, height: 297, unit: "mm"},
        {id: "a4p",  width: 210, height: 297, unit: "mm"},
        {id: "a4l",  width: 297, height: 210, unit: "mm"},
        {id: "ll", width: 279, height: 216, unit: "mm"},
        {id: "lp",  width: 216, height: 279, unit: "mm"},
        {id: "hd", width: 1920, height: 1080, unit: "pixels"},
        {id: "4k", width: 3840, height: 2160, unit: "pixels"},
    ]

    // ---- Register plugin
    PLUGINS['IMAGE_INFO_BUTTONS'].push({ 
        html: '<i class="fa-solid fa-table-cells-large"></i> Download tiled image',
        on_click: onDownloadTiledImage,
        filter: (req, img) => req.tiling != "none",
    })

    var thisImage

    function onDownloadTiledImage(req, img) {
        document.getElementById("download-tiled-image-dialog").showModal()
        thisImage = new Image()
        thisImage.src = img.src
        thisImage.dataset["prompt"] = img.dataset["prompt"]
    }

    // ---- Add HTML
    document.getElementById('container').lastElementChild.insertAdjacentHTML("afterend",
        `<dialog id="download-tiled-image-dialog">
            <h1>Download tiled image</h1>
            <div class="download-tiled-image dtim-container">
                <div class="download-tiled-image-top">
                    <div class="tab-container">
                       <span id="tab-image-tiles" class="tab active">
                           <span>Number of tiles</small></span>
                       </span>
                       <span id="tab-image-size" class="tab">
                           <span>Image dimensions</span>
                       </span>
                    </div>
                    <div>
                        <div id="tab-content-image-tiles" class="tab-content active">
                            <div class="tab-content-inner">
                                <label for="dtim1-width">Width:</label> <input id="dtim1-width" min="1" max="99" type="number" value="2">
                                <label for="dtim1-height">Height:</label> <input id="dtim1-height" min="1" max="99" type="number" value="2">
                            </div>
                        </div>
                        <div id="tab-content-image-size" class="tab-content">
                            <div class="tab-content-inner">
                                <div class="method-2-options">
                                    <label for="dtim2-width">Width:</label> <input id="dtim2-width" size="3" value="1920">
                                    <label for="dtim2-height">Height:</label> <input id="dtim2-height" size="3" value="1080">
                                    <select id="dtim2-unit">
                                        <option>pixels</option>
                                        <option>mm</option>
                                        <option>inches</option>
                                    </select>
                                </div>
                                <div class="method-2-dpi">
                                    <label for="dtim2-dpi">DPI:</label> <input id="dtim2-dpi" size="3" value="72">
                                </div>
                                <div class="method-2-paper">
                                    <i>Some standard sizes:</i><br>
                                    <button id="dtim2-a3p">A3 portrait</button><button id="dtim2-a3l">A3 landscape</button><br>
                                    <button id="dtim2-a4p">A4 portrait</button><button id="dtim2-a4l">A4 landscape</button><br>
                                    <button id="dtim2-lp">Letter portrait</button><button id="dtim2-ll">Letter landscape</button><br>
                                    <button id="dtim2-hd">Full HD</button><button id="dtim2-4k">4K</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="download-tiled-image-placement">
                    <div class="tab-container">
                        <span id="tab-image-placement" class="tab active">
                            <span>Tile placement</span>
                        </span>
                    </div>
                    <div>
                        <div id="tab-content-image-placement" class="tab-content active">
                            <div class="tab-content-inner">
                                <img id="dtim-1tl" class="active" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUCAgKIhobRz89vMJ7s6uo9PDx4d3ewra0bHR1dXV19NrLa2Nj+/f29u7uWlJQuLi7ws27qAAAACXBIWXMAAAsTAAALEwEAmpwYAAABlUlEQVQ4y7VUMU7DQBCckpYCJEpS0ByhcecuUZQUtvIHGku0vICSDtHkA9eltCylOEBKFInCRworXToK3kDJ7jpn2SYmgGESeWyPRuudvTugHTyC72momKDGMMJDLIhmgK+nWmuPXxtlxkhjExszRKqU6uRuTW7TYTwh6HTpR25+JLcngBJ5jL5wIecqu9nFbid3t27N7vhrtypqV2SfP4zc5pfu/Msb3P6U4fru1eXpVg7tcmnDZ1gb0s1ceAEcSPI3uM2B9xLf7Z3YLlfJ/WCppF1QbbqxeW0brlztjXzprBhJrW8nu4HWGlt/xz1qcrervfmT2ma3WxpTjfK5ZUioNg+VsUL+tiXuI8YJLrd8KHyENyaqPWC8QGiwwlJ4LtyvNtb9vFKrqZXXeebkrEiN3ZUNXHJnO3aJkxt2aH2gDRNTLdyzJvee1CZXUTSJrhA55itlfszUdqDrxCQmGIEu9KfFFCRJYnpIgyB4JJlPWM6cY6MjN+UW5MjdM7FKavF/pFbfRD9zv8rjBa6FT5EJn0HoA8lOiD4+8B3mAAAAAElFTkSuQmCC" />
                                <img id="dtim-1tr" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUCAgKIhobRz89vMJ7s6uo9PDx4d3ewra0bHR1dXV19NrLa2Nj+/f29u7uWlJQuLi7ws27qAAAACXBIWXMAAAsTAAALEwEAmpwYAAABoUlEQVQ4y61UsU7CUBQ9o6uGmJgwNSQOLS7dOndrGfwEDWnC4tQNB8duxoUfYGN86fZgIWy+iVUnwwYf4Oi9lxa0tFSpB9IDXM6775773gUaYWjbtoO+IrI1VsIKnjt2CYCllJqir7Wt7SlWWmn+m+t53sQbU5hAamtJrxRr/mppUnuZOgszOlgJK7gCUS93YVbzKqx2q9U2q71Mbf1Qbxc/qqadu7y509W7nX8Pt/K6JwwKO+HCGLRNKPy4oA9mkYUnwGeSJM9IBDknOJN+PNV2rEy9XyXLvaGcktuY0FBux9AP5rVYd96SofCsWFje0NJwUd2rUse/UTfLPTspd83iFFZZYY4xbKoRsKlmaypjjoaICA+4ZYrO8SJ8mfEV0PF9P0Tb94U3wj7eheaHJ3VZLKxbcs4P6uartz/nMYlKbFnzYtWe5ze0wtSjDd1Ph7iReheucS0aRYM78pwoiiDPUc6Dpg19S9N0ipYOgiClw5TqgN6I6aGD7S2RkcsbppnKbLPnPHt7VZOpxvN/cq1cHf9BbeFeqIsL4Wt8CN/gC1XPfwv6U6jJAAAAAElFTkSuQmCC" /><br>
                                <img id="dtim-1bl" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUCAgKIhobRz89vMJ7s6uo9PDx4d3ewra0bHR1dXV19NrLa2Nj+/f29u7uWlJQuLi7ws27qAAAACXBIWXMAAAsTAAALEwEAmpwYAAABjElEQVQ4y7WUsU7DMBCG/5GVoUiMMLC4ZcmWrVXVDq7yDiyRuvIEjGyIpS/graMVKYMBqVUkhoYOUTc2Bp6BEd+5qdI2SasGLpE/KZc/d76LD/i6JrvFPfMKGfMGDOCTGUMwA/SYA8yL7iEM8w2yzH1AHderh9D1alGuXmmtBaVGchFgQe8J69bOHZnIyCHsYu8AUkZRZLpYSClf0dAm4zCchGOEOWkNW7ggAO0+2QcY/SUS5ozZ+6uqNVHHR9Q8q1Bnm9+B1B17HdGxbDe1zn7sA1V7DskucbfmObOFb0LThrZTsjlGzHcw0iXcU6woQxFv9i3rah5UdSxvSa/+EMn6hp6iroy9+s/YL2mSjlxRE1fUkX2wZNqiPjrDTwmfDnbsjNeH0q9Y9ZRN2diJjeliJ+ksj+2v3W5j3d2N+R5b4T/fcjuvqppMvqfsdbqaxF7VCc3Vho9eUd0pZi6q1cpThemwdYB9NVVK2cy1NsLYmaqNNmZgZ6sQa7VPmWuavQG9ZkkjV+u42QH8BWe+iD71TSARAAAAAElFTkSuQmCC" />
                                <img id="dtim-1br" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUCAgKIhobRz89vMJ7s6uo9PDx4d3ewra0bHR1dXV19NrLa2Nj+/f29u7uWlJQuLi7ws27qAAAACXBIWXMAAAsTAAALEwEAmpwYAAABkklEQVQ4y7VULU/DUBQ9EgtZSEimliWIt2Lmnq5rJ/gJkKXJDGquCGQdwewPzE2+PPeGWeaomgVF6tgPmOTe2zWkox8LC6dLT/ZuTs9957UXPQbuhTxcCF/jU/gGYBpgLH/7yIQNYuHXctniS9hhWlU+Wh03qVX1wwt1+eGKyoZXl8iYFbdmlFIj4N1au0THBUFgLTLrAvphSjcXUPk0RLNocodbpiiC3GcFT4C+7/shur4vvBX28SG0+o/UTHNqrZnXe3uVrW3oKtScuVeUvZYTK3lvyq21pBYRHihzxjlehC/3fHXqgQ5SArapAI9C63w1XR3GUuw75neuN6otV++78SOyza9Dv/lAj1Yf5T061XsQrjnUdSpMoYZ5qLSQvgG7JEmekQgOOWk9sV2l6kxqT4V3Nw1zb7IMyVsvBIcb6+w3lpfn1V+Jw1Awr5tMOq/XqzVdf1Mr9tZ7701JzZ+ia1ab352z6mc6cGO52hiapWPnlFM0U51ximbqUGu90KSOabTyyCWi5UyYO5/n6pPwDYr8fwvXgN7jAAAAAElFTkSuQmCC" /> <br>
                                <img id="dtim-1center" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUBAQGPjY3Rz89DQ0O0sbHz8vJZJoFwbm6+vLxBHV10MqY6ODifnZ1XV1cdHR3c2trZIbFLAAAACXBIWXMAAAsTAAALEwEAmpwYAAACCElEQVQ4y7VVPWvjQBCVnZXt5JRT1LsQwwnS7YHrgLg6hTBsva4MSeUurX5CqqtTuXCVn6CAf0DgimvvJ4iFredmd2VZcqTYIWSap2V4+/HezMgDSAdgIw0cFhVeMQAPIAuUWfF44jAdOix8k8YsmCNXyEuGynzkQ4u68BE8LbKJsJGFDkvfYe6Lufe5ePi7unywUePY4Suhl4iMCaGlFOWQQGiRhw5Tc3b9MIiZe1Deehhkg3nXu5MirGW5NmnmkNjX9EFs1VKN7djgNg+bovae3a15cpg+ZIfquGP97J2h0viJ5cT6SoZaX+lspYyoX2doFBXjlyiaRtHV+XL5++5u+TiITExXQRR5XGeBQIkoyouNjadQSqm1Tn2pmg8bbe5NelFb0nQMKL25XxO7tsSoxmu/q80XbdXK/eZrs/2iR/OR4a53m1eO8ayPfWDJqHU2VuwywNbVnt6czflbttucVGOkmUbdUI2WSG0g0ZvepuNbiu22IM1NPAZ2uV2x7csnDf3FywFdhQPmttYUT6kkubv5D4+bWlOoFNWa5BRJHiJNDzTDQzUd21fqoaHdPaaO1/kp/d1mwwkdunfsIx2K7Q6lapmDIrVihqAsG41qiZ0tbq7ZFhw6jKsWtHPtz+z128zGT4c3z2cVXs5ujjn2773k9472j5vtf/pkYp2TKe/7E00A/gO7G7pwJRGqtAAAAABJRU5ErkJggg==" />
                                <img id="dtim-4center" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAA3BAMAAACiFTSCAAAAMFBMVEUBAQGbmppqLZZ8fHzRz8+LPMZ0MqXs6upTUVE+G1q5ublubm6iRedQInH9/Pze3NzAiv2WAAAACXBIWXMAAAsTAAALEwEAmpwYAAABuUlEQVQ4y6VVvU7CUBSmCEVQao0vQDrcWWcWhiYkDsICc3OS3pmhSbfiI7C6MfgAJo4sPoCDcSdhMGFDBhI24zn3tBjh2DTy3eEj9+P0/Pa05BtMrZQbzG7KJaZFIyLqrS82hqfOnszXnUquLFv3dvLOeiP57uT69teFfHcP5CKJ+QdyN0YsF5Ul8WpUI06SjmM4Ks0Mrk+Zn86Y70+YS8fha0VY2K8GL1XmwDHXnxj54y1GXgbwwNOBrT2lPNWiyFHIEitrhYdkDaB1ay/vsjY6yiiCJJMVykq0pkt+OPoQH44gGX8IskqtFYjWgODQQAwt8y1aAygPdN8GCkHt5LSoaTH75ylnRT0ON5cEtz4nvL81Dc8nlrm+wmEax9t4YQ/0MNRhn1iHYcuJt+PxOMoib5p8uGpySzAfUxbQKqcsSov9VmlRqSPyOHBR6Q9Sx9haKci3lvsNPGtyv8Hogc3c+nkFl3iwasNBOBz0q8yBk8RJvMRXsE3HrT8YTDK2Zs942kc29I6npbZi5ilZjZg/DpaHL++WqNDiypfzt+LfO3VTaO39c6dmsp9nPZJ9Z18i19r8+hJ9A3EAErhB3eXkAAAAAElFTkSuQmCC" /> <br>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="dtim-ok">
                    <button class="primaryButton" id="dti-ok">Download</button>
                </div>
                <div class="dtim-newtab">
                    <button class="primaryButton" id="dti-newtab">Open in new tab</button>
                </div>
                <div class="dtim-cancel">
                    <button class="primaryButton" id="dti-cancel">Cancel</button>
                </div>
            </div>
         </dialog>`)

    let downloadTiledImageDialog = document.getElementById("download-tiled-image-dialog")
    let dtim1_width = document.getElementById("dtim1-width")
    let dtim1_height = document.getElementById("dtim1-height")
    let dtim2_width = document.getElementById("dtim2-width")
    let dtim2_height = document.getElementById("dtim2-height")
    let dtim2_unit = document.getElementById("dtim2-unit")
    let dtim2_dpi = document.getElementById("dtim2-dpi")
    let tabTiledTilesOptions = document.getElementById("tab-image-tiles")
    let tabTiledSizeOptions = document.getElementById("tab-image-size")

    linkTabContents(tabTiledTilesOptions)
    linkTabContents(tabTiledSizeOptions)

    prettifyInputs(downloadTiledImageDialog)

    // ---- Predefined image dimensions
    PAPERSIZE.forEach( function(p) {
        document.getElementById("dtim2-" + p.id).addEventListener("click", (e) => {
            dtim2_unit.value = p.unit
            dtim2_width.value = p.width
            dtim2_height.value = p.height
        })
    })

    // ---- Close popup
    document.getElementById("dti-cancel").addEventListener("click", (e) => downloadTiledImageDialog.close())
    downloadTiledImageDialog.addEventListener('click', function (event) {
        var rect = downloadTiledImageDialog.getBoundingClientRect();
        var isInDialog=(rect.top <= event.clientY && event.clientY <= rect.top + rect.height
          && rect.left <= event.clientX && event.clientX <= rect.left + rect.width);
        if (!isInDialog) {
            downloadTiledImageDialog.close();
        }
    });

    // ---- Stylesheet
    const styleSheet = document.createElement("style")
    styleSheet.textContent = `
        dialog {
          background: var(--background-color2);
          color: var(--text-color);
          border-radius: 7px;
          border: 1px solid var(--background-color3);
        }

        dialog::backdrop {
          background: rgba(0, 0, 0, 0.5);
        }


        button[disabled] {
          opacity: 0.5;
        }

        .method-2-dpi {
          margin-top: 1em;
          margin-bottom: 1em;
        }

        .method-2-paper button {
          width: 10em;
          padding: 4px;
          margin: 4px;
        }
        
        .download-tiled-image .tab-content {
          background: var(--background-color1);
          border-radius: 3pt;
        }

        .dtim-container {  display: grid;
          grid-template-columns: auto auto;
          grid-template-rows: auto auto;
          gap: 1em 0px;
          grid-auto-flow: row;
          grid-template-areas:
            "dtim-tab dtim-tab dtim-plc"
            "dtim-ok dtim-newtab dtim-cancel";
        }

        .download-tiled-image-top {
          justify-self: center;
          grid-area: dtim-tab;
        }

        .download-tiled-image-placement {
          justify-self: center;
          grid-area: dtim-plc;
          margin-left: 1em;
        }

        .dtim-ok {
          justify-self: center;
          align-self: start;
          grid-area: dtim-ok;
        }

        .dtim-newtab {
          justify-self: center;
          align-self: start;
          grid-area: dtim-newtab;
        }

        .dtim-cancel {
          justify-self: center;
          align-self: start;
          grid-area: dtim-cancel;
        }

        #tab-content-image-placement img {
          margin: 4px;
          opacity: 0.3;
          border: solid 2px var(--background-color1);
        }

        #tab-content-image-placement img:hover {
          margin: 4px;
          opacity: 1;
          border: solid 2px var(--accent-color);
          filter: brightness(2);
        }

        #tab-content-image-placement img.active {
          margin: 4px;
          opacity: 1;
          border: solid 2px var(--background-color1);
        }

    `
    document.head.appendChild(styleSheet)

    // ---- Placement widget

    function updatePlacementWidget(event) {
        document.querySelector("#tab-content-image-placement img.active").classList.remove("active")
        event.target.classList.add("active")
    }

    document.querySelectorAll("#tab-content-image-placement img").forEach(
        (i) => i.addEventListener("click", updatePlacementWidget)
    )

    function getPlacement() {
        return document.querySelector("#tab-content-image-placement img.active").id.substr(5)
    }
        
    // ---- Make the image
    function downloadTiledImage(image, width, height, offsetX=0, offsetY=0, new_tab=false) {

        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d')

        const w = image.width
        const h = image.height

        for (var x = offsetX; x < width; x += w) {
            for (var y = offsetY; y < height; y += h) {
                context.drawImage(image, x, y, w, h)
            }
        }
        if (new_tab) {
            var newTab = window.open("")
            newTab.document.write(`<html><head><title>${width}×${height}, "${image.dataset["prompt"]}"</title></head><body><img src="${canvas.toDataURL()}"></body></html>`)
        } else {
            const link = document.createElement('a')
            link.href = canvas.toDataURL()
            link.download = image.dataset["prompt"].replace(/[^a-zA-Z0-9]+/g, "-").substr(0,22)+crypto.randomUUID()+".png"
            link.click()
        }
    }

    function onDownloadTiledImageClick(e, newtab=false) {
        var width, height, offsetX, offsetY

        if (isTabActive(tabTiledTilesOptions)) {
            width = thisImage.width * dtim1_width.value
            height = thisImage.height * dtim1_height.value
        } else {
            if ( dtim2_unit.value == "pixels" ) {
                width = dtim2_width.value
                height= dtim2_height.value
            } else if ( dtim2_unit.value == "mm" ) {
                width = Math.floor( dtim2_width.value * dtim2_dpi.value / 25.4 )
                height = Math.floor( dtim2_height.value * dtim2_dpi.value / 25.4 )
            } else { // inch
                width = Math.floor( dtim2_width.value * dtim2_dpi.value )
                height = Math.floor( dtim2_height.value * dtim2_dpi.value )
            }
        }

        var placement = getPlacement()
        if (placement == "1tl") {
            offsetX = 0
            offsetY = 0
        } else if (placement == "1tr") {
            offsetX = width - thisImage.width * Math.ceil( width / thisImage.width )
            offsetY = 0
        } else if (placement == "1bl") {
            offsetX = 0
            offsetY = height - thisImage.height * Math.ceil( height / thisImage.height )
        } else if (placement == "1br") {
            offsetX = width - thisImage.width * Math.ceil( width / thisImage.width )
            offsetY = height - thisImage.height * Math.ceil( height / thisImage.height )
        } else if (placement == "4center") {
            offsetX = width/2 - thisImage.width * Math.ceil( width/2 / thisImage.width )
            offsetY = height/2 - thisImage.height * Math.ceil( height/2 / thisImage.height )
        } else if (placement == "1center") {
            offsetX = width/2 - thisImage.width/2 - thisImage.width * Math.ceil( (width/2 - thisImage.width/2) / thisImage.width )
            offsetY = height/2 - thisImage.height/2 - thisImage.height * Math.ceil( (height/2 - thisImage.height/2) / thisImage.height )
        }
        downloadTiledImage(thisImage, width, height, offsetX, offsetY, newtab)
        downloadTiledImageDialog.close()
    }

    document.getElementById("dti-ok").addEventListener("click", onDownloadTiledImageClick)
    document.getElementById("dti-newtab").addEventListener("click", (e) => onDownloadTiledImageClick(e,true))

})()
