var editorControlsLeft = document.getElementById("image-editor-controls-left")

const IMAGE_EDITOR_MAX_SIZE = 800

const IMAGE_EDITOR_BUTTONS = [
	{
		name: "Clear",
		icon: "fa-solid fa-xmark",
		handler: editor => {
			editor.clear()
		}
	},
	{
		name: "Cancel",
		icon: "fa-regular fa-circle-xmark",
		handler: editor => {
			editor.close()
		}
	},
	{
		name: "Save",
		icon: "fa-solid fa-floppy-disk",
		handler: editor => {
			editor.saveImage()
		}
	}
]

const IMAGE_EDITOR_TOOLS = [
	{
		id: "draw",
		name: "Draw",
		icon: "fa-solid fa-pencil"
	},
	{
		id: "erase",
		name: "Erase",
		icon: "fa-solid fa-eraser"
	}
]

var IMAGE_EDITOR_SECTIONS = [
	{
		name: "tool",
		title: "Tool",
		default: "draw",
		options: Array.from(IMAGE_EDITOR_TOOLS.map(t => t.id)),
		initElement: (element, option) => {
			var tool_info = IMAGE_EDITOR_TOOLS.find(t => t.id == option)
			element.className = "image-editor-button button"
			var sub_element = document.createElement("div")
			var icon = document.createElement("i")
			tool_info.icon.split(" ").forEach(c => icon.classList.add(c))
			sub_element.appendChild(icon)
			sub_element.append(tool_info.name)
			element.appendChild(sub_element)
		}
	},
	{
		name: "color",
		title: "Color",
		default: "#f1c232",
		options: [
			"custom",
			"#ea9999", "#e06666", "#cc0000", "#990000", "#660000",
			"#f9cb9c", "#f6b26b", "#e69138", "#b45f06", "#783f04",
			"#ffe599", "#ffd966", "#f1c232", "#bf9000", "#7f6000",
			"#b6d7a8", "#93c47d", "#6aa84f", "#38761d", "#274e13",
			"#a4c2f4", "#6d9eeb", "#3c78d8", "#1155cc", "#1c4587",
			"#b4a7d6", "#8e7cc3", "#674ea7", "#351c75", "#20124d",
			"#d5a6bd", "#c27ba0", "#a64d79", "#741b47", "#4c1130",
			"#ffffff", "#c0c0c0", "#838383", "#525252", "#000000",
		],
		initElement: (element, option) => {
			if (option == "custom") {
				var input = document.createElement("input")
				input.type = "color"
				element.appendChild(input)
				var span = document.createElement("span")
				span.textContent = "Custom"
				element.appendChild(span)
			}
			else {
				element.style.background = option
			}
		},
		getCustom: editor => {
			var input = editor.popup.querySelector(".image_editor_color input")
			return input.value
		}
	},
	{
		name: "brush_size",
		title: "Brush Size",
		default: 48,
		options: [ 16, 24, 32, 48, 64 ],
		initElement: (element, option) => {
			element.parentElement.style.flex = option
			element.style.width = option + "px"
			element.style.height = option + "px"
			element.style["border-radius"] = (option / 2).toFixed() + "px"
		}
	},
	{
		name: "opacity",
		title: "Opacity",
		default: 0,
		options: [ 0, 0.25, 0.5, 0.75, 1 ],
		initElement: (element, option) => {
			element.style.background = `repeating-conic-gradient(rgba(0, 0, 0, ${option}) 0% 25%, rgba(255, 255, 255, ${option}) 0% 50%) 50% / 10px 10px`
		}
	},
	{
		name: "sharpness",
		title: "Sharpness",
		default: 0.05,
		options: [ 0, 0.05, 0.1, 0.2, 0.3 ],
		initElement: (element, option) => {
			var size = 32
			var blur_amount = parseInt(option * size)
			var sub_element = document.createElement("div")
			sub_element.style.background = `var(--background-color3)`
			sub_element.style.filter = `blur(${blur_amount}px)`
			sub_element.style.width = `${size - 4}px`
			sub_element.style.height = `${size - 4}px`
			sub_element.style['border-radius'] = `${size}px`
			element.style.background = "none"
			element.appendChild(sub_element)
		}
	}
]

class ImageEditor {
	constructor(popup, inpainter = false) {
		this.inpainter = inpainter
		this.popup = popup
		if (inpainter) {
			this.popup.classList.add("inpainter")
		}
		this.drawing = false
		this.dropper_active = false
		this.container = popup.querySelector(".editor-controls-center > div")
		this.cursor_icon = document.createElement("i")
		this.layers = {}
		var layer_names = [
			"background",
			"drawing",
			"overlay"
		]
		layer_names.forEach(name => {
			let canvas = document.createElement("canvas")
			canvas.className = `editor-canvas-${name}`
			this.container.appendChild(canvas)
			this.layers[name] = {
				name: name,
				canvas: canvas,
				ctx: canvas.getContext("2d")
			}
		})

		this.setSize(512, 512)

		this.cursor_icon.classList.add("cursor-icon")
		this.container.appendChild(this.cursor_icon)

		// add mouse handlers
		this.container.addEventListener("mousedown", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseup", this.mouseHandler.bind(this))
		this.container.addEventListener("mousemove", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseout", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseenter", this.mouseHandler.bind(this))
		// setup forwarding for keypresses so the eyedropper works accordingly
		var mouseHandlerHelper = this.mouseHandler.bind(this)
		this.container.addEventListener("mouseenter",function() {
			document.addEventListener("keyup", mouseHandlerHelper)
			document.addEventListener("keydown", mouseHandlerHelper)
		})
		this.container.addEventListener("mouseout",function() {
			document.removeEventListener("keyup", mouseHandlerHelper)
			document.removeEventListener("keydown", mouseHandlerHelper)
		})

		// initialize editor controls
		this.options = {}
		IMAGE_EDITOR_SECTIONS.forEach(section => {
			section.id = `image_editor_${section.name}`
			var sectionElement = document.createElement("div")
			sectionElement.className = section.id
	
			var title = document.createElement("h4")
			title.innerText = section.title
			sectionElement.appendChild(title)
	
			var optionsContainer = document.createElement("div")
			optionsContainer.classList.add("editor-options-container")
	
			section.optionElements = []
			section.options.forEach((option, index) => {
				var optionHolder = document.createElement("div")
				var optionElement = document.createElement("div")
				optionHolder.appendChild(optionElement)
				section.initElement(optionElement, option)
				optionElement.addEventListener("click", target => this.selectOption(section.name, index))
				optionsContainer.appendChild(optionHolder)
				section.optionElements.push(optionElement)
			})
			this.selectOption(section.name, section.options.indexOf(section.default))
	
			sectionElement.appendChild(optionsContainer)
	
			this.popup.querySelector(".editor-controls-left").appendChild(sectionElement)
		})

		this.custom_color_input = this.popup.querySelector(`input[type="color"]`)
		this.custom_color_input.addEventListener("change", () => {
			this.custom_color_input.parentElement.style.background = this.custom_color_input.value
			this.selectOption("color", 0)
		})

		if (this.inpainter) {
			this.selectOption("color", IMAGE_EDITOR_SECTIONS.find(s => s.name == "color").options.indexOf("#ffffff"))
		}

		// initialize the right-side controls
		var buttonContainer = document.createElement("div")
		IMAGE_EDITOR_BUTTONS.forEach(button => {
			var element = document.createElement("div")
			var icon = document.createElement("i")
			element.className = "image-editor-button button"
			icon.className = button.icon
			element.appendChild(icon)
			element.append(button.name)
			buttonContainer.appendChild(element)
			element.addEventListener("click", event => button.handler(this))
		})
		this.popup.querySelector(".editor-controls-right").appendChild(buttonContainer)
	}
	setSize(width, height) {
		if (width == this.width && height == this.height) {
			return
		}

		var max_size = Math.min(window.innerWidth, width, 768)
		if (width > height) {
			var multiplier = max_size / width
			width = (multiplier * width).toFixed()
			height = (multiplier * height).toFixed()
		}
		else {
			var multiplier = max_size / height
			width = (multiplier * width).toFixed()
			height = (multiplier * height).toFixed()
		}
		this.width = width
		this.height = height
	
		this.container.style.width = width + "px"
		this.container.style.height = height + "px"
		
		Object.values(this.layers).forEach(layer => {
			layer.canvas.width = width
			layer.canvas.height = height
		})

		if (this.inpainter) {
			this.saveImage() // We've reset the size of the image so inpainting is different
		}
		this.setBrush()
	}
	setCursorIcon(icon_class = null) {
		if (icon_class == null) {
			var tool = this.getOptionValue("tool")
			icon_class = IMAGE_EDITOR_TOOLS.find(t => t.id == tool).icon
		}
		this.cursor_icon.className = `cursor-icon ${icon_class}`
	}
	setImage(url, width, height) {
		this.setSize(width, height)
		this.layers.drawing.ctx.clearRect(0, 0, this.width, this.height)
		this.layers.background.ctx.clearRect(0, 0, this.width, this.height)
		if (url) {
			var image = new Image()
			image.onload = () => { 
				this.layers.background.ctx.drawImage(image, 0, 0, this.width, this.height)
			}
			image.src = url
		}
		else {
			this.layers.background.ctx.fillStyle = "#ffffff"
			this.layers.background.ctx.beginPath()
			this.layers.background.ctx.rect(0, 0, this.width, this.height)
			this.layers.background.ctx.fill()
		}
	}
	saveImage() {
		if (!this.inpainter) {
			// This is not an inpainter, so save the image as the new img2img input
			this.layers.background.ctx.drawImage(this.layers.drawing.canvas, 0, 0, this.width, this.height)
			var base64 = this.layers.background.canvas.toDataURL()
			initImagePreview.src = base64 // this will trigger the rest of the app to use it
		}
		else {
			// This is an inpainter, so make sure the toggle is set accordingly
			var is_blank = !this.layers.drawing.ctx
				.getImageData(0, 0, this.width, this.height).data
				.some(channel => channel !== 0)
			maskSetting.checked = !is_blank
		}
		this.close()
	}
	getImg() { // a drop-in replacement of the drawingboard version
		return this.layers.drawing.canvas.toDataURL()
	}
	close() {
		this.popup.classList.remove("active")
	}
	clear() {
		this.ctx_current.clearRect(0, 0, this.width, this.height)
	}
	get eraser_active() {
		return this.getOptionValue("tool") == "erase"
	}
	setBrush(layer = null) {
		if (layer) {
			layer.ctx.lineCap = "round"
			layer.ctx.lineJoin = "round"
			layer.ctx.lineWidth = this.getOptionValue("brush_size")
			layer.ctx.fillStyle = this.getOptionValue("color")
			layer.ctx.strokeStyle = this.getOptionValue("color")
			var sharpness = parseInt(this.getOptionValue("sharpness") * this.getOptionValue("brush_size"))
			layer.ctx.filter = sharpness == 0 ? `none` : `blur(${sharpness}px)`
			layer.ctx.globalAlpha = (1 - this.getOptionValue("opacity"))
			layer.ctx.globalCompositeOperation = this.eraser_active ? "destination-out" : "source-over"
		}
		else {
			Object.values([ "drawing", "overlay" ]).map(name => this.layers[name]).forEach(l => {
				this.setBrush(l)
			})
		}
	}
	get ctx_overlay() {
		return this.layers.overlay.ctx
	}
	get ctx_current() { // the idea is this will help support having custom layers and editing each one
		return this.layers.drawing.ctx
	}
	get canvas_current() {
		return this.layers.drawing.canvas
	}
	mouseHandler(event) {
		var bbox = this.layers.overlay.canvas.getBoundingClientRect()
		var x = (event.clientX || 0) - bbox.left
		var y = (event.clientY || 0) - bbox.top
	
		// do drawing-related stuff
		if (event.type == "mousedown" || (event.type == "mouseenter" && event.buttons == 1)) {
			if (this.dropper_active) {
				var img_rgb = this.layers.background.ctx.getImageData(x, y, 1, 1).data
				var drw_rgb = this.ctx_current.getImageData(x, y, 1, 1).data
				var drw_opacity = drw_rgb[3] / 255
				var test = rgbToHex({ 
					r: (drw_rgb[0] * drw_opacity) + (img_rgb[0] * (1 - drw_opacity)),
					g: (drw_rgb[1] * drw_opacity) + (img_rgb[1] * (1 - drw_opacity)),
					b: (drw_rgb[2] * drw_opacity) + (img_rgb[2] * (1 - drw_opacity)),
				})
				this.custom_color_input.value = test
				this.custom_color_input.dispatchEvent(new Event("change"))
			}
			else {
				this.drawing = true
				this.ctx_overlay.beginPath()
				this.ctx_overlay.moveTo(x, y)
				this.ctx_current.beginPath()
				this.ctx_current.moveTo(x, y)
			}
		}
		if (event.type == "mouseup" || event.type == "mousemove") {
			if (this.drawing) {
				this.ctx_current.lineTo(x, y)
				this.ctx_overlay.lineTo(x, y)

				// This isnt super efficient, but its the only way ive found to have clean updating for the drawing
				this.ctx_overlay.clearRect(0, 0, this.width, this.height)
				if (this.eraser_active) {
					this.ctx_overlay.globalCompositeOperation = "source-over"
					this.ctx_overlay.globalAlpha = 1
					this.ctx_overlay.filter = "none"
					this.ctx_overlay.drawImage(this.canvas_current, 0, 0)
					this.setBrush(this.layers.overlay)
					this.canvas_current.style.opacity = 0
				}

				this.ctx_overlay.stroke()
			}
		}
		if (event.type == "mouseup" || event.type == "mouseout") {
			if (this.drawing) {
				this.drawing = false
				this.ctx_current.stroke()
				this.ctx_overlay.clearRect(0, 0, this.width, this.height)

				if (this.eraser_active) {
					this.canvas_current.style.opacity = ""
				}
			}
		}

		// cursor-icon stuff
		if (event.type == "mousemove") {
			this.cursor_icon.style.left = `${x + 10}px`
			this.cursor_icon.style.top = `${y + 20}px`
		}
		if (event.type == "mouseenter") {
			this.cursor_icon.style.opacity = 1
		}
		if (event.type == "mouseout") {
			this.cursor_icon.style.opacity = 0
		}
		if ([ "mouseenter", "mousemove", "keydown", "keyup" ].includes(event.type)) {
			if (this.dropper_active && !event.ctrlKey) {
				this.dropper_active = false
				this.setCursorIcon()
			}
			else if (!this.dropper_active && event.ctrlKey) {
				this.dropper_active = true
				this.setCursorIcon("fa-solid fa-eye-dropper")
			}
		}
	}
	getOptionValue(section_name) {
		var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name)
		return this.options && section_name in this.options ? this.options[section_name] : section.default
	}
	selectOption(section_name, option_index) {
		var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name)
		var value = section.options[option_index]
		this.options[section_name] = value == "custom" ? section.getCustom(this) : value
	
		section.optionElements.forEach(element => element.classList.remove("active"))
		section.optionElements[option_index].classList.add("active")
	
		// change the editor
		this.setBrush()
		if (section.name == "tool") {
			this.setCursorIcon()
		}
	}
}

function rgbToHex(rgb) {
	function componentToHex(c) {
		var hex = parseInt(c).toString(16)
		return hex.length == 1 ? "0" + hex : hex
	}
	return "#" + componentToHex(rgb.r) + componentToHex(rgb.g) + componentToHex(rgb.b)
}

const imageEditor = new ImageEditor(document.getElementById("image-editor"))
const imageInpainter = new ImageEditor(document.getElementById("image-inpainter"), true)

imageEditor.setImage(null, 512, 512)
imageInpainter.setImage(null, 512, 512)

document.getElementById("init_image_button_draw").addEventListener("click", () => {
	document.getElementById("image-editor").classList.toggle("active")
})
document.getElementById("init_image_button_inpaint").addEventListener("click", () => {
	document.getElementById("image-inpainter").classList.toggle("active")
})