var editorControlsLeft = document.getElementById("image-editor-controls-left")

const IMAGE_EDITOR_MAX_SIZE = 800

const IMAGE_EDITOR_BUTTONS = [
	{
		name: "Cancel",
		icon: "fa-regular fa-circle-xmark",
		handler: editor => {
			editor.hide()
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

const defaultToolBegin = (editor, ctx, x, y, is_overlay = false) => {
	ctx.beginPath()
	ctx.moveTo(x, y)
}
const defaultToolMove = (editor, ctx, x, y, is_overlay = false) => {
	ctx.lineTo(x, y)
	if (is_overlay) {
		ctx.clearRect(0, 0, editor.width, editor.height)
		ctx.stroke()
	}
}
const defaultToolEnd = (editor, ctx, x, y, is_overlay = false) => {
	ctx.stroke()
	if (is_overlay) {
		ctx.clearRect(0, 0, editor.width, editor.height)
	}
}
const toolDoNothing = (editor, ctx, x, y, is_overlay = false) => {}

const IMAGE_EDITOR_TOOLS = [
	{
		id: "draw",
		name: "Draw",
		icon: "fa-solid fa-pencil",
		cursor: "url(/media/images/fa-pencil.svg) 0 24, pointer",
		begin: defaultToolBegin,
		move: defaultToolMove,
		end: defaultToolEnd
	},
	{
		id: "erase",
		name: "Erase",
		icon: "fa-solid fa-eraser",
		cursor: "url(/media/images/fa-eraser.svg) 0 14, pointer",
		begin: defaultToolBegin,
		move: (editor, ctx, x, y, is_overlay = false) => {
			ctx.lineTo(x, y)
			if (is_overlay) {
				ctx.clearRect(0, 0, editor.width, editor.height)
				ctx.globalCompositeOperation = "source-over"
				ctx.globalAlpha = 1
				ctx.filter = "none"
				ctx.drawImage(editor.canvas_current, 0, 0)
				editor.setBrush(editor.layers.overlay)
				ctx.stroke()
				editor.canvas_current.style.opacity = 0
			}
		},
		end: (editor, ctx, x, y, is_overlay = false) => {
			ctx.stroke()
			if (is_overlay) {
				ctx.clearRect(0, 0, editor.width, editor.height)
				editor.canvas_current.style.opacity = ""
			}
		},
		setBrush: (editor, layer) => {
			layer.ctx.globalCompositeOperation = "destination-out"
		}
	},
	{
		id: "fill",
		name: "Fill",
		icon: "fa-solid fa-fill",
		cursor: "url(/media/images/fa-fill.svg) 20 6, pointer",
		begin: (editor, ctx, x, y, is_overlay = false) => {
			if (!is_overlay) {
				var color = hexToRgb(ctx.fillStyle)
				color.a = parseInt(ctx.globalAlpha * 255) // layer.ctx.globalAlpha
				flood_fill(editor, ctx, parseInt(x), parseInt(y), color)
			}
		},
		move: toolDoNothing,
		end: toolDoNothing
	},
	{
		id: "colorpicker",
		name: "Picker",
		icon: "fa-solid fa-eye-dropper",
		cursor: "url(/media/images/fa-eye-dropper.svg) 0 24, pointer",
		begin: (editor, ctx, x, y, is_overlay = false) => {
			if (!is_overlay) {
				var img_rgb = editor.layers.background.ctx.getImageData(x, y, 1, 1).data
				var drawn_rgb = editor.ctx_current.getImageData(x, y, 1, 1).data
				var drawn_opacity = drawn_rgb[3] / 255
				editor.custom_color_input.value = rgbToHex({ 
					r: (drawn_rgb[0] * drawn_opacity) + (img_rgb[0] * (1 - drawn_opacity)),
					g: (drawn_rgb[1] * drawn_opacity) + (img_rgb[1] * (1 - drawn_opacity)),
					b: (drawn_rgb[2] * drawn_opacity) + (img_rgb[2] * (1 - drawn_opacity)),
				})
				editor.custom_color_input.dispatchEvent(new Event("change"))
			}
		},
		move: toolDoNothing,
		end: toolDoNothing
	}
]

const IMAGE_EDITOR_ACTIONS = [
	{
		id: "load_mask",
		name: "Load mask from file",
		className: "load_mask",
		icon: "fa-regular fa-folder-open",
		handler: (editor) => {
			let el = document.createElement('input')
			el.setAttribute("type", "file")
			el.addEventListener("change", function() {
				if (this.files.length === 0) {
					return
				}

				let reader = new FileReader()
				let file = this.files[0]

				reader.addEventListener('load', function(event) {
					let maskData = reader.result

					editor.layers.drawing.ctx.clearRect(0, 0, editor.width, editor.height)
					var image = new Image()
					image.onload = () => {
						editor.layers.drawing.ctx.drawImage(image, 0, 0, editor.width, editor.height)
					}
					image.src = maskData
				})

				if (file) {
					reader.readAsDataURL(file)
				}
			})

			el.click()
		},
		trackHistory: true
	},
	{
		id: "fill_all",
		name: "Fill all",
		icon: "fa-solid fa-paint-roller",
		handler: (editor) => {
			editor.ctx_current.globalCompositeOperation = "source-over"
			editor.ctx_current.rect(0, 0, editor.width, editor.height)
			editor.ctx_current.fill()
			editor.setBrush()
		},
		trackHistory: true
	},
	{
		id: "clear",
		name: "Clear",
		icon: "fa-solid fa-xmark",
		handler: (editor) => {
			editor.ctx_current.clearRect(0, 0, editor.width, editor.height)
			imageEditor.setImage(null, editor.width, editor.height) // properly reset the drawing canvas
		},
		trackHistory: true
	},
	{
		id: "undo",
		name: "Undo",
		icon: "fa-solid fa-rotate-left",
		handler: (editor) => {
			editor.history.undo()
		},
		trackHistory: false
	},
	{
		id: "redo",
		name: "Redo",
		icon: "fa-solid fa-rotate-right",
		handler: (editor) => {
			editor.history.redo()
		},
		trackHistory: false
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
				span.onclick = function(e) {
					input.click()
				}
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
		options: [ 6, 12, 16, 24, 30, 40, 48, 64 ],
		initElement: (element, option) => {
			element.parentElement.style.flex = option
			element.style.width = option + "px"
			element.style.height = option + "px"
			element.style['margin-right'] = '2px'
			element.style["border-radius"] = (option / 2).toFixed() + "px"
		}
	},
	{
		name: "opacity",
		title: "Opacity",
		default: 0,
		options: [ 0, 0.2, 0.4, 0.6, 0.8 ],
		initElement: (element, option) => {
			element.style.background = `repeating-conic-gradient(rgba(0, 0, 0, ${option}) 0% 25%, rgba(255, 255, 255, ${option}) 0% 50%) 50% / 10px 10px`
		}
	},
	{
		name: "sharpness",
		title: "Sharpness",
		default: 0,
		options: [ 0, 0.05, 0.1, 0.2, 0.3 ],
		initElement: (element, option) => {
			var size = 32
			var blur_amount = parseInt(option * size)
			var sub_element = document.createElement("div")
			sub_element.style.background = `var(--background-color3)`
			sub_element.style.filter = `blur(${blur_amount}px)`
			sub_element.style.width = `${size - 2}px`
			sub_element.style.height = `${size - 2}px`
			sub_element.style['border-radius'] = `${size}px`
			element.style.background = "none"
			element.appendChild(sub_element)
		}
	}
]

class EditorHistory {
	constructor(editor) {
		this.editor = editor
		this.events = [] // stack of all events (actions/edits)
		this.current_edit = null
		this.rewind_index = 0 // how many events back into the history we've rewound to. (current state is just after event at index 'length - this.rewind_index - 1')
	}
	push(event) {
		// probably add something here eventually to save state every x events
		if (this.rewind_index != 0) {
			this.events = this.events.slice(0, 0 - this.rewind_index)
			this.rewind_index = 0
		}
		var snapshot_frequency = 20 // (every x edits, take a snapshot of the current drawing state, for faster rewinding)
		if (this.events.length > 0 && this.events.length % snapshot_frequency == 0) {
			event.snapshot = this.editor.layers.drawing.ctx.getImageData(0, 0, this.editor.width, this.editor.height)
		}
		this.events.push(event)
	}
	pushAction(action) {
		this.push({
			type: "action",
			id: action
		});
	}
	editBegin(x, y) {
		this.current_edit = {
			type: "edit",
			id: this.editor.getOptionValue("tool"),
			options: Object.assign({}, this.editor.options),
			points: [ { x: x, y: y } ]
		}
	}
	editMove(x, y) {
		if (this.current_edit) {
			this.current_edit.points.push({ x: x, y: y })
		}
	}
	editEnd(x, y) {
		if (this.current_edit) {
			this.push(this.current_edit)
			this.current_edit = null
		}
	}
	clear() {
		this.events = []
	}
	undo() {
		this.rewindTo(this.rewind_index + 1)
	}
	redo() {
		this.rewindTo(this.rewind_index - 1)
	}
	rewindTo(new_rewind_index) {
		if (new_rewind_index < 0 || new_rewind_index > this.events.length) {
			return; // do nothing if target index is out of bounds
		}

		var ctx = this.editor.layers.drawing.ctx
		ctx.clearRect(0, 0, this.editor.width, this.editor.height)

		var target_index = this.events.length - 1 - new_rewind_index
		var snapshot_index = target_index
		while (snapshot_index > -1) {
			if (this.events[snapshot_index].snapshot) {
				break
			}
			snapshot_index--
		}

		if (snapshot_index != -1) {
			ctx.putImageData(this.events[snapshot_index].snapshot, 0, 0);
		}

		for (var i = (snapshot_index + 1); i <= target_index; i++) {
			var event = this.events[i]
			if (event.type == "action") {
				var action = IMAGE_EDITOR_ACTIONS.find(a => a.id == event.id)
				action.handler(this.editor)
			}
			else if (event.type == "edit") {
				var tool = IMAGE_EDITOR_TOOLS.find(t => t.id == event.id)
				this.editor.setBrush(this.editor.layers.drawing, event.options)

				var first_point = event.points[0]
				tool.begin(this.editor, ctx, first_point.x, first_point.y)
				for (var point_i = 1; point_i < event.points.length; point_i++) {
					tool.move(this.editor, ctx, event.points[point_i].x, event.points[point_i].y)
				}
				var last_point = event.points[event.points.length - 1]
				tool.end(this.editor, ctx, last_point.x, last_point.y)
			}
		}

		// re-set brush to current settings
		this.editor.setBrush(this.editor.layers.drawing)

		this.rewind_index = new_rewind_index
	}
}

class ImageEditor {
	constructor(popup, inpainter = false) {
		this.inpainter = inpainter
		this.popup = popup
		this.history = new EditorHistory(this)
		if (inpainter) {
			this.popup.classList.add("inpainter")
		}
		this.drawing = false
		this.temp_previous_tool = null // used for the ctrl-colorpicker functionality
		this.container = popup.querySelector(".editor-controls-center > div")
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

		// add mouse handlers
		this.container.addEventListener("mousedown", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseup", this.mouseHandler.bind(this))
		this.container.addEventListener("mousemove", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseout", this.mouseHandler.bind(this))
		this.container.addEventListener("mouseenter", this.mouseHandler.bind(this))

		this.container.addEventListener("touchstart", this.mouseHandler.bind(this))
		this.container.addEventListener("touchmove", this.mouseHandler.bind(this))
		this.container.addEventListener("touchcancel", this.mouseHandler.bind(this))
		this.container.addEventListener("touchend", this.mouseHandler.bind(this))

		// initialize editor controls
		this.options = {}
		this.optionElements = {}
		IMAGE_EDITOR_SECTIONS.forEach(section => {
			section.id = `image_editor_${section.name}`
			var sectionElement = document.createElement("div")
			sectionElement.className = section.id
	
			var title = document.createElement("h4")
			title.innerText = section.title
			sectionElement.appendChild(title)
	
			var optionsContainer = document.createElement("div")
			optionsContainer.classList.add("editor-options-container")
	
			this.optionElements[section.name] = []
			section.options.forEach((option, index) => {
				var optionHolder = document.createElement("div")
				var optionElement = document.createElement("div")
				optionHolder.appendChild(optionElement)
				section.initElement(optionElement, option)
				optionElement.addEventListener("click", target => this.selectOption(section.name, index))
				optionsContainer.appendChild(optionHolder)
				this.optionElements[section.name].push(optionElement)
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
			this.selectOption("opacity", IMAGE_EDITOR_SECTIONS.find(s => s.name == "opacity").options.indexOf(0.4))
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
		var actionsContainer = document.createElement("div")
		var actionsTitle = document.createElement("h4")
		actionsTitle.textContent = "Actions"
		actionsContainer.appendChild(actionsTitle);
		IMAGE_EDITOR_ACTIONS.forEach(action => {
			var element = document.createElement("div")
			var icon = document.createElement("i")
			element.className = "image-editor-button button"
			if (action.className) {
				element.className += " " + action.className
			}
			icon.className = action.icon
			element.appendChild(icon)
			element.append(action.name)
			actionsContainer.appendChild(element)
			element.addEventListener("click", event => this.runAction(action.id))
		})
		this.popup.querySelector(".editor-controls-right").appendChild(actionsContainer)
		this.popup.querySelector(".editor-controls-right").appendChild(buttonContainer)

		this.keyHandlerBound = this.keyHandler.bind(this)

		this.setSize(512, 512)
	}
	show() {
		this.popup.classList.add("active")
		document.addEventListener("keydown", this.keyHandlerBound, true)
		document.addEventListener("keyup", this.keyHandlerBound, true)
	}
	hide() {
		this.popup.classList.remove("active")
		document.removeEventListener("keydown", this.keyHandlerBound, true)
		document.removeEventListener("keyup", this.keyHandlerBound, true)
	}
	setSize(width, height) {
		if (width == this.width && height == this.height) {
			return
		}

		if (width > height) {
		        var max_size = Math.min(parseInt(window.innerWidth * 0.9), width, 768)
			var multiplier = max_size / width
			width = (multiplier * width).toFixed()
			height = (multiplier * height).toFixed()
		}
		else {
		        var max_size = Math.min(parseInt(window.innerHeight * 0.9), height, 768)
			var multiplier = max_size / height
			width = (multiplier * width).toFixed()
			height = (multiplier * height).toFixed()
		}
		this.width = parseInt(width)
		this.height = parseInt(height)
	
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
		this.history.clear()
	}
	get tool() {
		var tool_id = this.getOptionValue("tool")
		return IMAGE_EDITOR_TOOLS.find(t => t.id == tool_id);
	}
	loadTool() {
		this.drawing = false
		this.container.style.cursor = this.tool.cursor;
	}
	setImage(url, width, height) {
		this.setSize(width, height)
		this.layers.background.ctx.clearRect(0, 0, this.width, this.height)
		if (!(url && this.inpainter)) {
			this.layers.drawing.ctx.clearRect(0, 0, this.width, this.height)
		}
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
		this.history.clear()
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
		this.hide()
	}
	getImg() { // a drop-in replacement of the drawingboard version
		return this.layers.drawing.canvas.toDataURL()
	}
	setImg(dataUrl) { // a drop-in replacement of the drawingboard version
		var image = new Image()
		image.onload = () => {
			var ctx = this.layers.drawing.ctx;
			ctx.clearRect(0, 0, this.width, this.height)
			ctx.globalCompositeOperation = "source-over"
			ctx.globalAlpha = 1
			ctx.filter = "none"
			ctx.drawImage(image, 0, 0, this.width, this.height)
			this.setBrush(this.layers.drawing)
		}
		image.src = dataUrl
	}
	runAction(action_id) {
		var action = IMAGE_EDITOR_ACTIONS.find(a => a.id == action_id)
		if (action.trackHistory) {
			this.history.pushAction(action_id)
		}
		action.handler(this)
	}
	setBrush(layer = null, options = null) {
		if (options == null) {
			options = this.options
		}
		if (layer) {
			layer.ctx.lineCap = "round"
			layer.ctx.lineJoin = "round"
			layer.ctx.lineWidth = options.brush_size
			layer.ctx.fillStyle = options.color
			layer.ctx.strokeStyle = options.color
			var sharpness = parseInt(options.sharpness * options.brush_size)
			layer.ctx.filter = sharpness == 0 ? `none` : `blur(${sharpness}px)`
			layer.ctx.globalAlpha = (1 - options.opacity)
			layer.ctx.globalCompositeOperation = "source-over"
			var tool = IMAGE_EDITOR_TOOLS.find(t => t.id == options.tool)
			if (tool && tool.setBrush) {
				tool.setBrush(editor, layer)
			}
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
	keyHandler(event) { // handles keybinds like ctrl+z, ctrl+y
		if (!this.popup.classList.contains("active")) {
			document.removeEventListener("keydown", this.keyHandlerBound)
			document.removeEventListener("keyup", this.keyHandlerBound)
			return // this catches if something else closes the window but doesnt properly unbind the key handler
		}

		// keybindings
		if (event.type == "keydown") {
			if ((event.key == "z" || event.key == "Z") && event.ctrlKey) {
				if (!event.shiftKey) {
					this.history.undo()
				}
				else {
					this.history.redo()
				}
				event.stopPropagation();
				event.preventDefault();
			}
			if (event.key == "y" && event.ctrlKey) {
				this.history.redo()
				event.stopPropagation();
				event.preventDefault();
			}
			if (event.key === "Escape") {
				this.hide()
				event.stopPropagation();
				event.preventDefault();
			}
		}
		
		// dropper ctrl holding handler stuff
		var dropper_active = this.temp_previous_tool != null;
		if (dropper_active && !event.ctrlKey) {
			this.selectOption("tool", IMAGE_EDITOR_TOOLS.findIndex(t => t.id == this.temp_previous_tool))
			this.temp_previous_tool = null
		}
		else if (!dropper_active && event.ctrlKey) {
			this.temp_previous_tool = this.getOptionValue("tool")
			this.selectOption("tool", IMAGE_EDITOR_TOOLS.findIndex(t => t.id == "colorpicker"))
		}
	}
	mouseHandler(event) {
		var bbox = this.layers.overlay.canvas.getBoundingClientRect()
		var x = (event.clientX || 0) - bbox.left
		var y = (event.clientY || 0) - bbox.top
		var type = event.type;
		var touchmap = {
			touchstart: "mousedown",
			touchmove: "mousemove",
			touchend: "mouseup",
			touchcancel: "mouseup"
		}
		if (type in touchmap) {
			type = touchmap[type]
			if (event.touches && event.touches[0]) {
				var touch = event.touches[0]				
				var x = (touch.clientX || 0) - bbox.left
				var y = (touch.clientY || 0) - bbox.top
			}
		}
		event.preventDefault()	
		// do drawing-related stuff
		if (type == "mousedown" || (type == "mouseenter" && event.buttons == 1)) {
			this.drawing = true
			this.tool.begin(this, this.ctx_current, x, y)
			this.tool.begin(this, this.ctx_overlay, x, y, true)
			this.history.editBegin(x, y)
		}
		if (type == "mouseup" || type == "mousemove") {
			if (this.drawing) {
				if (x > 0 && y > 0) {
					this.tool.move(this, this.ctx_current, x, y)
					this.tool.move(this, this.ctx_overlay, x, y, true)
					this.history.editMove(x, y)
				}
			}
		}
		if (type == "mouseup" || type == "mouseout") {
			if (this.drawing) {
				this.drawing = false
				this.tool.end(this, this.ctx_current, x, y)
				this.tool.end(this, this.ctx_overlay, x, y, true)
				this.history.editEnd(x, y)
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
		
		this.optionElements[section_name].forEach(element => element.classList.remove("active"))
		this.optionElements[section_name][option_index].classList.add("active")
	
		// change the editor
		this.setBrush()
		if (section.name == "tool") {
			this.loadTool()
		}
	}
}

const imageEditor = new ImageEditor(document.getElementById("image-editor"))
const imageInpainter = new ImageEditor(document.getElementById("image-inpainter"), true)

imageEditor.setImage(null, 512, 512)
imageInpainter.setImage(null, 512, 512)

document.getElementById("init_image_button_draw").addEventListener("click", () => {
	imageEditor.show()
})
document.getElementById("init_image_button_inpaint").addEventListener("click", () => {
	imageInpainter.show()
})

img2imgUnload() // no init image when the app starts


function rgbToHex(rgb) {
	function componentToHex(c) {
		var hex = parseInt(c).toString(16)
		return hex.length == 1 ? "0" + hex : hex
	}
	return "#" + componentToHex(rgb.r) + componentToHex(rgb.g) + componentToHex(rgb.b)
}

function hexToRgb(hex) {
	var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result ? {
		r: parseInt(result[1], 16),
		g: parseInt(result[2], 16),
		b: parseInt(result[3], 16)
	} : null;
}

function pixelCompare(int1, int2) {
	return Math.abs(int1 - int2) < 4
}

// adapted from https://ben.akrin.com/canvas_fill/fill_04.html
function flood_fill(editor, the_canvas_context, x, y, color) {
	pixel_stack = [{x:x, y:y}] ;
	pixels = the_canvas_context.getImageData( 0, 0, editor.width, editor.height ) ;
	var linear_cords = ( y * editor.width + x ) * 4 ;
	var original_color = {r:pixels.data[linear_cords],
						g:pixels.data[linear_cords+1],
						b:pixels.data[linear_cords+2],
						a:pixels.data[linear_cords+3]} ;
	
	var opacity = color.a / 255;
	var new_color = {
		r: parseInt((color.r * opacity) + (original_color.r * (1 - opacity))),
		g: parseInt((color.g * opacity) + (original_color.g * (1 - opacity))),
		b: parseInt((color.b * opacity) + (original_color.b * (1 - opacity)))
	}

	if ((pixelCompare(new_color.r, original_color.r) &&
		pixelCompare(new_color.g, original_color.g) &&
		pixelCompare(new_color.b, original_color.b)))
	{
		return; // This color is already the color we want, so do nothing
	}
	var max_stack_size = editor.width * editor.height;
	while( pixel_stack.length > 0 && pixel_stack.length < max_stack_size ) {
		new_pixel = pixel_stack.shift() ;
		x = new_pixel.x ;
		y = new_pixel.y ;
	
		linear_cords = ( y * editor.width + x ) * 4 ;
		while( y-->=0 &&
			   (pixelCompare(pixels.data[linear_cords], original_color.r) &&
				pixelCompare(pixels.data[linear_cords+1], original_color.g) &&
				pixelCompare(pixels.data[linear_cords+2], original_color.b))) {
			linear_cords -= editor.width * 4 ;
		}
		linear_cords += editor.width * 4 ;
		y++ ;

		var reached_left = false ;
		var reached_right = false ;
		while( y++<editor.height &&
			   (pixelCompare(pixels.data[linear_cords], original_color.r) &&
				pixelCompare(pixels.data[linear_cords+1], original_color.g) &&
				pixelCompare(pixels.data[linear_cords+2], original_color.b))) {
			pixels.data[linear_cords]   = new_color.r ;
			pixels.data[linear_cords+1] = new_color.g ;
			pixels.data[linear_cords+2] = new_color.b ;
			pixels.data[linear_cords+3] = 255 ;

			if( x>0 ) {
				if( pixelCompare(pixels.data[linear_cords-4], original_color.r) &&
					pixelCompare(pixels.data[linear_cords-4+1], original_color.g) &&
					pixelCompare(pixels.data[linear_cords-4+2], original_color.b)) {
					if( !reached_left ) {
						pixel_stack.push( {x:x-1, y:y} ) ;
						reached_left = true ;
					}
				} else if( reached_left ) {
					reached_left = false ;
				}
			}
		
			if( x<editor.width-1 ) {
				if( pixelCompare(pixels.data[linear_cords+4], original_color.r) &&
					pixelCompare(pixels.data[linear_cords+4+1], original_color.g) &&
					pixelCompare(pixels.data[linear_cords+4+2], original_color.b)) {
					if( !reached_right ) {
						pixel_stack.push( {x:x+1,y:y} ) ;
						reached_right = true ;
					}
				} else if( reached_right ) {
					reached_right = false ;
				}
			}
			
			linear_cords += editor.width * 4 ;
		}
	}
	the_canvas_context.putImageData( pixels, 0, 0 ) ;
}
