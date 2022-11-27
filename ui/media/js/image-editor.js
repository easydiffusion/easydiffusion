var editorRoot = document.getElementById("image-editor")
var editorControls = document.getElementById("image-editor-controls")

const IMAGE_EDITOR_MAX_SIZE = 800;

var IMAGE_EDITOR_SECTIONS = [
	{
		name: "color",
		title: "Color",
		default: "#f1c232",
		options: [
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
			element.style.background = option
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
		default: 0,
		options: [ 0, 2.5, 5, 7.5, 10 ],
		initElement: (element, option) => {
			var sub_element = document.createElement("div");
			sub_element.style.background = `var(--background-color3)`
			sub_element.style.filter = `blur(${option}px)`
			sub_element.style.width = "28px"
			sub_element.style.height = "28px"
			sub_element.style['border-radius'] = "32px"
			element.style.background = "none"
			element.appendChild(sub_element)
		}
	}
]

class ImageEditor {
	constructor(container) {
		this.drawing = false
		this.container = container
		this.layers = {};
		var layer_names = [
			"drawing",
			"overlay"
		];
		layer_names.forEach(name => {
			let canvas = document.createElement("canvas");
			this.container.appendChild(canvas);
			this.layers[name] = {
				name: name,
				canvas: canvas,
				ctx: canvas.getContext("2d")
			};
		})

		this.setSize(512, 512)

		// add mouse handlers
		this.container.addEventListener("mousedown", this.mouseHandler.bind(this));
		this.container.addEventListener("mouseup", this.mouseHandler.bind(this));
		this.container.addEventListener("mousemove", this.mouseHandler.bind(this));
		this.container.addEventListener("mouseout", this.mouseHandler.bind(this));
		this.container.addEventListener("mouseenter", this.mouseHandler.bind(this));

		// initialize editor controls
		IMAGE_EDITOR_SECTIONS.forEach(section => {
			section.id = `image_editor_${section.name}`
			var sectionElement = document.createElement("div")
			sectionElement.id = section.id
	
			var title = document.createElement("h4");
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
				optionElement.addEventListener("click", target => this.selectOption(section.name, index));
				optionsContainer.appendChild(optionHolder);
				section.optionElements.push(optionElement);
			})
			this.selectOption(section.name, section.options.indexOf(section.default))
	
			sectionElement.appendChild(optionsContainer)
	
			editorControls.appendChild(sectionElement)
		});
	}
	setSize(width, height) {
		var max_size = Math.min(window.innerWidth, IMAGE_EDITOR_MAX_SIZE)
	
		if (width > height) {
			var multiplier = max_size / width;
			width = (multiplier * width).toFixed();
			height = (multiplier * height).toFixed();
		}
		else {
			var multiplier = max_size / height;
			width = (multiplier * width).toFixed();
			height = (multiplier * height).toFixed();
		}
		this.width = width;
		this.height = height;
	
		this.container.style.width = width + "px"
		this.container.style.height = height + "px"
		
		Object.values(this.layers).forEach(layer => {
			layer.canvas.width = width
			layer.canvas.height = height
		})

		this.setBrush()
	}
	setImage(url, width, height) {
		this.container.style.backgroundImage = `url('${url}')`
		this.setSize(width, height)
	}
	setBrush() {
		Object.values(this.layers).forEach(layer => {
			layer.ctx.lineCap = "round"
			layer.ctx.lineJoin = "round"
			layer.ctx.lineWidth = this.getOptionValue("brush_size");
			layer.ctx.fillStyle = this.getOptionValue("color");
			layer.ctx.strokeStyle = this.getOptionValue("color");
			var sharpness = this.getOptionValue("sharpness");
			layer.ctx.filter = sharpness == 0 ? `none` : `blur(${sharpness}px)`;
			layer.ctx.globalAlpha = (1 - this.getOptionValue("opacity"));
		})
	}
	get ctx_overlay() {
		return this.layers.overlay.ctx;
	}
	get ctx_current() {
		return this.layers.drawing.ctx;
	}
	mouseHandler(event) {
		var bbox = this.layers.overlay.canvas.getBoundingClientRect();
		var x = event.clientX - bbox.left;
		var y = event.clientY - bbox.top;
	
		if (event.type == "mousedown" || (event.type == "mouseenter" && event.buttons == 1)) {
			this.drawing = true;
			this.ctx_overlay.beginPath();
			this.ctx_overlay.moveTo(x, y);
			this.ctx_current.beginPath();
			this.ctx_current.moveTo(x, y);
		}
		if (event.type == "mouseup" || event.type == "mousemove") {
			if (this.drawing) {
				this.ctx_overlay.lineTo(x, y);
				this.ctx_current.lineTo(x, y);
				this.ctx_overlay.clearRect(0, 0, this.width, this.height);
				this.ctx_overlay.stroke();
			}
		}
		if (event.type == "mouseup" || event.type == "mouseout") {
			if (this.drawing) {
				this.drawing = false;
				this.ctx_current.stroke();
				this.ctx_overlay.clearRect(0, 0, this.width, this.height);
			}
		}
	}
	getOptionValue(section_name) {
		var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name);
		return section.value === undefined ? section.default : section.value;
	}
	selectOption(section_name, option_index) {
		var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name)
		var value = section.options[option_index]
		section.value = value
	
		section.optionElements.forEach(element => element.classList.remove("active"))
		section.optionElements[option_index].classList.add("active")
	
		// change the editor
		if (["color", "brush_size", "sharpness", "opacity"].includes(section_name)) {
			this.setBrush()
		}
	}
	doThing() {
		console.time("clearing");
		for(var i = 0; i < 1000; i++) {
			this.ctx_overlay.clearRect(0, 0, this.width, this.height);
			this.ctx_overlay.stroke();
			this.ctx_overlay.drawImage(this.layers.drawing.canvas, 0, 0); // CAN USE THIS FOR ERASING
		}
		console.timeEnd("clearing");

	}
}

const imageEditor = new ImageEditor(document.getElementById("image-editor-canvas"));



// var drawingBoardElement = document.getElementById("image-editor-canvas canvas")
// var drawingBoardElement = document.getElementById("image-editor-canvas")
// var editorCanvas = drawingBoardElement.querySelector("canvas")
// var editorContext = editorCanvas.getContext("2d")
// drawingBoardElement.style.width = IMAGE_EDITOR_MAX_SIZE + "px"
// drawingBoardElement.style.height = IMAGE_EDITOR_MAX_SIZE + "px"

