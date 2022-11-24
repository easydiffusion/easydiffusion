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
		default: 0.25,
		options: [ 1, 0.25, 0 ],
		initElement: (element, option) => {
			var percent = (option * 100).toFixed()
			element.style.background = `radial-gradient(var(--background-color3) 0%, var(--background-color3) ${percent}%, var(--background-color1) 100%)`
		}
	}
]

function getOptionValue(section_name) {
	var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name);
	return section.value === undefined ? section.default : section.value;
}

function selectOption(section_name, option_index) {
	var section = IMAGE_EDITOR_SECTIONS.find(s => s.name == section_name)
	var value = section.options[option_index]
	section.value = value

	section.optionElements.forEach(element => element.classList.remove("active"))
	section.optionElements[option_index].classList.add("active")

	// change the editor
	if (["color", "brush_size", "sharpness", "opacity"].includes(section_name)) {
		setBrush()
	}
}

function initImageEditor() {
	editorCanvas.addEventListener("mousedown", canvasMouseHandler);
	editorCanvas.addEventListener("mouseup", canvasMouseHandler);
	editorCanvas.addEventListener("mousemove", canvasMouseHandler);
	editorCanvas.addEventListener("mouseout", canvasMouseHandler);
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
			optionElement.addEventListener("click", target => selectOption(section.name, index));
			optionsContainer.appendChild(optionHolder);
			section.optionElements.push(optionElement);
		})
		selectOption(section.name, section.options.indexOf(section.default))

		sectionElement.appendChild(optionsContainer)

		editorControls.appendChild(sectionElement)
	});
	setBrush();
}

// var drawingBoardElement = document.getElementById("image-editor-canvas canvas")
var drawingBoardElement = document.getElementById("image-editor-canvas")
var editorCanvas = drawingBoardElement.querySelector("canvas")
var editorContext = editorCanvas.getContext("2d")
drawingBoardElement.style.width = IMAGE_EDITOR_MAX_SIZE + "px"
drawingBoardElement.style.height = IMAGE_EDITOR_MAX_SIZE + "px"
// var editorDrawingBoard = new DrawingBoard.Board(drawingBoardElement.id, {
//     color: "#ffffff",
//     background: false,
//     size: 30,
//     webStorage: false,
//     controls: [{"DrawingMode": {"filler": false}}, "Size", "Navigation"]
// })

function setImageEditorImage(url, width, height) {
	drawingBoardElement.style.backgroundImage = `url('${url}')`

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

	// editorDrawingBoard.opts.aspectRatio = (width / height).toFixed(3)

	drawingBoardElement.style.width = width + "px"
	drawingBoardElement.style.height = height + "px"
	
	editorCanvas.width = width
	editorCanvas.height = height
	// editorDrawingBoard.opts.enlargeYourContainer = true
    // editorDrawingBoard.opts.size = inpaintingEditor.ctx.lineWidth
	// editorDrawingBoard.resize()
	
	setBrush()
}

function setBrush() {
	editorContext.lineCap = "round"
	editorContext.lineJoin = "round"
	editorContext.lineWidth = getOptionValue("brush_size");
	editorContext.fillStyle = getOptionValue("color");
	editorContext.strokeStyle = getOptionValue("color");
	var sharpness = getOptionValue("sharpness");
	editorContext.filter = sharpness == 1 ? `none` : `blur(${10}px)`;
	editorContext.globalAlpha = (1 - getOptionValue("opacity"));
}

var DRAWING = false;
function canvasMouseHandler(event) {
	var bbox = editorCanvas.getBoundingClientRect();
	var x = event.clientX - bbox.left;
	var y = event.clientY - bbox.top;

	if (event.type == "mousedown") {
		DRAWING = true;
		editorContext.beginPath();
		editorContext.moveTo(x, y);
	}
	if (event.type == "mouseup" || event.type == "mousemove") {
		if (DRAWING) {
			editorContext.lineTo(x, y);
			editorContext.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
			editorContext.stroke();
		}
	}
	if (event.type == "mouseup" || event.type == "mouseout") {
		DRAWING = false;
	}
}

initImageEditor()
setBrush()
