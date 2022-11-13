

/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
 var ParameterType = {
    checkbox: "checkbox",
	select: "select",
	select_multiple: "select_multiple",
	custom: "custom",
};

/**
 * JSDoc style
 * @typedef {object} Parameter
 * @property {string} id
 * @property {ParameterType} type
 * @property {string} label
 * @property {?string} note
 * @property {number|boolean|string} default
 */


/** @type {Array.<Parameter>} */
var PARAMETERS = [
	{
		id: "theme",
		type: ParameterType.select,
		label: "Theme",
		default: "theme-default",
		options: [ // Note: options expanded dynamically
			{
				value: "theme-default",
				label: "Default"
			}
		]
	},
	{
		id: "save_to_disk",
		type: ParameterType.checkbox,
		label: "Auto-Save Images",
		note: "automatically saves images to the specified location",
		default: false,
	},
	{
		id: "diskPath",
		type: ParameterType.custom,
		label: "Save Location",
		render: (parameter) => {
			return `<input id="${parameter.id}" name="${parameter.id}" size="30" disabled>`
		}
	},
	{
		id: "sound_toggle",
		type: ParameterType.checkbox,
		label: "Enable Sound",
		note: "plays a sound on task completion",
		default: true,
	},
	{
		id: "process_order_toggle",
		type: ParameterType.checkbox,
		label: "Process newest jobs first",
		note: "reverse the normal processing order",
		default: false,
	},
	{
		id: "turbo",
		type: ParameterType.checkbox,
		label: "Turbo Mode",
		default: true,
		note: "generates images faster, but uses an additional 1 GB of GPU memory",
	},
	{
		id: "use_cpu",
		type: ParameterType.checkbox,
		label: "Use CPU (not GPU)",
		note: "warning: this will be *very* slow",
		default: false,
	},
	{
		id: "use_gpus",
		type: ParameterType.select_multiple,
		label: "GPUs to use",
		note: "select multiple GPUs to process in parallel",
		default: false,
	},
	{
		id: "use_full_precision",
		type: ParameterType.checkbox,
		label: "Use Full Precision",
		note: "for GPU-only. warning: this will consume more VRAM",
		default: false,
	},
	{
		id: "auto_save_settings",
		type: ParameterType.checkbox,
		label: "Auto-Save Settings",
		note: "restores settings on browser load",
		default: true,
	},
	{
		id: "use_beta_channel",
		type: ParameterType.checkbox,
		label: "🔥Beta channel",
		note: "Get the latest features immediately (but could be less stable). Please restart the program after changing this.",
		default: false,
	},
];

function getParameterSettingsEntry(id) {
	let parameter = PARAMETERS.filter(p => p.id === id)
	if (parameter.length === 0) {
		return
	}
	return parameter[0].settingsEntry
}

function getParameterElement(parameter) {
	switch (parameter.type) {
		case ParameterType.checkbox:
			var is_checked = parameter.default ? " checked" : "";
			return `<input id="${parameter.id}" name="${parameter.id}"${is_checked} type="checkbox">`
		case ParameterType.select:
		case ParameterType.select_multiple:
			var options = (parameter.options || []).map(option => `<option value="${option.value}">${option.label}</option>`).join("")
			var multiple = (parameter.type == ParameterType.select_multiple ? 'multiple' : '')
			return `<select id="${parameter.id}" name="${parameter.id}" ${multiple}>${options}</select>`
		case ParameterType.custom:
			return parameter.render(parameter)
		default:
			console.error(`Invalid type for parameter ${parameter.id}`);
			return "ERROR: Invalid Type"
	}
}

var parametersTable = document.querySelector("#system-settings table")
/* fill in the system settings popup table */
function initParameters() {
	PARAMETERS.forEach(parameter => {
		var element = getParameterElement(parameter)
		var note = parameter.note ? `<small>${parameter.note}</small>` : "";
		var newrow = document.createElement('tr')
		newrow.innerHTML = `
			<td><label for="${parameter.id}">${parameter.label}</label></td>
			<td><div>${element}${note}<div></td>`
		parametersTable.appendChild(newrow)
		parameter.settingsEntry = newrow
	})
}

initParameters();

