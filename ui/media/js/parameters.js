

/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
 var ParameterType = {
    checkbox: "checkbox",
	select: "select",
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
			return `<input id="${parameter.id}" name="${parameter.id}" size="40" disabled>`
		}
	},
	{
		id: "default_vae_model",
		type: ParameterType.select, // Note: options generated dynamically
		label: "Default VAE",
	},
	{
		id: "sound_toggle",
		type: ParameterType.checkbox,
		label: "Enable Sound",
		note: "plays a sound on task completion",
		default: true,
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
		label: "Use CPU instead of GPU",
		note: "warning: this will be *very* slow",
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


function getParameterElement(parameter) {
	switch (parameter.type) {
		case ParameterType.checkbox:
			var is_checked = parameter.default ? " checked" : "";
			return `<input id="${parameter.id}" name="${parameter.id}"${is_checked} type="checkbox">`
		case ParameterType.select:
			var options = (parameter.options || []).map(option => `<option value="${option.value}">${option.label}</option>`).join("")
			return `<select id="${parameter.id}" name="${parameter.id}">${options}</select>`
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
		var newrow = `<tr>
			<td><label for="${parameter.id}">${parameter.label}</label></td>
			<td><div>${element}${note}<div></td></tr>`
		parametersTable.insertAdjacentHTML("beforeend", newrow)
	})
}

initParameters();

