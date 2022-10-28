

/**
 * Enum of parameter types
 * @readonly
 * @enum {string}
 */
 var ParameterType = {
    checkbox: "checkbox",
	select: "select",
	text: "text",
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
		label: "Automatically Save Images",
		default: false,
	},
	{
		id: "diskPath", // TODO: auto-disabling of this based on save_to_disk
		type: ParameterType.text,
		label: "Save Location",
		default: "", // Note: default value is generated
	},
	{
		id: "default_vae_model",
		type: ParameterType.select, // Note: options generated dynamically
		label: "Default VAE",
	},
	{
		id: "sound_toggle",
		type: ParameterType.checkbox,
		label: "Play sound on task completion",
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
		label: "Use full precision",
		note: "for GPU-only. warning: this will consume more VRAM",
		default: false,
	},
	{
		id: "auto_save_settings",
		type: ParameterType.checkbox,
		label: "Automatically save settings",
		note: "settings restored on browser load",
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