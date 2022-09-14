import React, { useEffect } from "react";
import { useImageCreate } from "../../../store/imageCreateStore";
import "./advancedSettings.css";

// todo: move this someplace more global
const IMAGE_DIMENSIONS = [
  { value: 128, label: "128 (*)" },
  { value: 192, label: "192" },
  { value: 256, label: "256 (*)" },
  { value: 320, label: "320" },
  { value: 384, label: "384" },
  { value: 448, label: "448" },
  { value: 512, label: "512 (*)" },
  { value: 576, label: "576" },
  { value: 640, label: "640" },
  { value: 704, label: "704" },
  { value: 768, label: "768 (*)" },
  { value: 832, label: "832" },
  { value: 896, label: "896" },
  { value: 960, label: "960" },
  { value: 1024, label: "1024 (*)" },
];

function SettingsList() {
  const parallelCount = useImageCreate((state) => state.parallelCount);
  const setParallelCount = useImageCreate((state) => state.setParallelCount);
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const toggleUseFaceCorrection = useImageCreate(
    (state) => state.toggleUseFaceCorrection
  );
  
  const isUsingFaceCorrection = useImageCreate((state) =>
    state.isUsingFaceCorrection()
  );

  const toggleUseUpscaling = useImageCreate(
    (state) => state.toggleUseUpscaling
  );
  const isUsingUpscaling = useImageCreate((state) => state.isUsingUpscaling());

  const toggleUseRandomSeed = useImageCreate(
    (state) => state.toggleUseRandomSeed
  );
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());

  const toggleUseAutoSave = useImageCreate((state) => state.toggleUseAutoSave);
  const isUseAutoSave = useImageCreate((state) => state.isUseAutoSave());

  const toggleSoundEnabled = useImageCreate(
    (state) => state.toggleSoundEnabled
  );
  const isSoundEnabled = useImageCreate((state) => state.isSoundEnabled());

  const use_upscale = useImageCreate((state) =>
    state.getValueForRequestKey("use_upscale")
  );
  const show_only_filtered_image = useImageCreate((state) =>
    state.getValueForRequestKey("show_only_filtered_image")
  );
  const seed = useImageCreate((state) => state.getValueForRequestKey("seed"));
  const width = useImageCreate((state) => state.getValueForRequestKey("width"));
  const num_outputs = useImageCreate((state) =>
    state.getValueForRequestKey("num_outputs")
  );
  const height = useImageCreate((state) =>
    state.getValueForRequestKey("height")
  );
  const steps = useImageCreate((state) =>
    state.getValueForRequestKey("num_inference_steps")
  );
  const guidance_scale = useImageCreate((state) =>
    state.getValueForRequestKey("guidance_scale")
  );
  const prompt_strength = useImageCreate((state) =>
    state.getValueForRequestKey("prompt_strength")
  );
  const save_to_disk_path = useImageCreate((state) =>
    state.getValueForRequestKey("save_to_disk_path")
  );
  const turbo = useImageCreate((state) => state.getValueForRequestKey("turbo"));
  const use_cpu = useImageCreate((state) =>
    state.getValueForRequestKey("use_cpu")
  );
  const use_full_precision = useImageCreate((state) =>
    state.getValueForRequestKey("use_full_precision")
  );

  return (
    <ul id="editor-settings-entries">
      {/*IMAGE CORRECTION */}
      <li>
        <label>
          <input
            type="checkbox"
            checked={isUsingFaceCorrection}
            onChange={(e) => toggleUseFaceCorrection()}
          />
          Fix incorrect faces and eyes (uses GFPGAN)
        </label>
      </li>
      <li>
        <label>
          <input
            type="checkbox"
            checked={isUsingUpscaling}
            onChange={(e) => toggleUseUpscaling()}
          />
          Upscale the image to 4x resolution using
          <select
            id="upscale_model"
            name="upscale_model"
            disabled={!isUsingUpscaling}
            value={use_upscale}
            onChange={(e) => {
              setRequestOption("use_upscale", e.target.value);
            }}
          >
            <option value="RealESRGAN_x4plus">RealESRGAN_x4plus</option>
            <option value="RealESRGAN_x4plus_anime_6B">
              RealESRGAN_x4plus_anime_6B
            </option>
          </select>
        </label>
      </li>
      <li>
        <label>
          <input
            type="checkbox"
            checked={show_only_filtered_image}
            onChange={(e) =>
              setRequestOption("show_only_filtered_image", e.target.checked)
            }
          />
          Show only filtered image
        </label>
      </li>
      {/* SEED */}
      <li>
        <label>
          Seed:
          <input
            size={10}
            value={seed}
            onChange={(e) => setRequestOption("seed", e.target.value)}
            disabled={isRandomSeed}
            placeholder="random"
          />
        </label>
        <label>
          <input
            type="checkbox"
            checked={isRandomSeed}
            onChange={(e) => toggleUseRandomSeed()}
          />{" "}
          Random Image
        </label>
      </li>
      {/* COUNT */}
      <li>
        <label>
          Number of images to make:{" "}
          <input
            type="number"
            value={num_outputs}
            onChange={(e) =>
              setRequestOption("num_outputs", parseInt(e.target.value, 10))
            }
            size={4}
          />
        </label>
        <label>
          Generate in parallel:
          <input
            type="number"
            value={parallelCount}
            onChange={(e) => setParallelCount(parseInt(e.target.value, 10))}
            size={4}
          />
        </label>
      </li>
      {/* DIMENTIONS */}
      <li>
        <label>
          Width:
          <select
            value={width}
            onChange={(e) => setRequestOption("width", e.target.value)}
          >
            {IMAGE_DIMENSIONS.map((dimension) => (
              <option
                key={"width-option_" + dimension.value}
                value={dimension.value}
              >
                {dimension.label}
              </option>
            ))}
          </select>
        </label>
      </li>
      <li>
        <label>
          Height:
          <select
            value={height}
            onChange={(e) => setRequestOption("height", e.target.value)}
          >
            {IMAGE_DIMENSIONS.map((dimension) => (
              <option
                key={"height-option_" + dimension.value}
                value={dimension.value}
              >
                {dimension.label}
              </option>
            ))}
          </select>
        </label>
      </li>
      {/* STEPS */}
      <li>
        <label>
          Number of inference steps:{" "}
          <input
            value={steps}
            onChange={(e) => {
              setRequestOption("num_inference_steps", e.target.value);
            }}
            size={4}
          />
        </label>
      </li>
      {/* GUIDANCE SCALE */}
      <li>
        <label>
          Guidance Scale:
          <input
            value={guidance_scale}
            onChange={(e) => setRequestOption("guidance_scale", e.target.value)}
            type="range"
            min="0"
            max="20"
            step=".1"
          />
        </label>
        <span>{guidance_scale}</span>
      </li>
      {/* PROMPT STRENGTH */}
      <li className="mb-4">
        <label>
          Prompt Strength:{" "}
          <input
            value={prompt_strength}
            onChange={(e) =>
              // setImageOptions({ promptStrength: Number(e.target.value) })
              setRequestOption("prompt_strength", e.target.value)
            }
            type="range"
            min="0"
            max="1"
            step=".05"
          />
        </label>
        <span>{prompt_strength}</span>
      </li>
      {/* AUTO SAVE */}
      <li>
        <label>
          <input
            checked={isUseAutoSave}
            onChange={(e) => toggleUseAutoSave()}
            type="checkbox"
          />
          Automatically save to{" "}
        </label>
        <label>
          <input
            value={save_to_disk_path}
            onChange={(e) =>
              setRequestOption("save_to_disk_path", e.target.value)
            }
            size={40}
            disabled={!isUseAutoSave}
          />
          <span className="visually-hidden">
            Path on disk where images will be saved
          </span>
        </label>
      </li>
      {/* SOUND */}
      <li>
        <label>
          <input
            checked={isSoundEnabled}
            onChange={(e) => toggleSoundEnabled()}
            type="checkbox"
          />
          Play sound on task completion
        </label>
      </li>
      {/* GENERATE */}
      <li>
        <label>
          <input
            checked={turbo}
            onChange={(e) => setRequestOption("turbo", e.target.checked)}
            type="checkbox"
          />
          Turbo mode (generates images faster, but uses an additional 1 GB of
          GPU memory)
        </label>
      </li>
      <li>
        <label>
          <input
            type="checkbox"
            checked={use_cpu}
            onChange={(e) => setRequestOption("use_cpu", e.target.checked)}
          />
          Use CPU instead of GPU (warning: this will be *very* slow)
        </label>
      </li>
      <li>
        <label>
          <input
            checked={use_full_precision}
            onChange={(e) =>
              setRequestOption("use_full_precision", e.target.checked)
            }
            type="checkbox"
          />
          Use full precision (for GPU-only. warning: this will consume more
          VRAM)
        </label>
      </li>
    </ul>
  );
}

//  {/* <!-- <li><input id="allow_nsfw" name="allow_nsfw" type="checkbox"/> <label htmlFor="allow_nsfw">Allow NSFW Content (You confirm you are above 18 years of age)</label></li> --> */}

export default function AdvancedSettings() {
  const advancedSettingsIsOpen = useImageCreate(
    (state) => state.uiOptions.advancedSettingsIsOpen
  );

  const toggleAdvancedSettingsIsOpen = useImageCreate(
    (state) => state.toggleAdvancedSettingsIsOpen
  );

  return (
    <div className="panel-box">
      <button
        type="button"
        onClick={toggleAdvancedSettingsIsOpen}
        className="panel-box-toggle-btn"
      >
        <h4>Advanced Settings</h4>
      </button>
      {advancedSettingsIsOpen && <SettingsList />}
    </div>
  );
}
