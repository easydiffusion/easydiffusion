import React, { useState } from "react";
import { useImageCreate } from "../../../../../stores/imageCreateStore";
import { useCreateUI } from "../../creationPanelUIStore";

import {
  MenuButton, //@ts-ignore
} from "../advancedsettings.css.ts";
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

export default function PropertySettings() {
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const toggleUseRandomSeed = useImageCreate(
    (state) => state.toggleUseRandomSeed
  );
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());

  const seed = useImageCreate((state) => state.getValueForRequestKey("seed"));
  const steps = useImageCreate((state) =>
    state.getValueForRequestKey("num_inference_steps")
  );
  const guidance_scale = useImageCreate((state) =>
    state.getValueForRequestKey("guidance_scale")
  );
  const prompt_strength = useImageCreate((state) =>
    state.getValueForRequestKey("prompt_strength")
  );
  const width = useImageCreate((state) => state.getValueForRequestKey("width"));
  const height = useImageCreate((state) =>
    state.getValueForRequestKey("height")
  );

  const propertyOpen = useCreateUI((state) => state.isOpenAdvPropertySettings);
  const togglePropertyOpen = useCreateUI(
    (state) => state.toggleAdvPropertySettings
  );

  return (
    <div>
      <button type="button" className={MenuButton} onClick={togglePropertyOpen}>
        <h4>Property Settings</h4>
      </button>
      {propertyOpen && (
        <>
          <div>
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
          </div>

          <div>
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
          </div>

          <div>
            <label>
              Guidance Scale:
              <input
                value={guidance_scale}
                onChange={(e) =>
                  setRequestOption("guidance_scale", e.target.value)
                }
                type="range"
                min="0"
                max="20"
                step=".1"
              />
            </label>
            <span>{guidance_scale}</span>
          </div>

          <div className="mb-4">
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
          </div>

          <div>
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
          </div>

          <div>
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
          </div>
        </>
      )}
    </div>
  );
}
