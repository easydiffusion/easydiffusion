import React from "react";
import { useImageCreate } from "@stores/imageCreateStore";
import { useCreateUI } from "../../creationPanelUIStore";

import Checkbox from "@atoms/headlessCheckbox";
import NumberInput from "@atoms/numberInput";


import SamplerOptions from "./samplerOptions";
import GuidanceScale from "./guidanceScale";
import SizeSelection from "./sizeSelection";

import {
  SettingItem,
} from "@styles/shared.css";

import {
  buttonStyle,
} from "../../../../_recipes/button.css";

import { useTranslation } from "react-i18next";

export default function PropertySettings() {
  const { t } = useTranslation();

  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const toggleUseRandomSeed = useImageCreate(
    (state) => state.toggleUseRandomSeed
  );
  const isRandomSeed = useImageCreate((state) => state.isRandomSeed());

  const seed = useImageCreate((state) => state.getValueForRequestKey("seed"));
  const steps = useImageCreate((state) =>
    state.getValueForRequestKey("num_inference_steps")
  );

  const initImage = useImageCreate((state) =>
    state.getValueForRequestKey("init_image")
  );

  const promptStrength = useImageCreate((state) =>
    state.getValueForRequestKey("prompt_strength")
  );

  const propertyOpen = useCreateUI((state) => state.isOpenAdvPropertySettings);
  const togglePropertyOpen = useCreateUI(
    (state) => state.toggleAdvPropertySettings
  );

  return (
    <div>
      <button type="button" className={buttonStyle({
        type: 'action',
        color: 'accent',
      })} onClick={togglePropertyOpen}>
        Property Settings
      </button>
      {propertyOpen && (
        <>
          <div className={SettingItem}>

            <Checkbox
              label="Random Image"
              isChecked={isRandomSeed}
              toggleCheck={toggleUseRandomSeed}
            ></Checkbox>

            <NumberInput
              label="Seed:"
              value={seed}
              onChange={(value) => setRequestOption("seed", value)}
              disabled={isRandomSeed}
            ></NumberInput>

          </div>

          <div className={SettingItem}>
            <NumberInput
              label={t("settings.steps")}
              value={steps}
              onChange={(value) => {
                setRequestOption("num_inference_steps", value);
              }}
            ></NumberInput>
          </div>

          <div className={SettingItem}>
            <GuidanceScale></GuidanceScale>
          </div>

          {void 0 !== initImage && (
            <div className={SettingItem}>
              <label>
                {t("settings.prompt-str")}{" "}
                <input
                  value={promptStrength}
                  onChange={(e) =>
                    setRequestOption("prompt_strength", e.target.value)
                  }
                  type="range"
                  min="0"
                  max="1"
                  step=".05"
                />
              </label>
              <span>{promptStrength}</span>
            </div>
          )}

          <div className={SettingItem}>
            <SizeSelection></SizeSelection>
          </div>

          <div className={SettingItem}>
            <SamplerOptions></SamplerOptions>
          </div>
        </>
      )}
    </div>
  );
}
