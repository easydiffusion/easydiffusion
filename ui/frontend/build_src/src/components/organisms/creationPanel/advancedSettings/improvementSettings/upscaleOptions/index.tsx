import React from "react";

import { useImageCreate, SAMPLER_OPTIONS } from "../../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../../creationPanelUIStore";

import {
  ListboxHeadless
} from "../../../../../_recipes/listbox_headless.css";


export default function UpscaleOptions() {
  // const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  // const sampler = useImageCreate((state) => state.getValueForRequestKey("sampler"));
  // const samplerOptions = useImageCreate((state) => state.getValueForRequestKey("sampler_options"));

  // const setSampler = (sampler: string) => {
  //   setRequestOption("sampler", sampler);
  // };

  // const setSamplerOption = (key: string, value: any) => {
  //   const newOptions = { ...samplerOptions, [key]: value };
  //   setRequestOption("sampler_options", newOptions);
  // };

  return (
    <div>
      UPSCALE
      {/* <ListboxHeadless
        label="Sampler"
        value={sampler}
        options={SAMPLER_OPTIONS}
        onChange={setSampler}
      /> */}

      {/* <label>
              {t("settings.ups")}
              <select
                id="upscale_model"
                name="upscale_model"
                value={useUpscale}
                onChange={(e) => {
                  setRequestOption("use_upscale", e.target.value);
                }}
              >
                <option value="">{t("settings.no-ups")}</option>
                <option value="RealESRGAN_x4plus">RealESRGAN_x4plus</option>
                <option value="RealESRGAN_x4plus_anime_6B">
                  RealESRGAN_x4plus_anime_6B
                </option>
              </select>
            </label> */}

    </div>
  );
}
