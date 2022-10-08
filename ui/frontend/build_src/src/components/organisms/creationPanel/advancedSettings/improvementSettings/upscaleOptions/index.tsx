/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useImageCreate } from "../../../../../../stores/imageCreateStore";

import HeadlessListbox, { listBoxOption } from "../../../../../atoms/headlessListbox";

import {
  IconFont,
} from "../../../../../../styles/shared.css";

const options: listBoxOption[] = [
  { id: 1, value: null, display: 'No Upscaling', unavailable: false },
  { id: 2, value: 'RealESRGAN_x4plus', display: 'RealESRGAN_x4plus', unavailable: false },
  { id: 3, value: 'RealESRGAN_x4plus_anime_6B', display: 'RealESRGAN_x4plus_anime_6B', unavailable: false },
]

export default function UpscaleOptions() {
  const { t } = useTranslation();
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const [selectedUpscaleOption, setSelectedUpscaleOption] = useState(options[0])
  const upscaleValue = useImageCreate((state) => state.getValueForRequestKey("use_upscale"));

  useEffect(() => {
    if (upscaleValue) {
      const upscaleOption = options.find((option) => option.value === upscaleValue)
      if (upscaleOption) {
        setSelectedUpscaleOption(upscaleOption);
      }
    }
    else {
      setSelectedUpscaleOption(options[0]);
    }
  }, [upscaleValue]);

  const handleChange = (option: listBoxOption) => {
    setRequestOption("use_upscale", option.value);
  };


  const FAIcon = [IconFont, 'fa-solid', 'fa-chevron-down'].join(" ");

  return (

    <HeadlessListbox
      options={options}
      currentOption={selectedUpscaleOption}
      handleChange={handleChange}
      label={t("settings.ups")}
      FAIcon={FAIcon}
    ></HeadlessListbox>

  );
}