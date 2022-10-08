import React, { useEffect, useState } from "react";
import { useImageCreate } from "../../../../../../stores/imageCreateStore";
import HeadlessListbox, { listBoxOption } from "../../../../../atoms/headlessListbox";


import {
  IconFont,
} from "../../../../../../styles/shared.css";


const options: listBoxOption[] = [
  {
    id: 1,
    value: 'sd-v1-4',
    display: 'sd-v1-4',
    unavailable: false
  },
]

export default function ModelOptions() {
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const [modelOption, setModelOption] = useState(options[0]);
  const modelValue = useImageCreate((state) => state.getValueForRequestKey("use_stable_diffusion_model"));


  const handleChange = (option: listBoxOption) => {
    setRequestOption("use_stable_diffusion_model", option.value);
  }


  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (modelValue) {
      const foundOption = options.find((option) => option.value === modelValue)
      if (foundOption != null) {
        setModelOption(modelOption);
      }
    }
    else {
      setModelOption(options[0]);
    }
  }, [modelValue]);

  const FAIcon = [IconFont, 'fa-solid', 'fa-chevron-down'].join(" ");

  return (
    <HeadlessListbox
      options={options}
      currentOption={modelOption}
      handleChange={handleChange}
      label={'Select Model'}
      FAIcon={FAIcon}
    ></HeadlessListbox>

    // TODO : some sort of file upload for the model
  );
}