import React, { useEffect, useState } from "react";
import { useImageCreate } from "../../../../../../stores/imageCreateStore";
import HeadlessListbox, { listBoxOption } from "../../../../../atoms/headlessListbox";

import {
  IconFont,
} from "../../../../../../styles/shared.css";


const options: listBoxOption[] = [
  {
    id: 1,
    value: 'jpeg',
    display: 'jpeg',
    unavailable: false
  },
  {
    id: 2,
    value: 'png',
    display: 'png',
    unavailable: false
  },
]


export default function fileOptions() {
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const [fileOption, setFileOption] = useState(options[0]);
  const fileValue = useImageCreate((state) => state.getValueForRequestKey("output_format"));

  const handleChange = (option: listBoxOption) => {
    setRequestOption("output_format", option.value);
  }

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/strict-boolean-expressions
    if (fileValue) {
      const foundOption = options.find((option) => option.value === fileValue)
      if (foundOption != null) {
        setFileOption(foundOption);
      }
    }
    else {
      setFileOption(options[0]);
    }
  }, [fileValue]);

  const FAIcon = [IconFont, 'fa-solid', 'fa-chevron-down'].join(" ");

  return (
    <HeadlessListbox
      options={options}
      currentOption={fileOption}
      handleChange={handleChange}
      label={'Select File Format'}
      FAIcon={FAIcon}
    ></HeadlessListbox>
  );
}