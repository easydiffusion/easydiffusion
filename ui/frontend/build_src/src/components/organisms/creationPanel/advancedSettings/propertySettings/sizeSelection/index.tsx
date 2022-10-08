import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import HeadlessListbox, { listBoxOption } from "../../../../../atoms/headlessListbox";
import { useImageCreate } from "../../../../../../stores/imageCreateStore";

import {
  IconFont,
} from "../../../../../../styles/shared.css";


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

const optionList: listBoxOption[] = IMAGE_DIMENSIONS.map((dimention, index) => {

  return {
    id: index,
    value: dimention.value,
    display: dimention.label,
    unavailable: false,
  };
});


export default function SizeSelection() {
  const { t } = useTranslation();
  const width = useImageCreate((state) => state.getValueForRequestKey("width"));
  const height = useImageCreate((state) =>
    state.getValueForRequestKey("height")
  );

  const [selectedWidth, setSelectedWidth] = useState(optionList[6]);
  const [selectedHeight, setSelectedHeight] = useState(optionList[6]);

  useEffect(() => {
    // using the ! to tell typescript that we know that the value is not null
    const widthOption = optionList.find((option) => option.value === width)
    const heightOption = optionList.find((option) => option.value === height)

    if (widthOption != null) {
      setSelectedWidth(widthOption);
    }

    if (heightOption != null) {
      setSelectedHeight(heightOption);
    }

  }, [width, height]);


  const setRequestOption = useImageCreate((state) => state.setRequestOptions);
  const handleWidthChange = (option: listBoxOption) => {
    setRequestOption("width", option.value);
  };

  const handleHeightChange = (option: listBoxOption) => {
    setRequestOption("height", option.value);
  };


  const FAIcon = [IconFont, 'fa-solid', 'fa-chevron-down'].join(" ");
  return (
    <div style={{ display: 'flex' }}>
      <HeadlessListbox
        label={t("settings.width")}
        options={optionList}
        currentOption={selectedWidth}
        handleChange={handleWidthChange}
        FAIcon={FAIcon}
      ></HeadlessListbox>
      <HeadlessListbox
        label={t("settings.height")}
        options={optionList}
        currentOption={selectedHeight}
        handleChange={handleHeightChange}
        FAIcon={FAIcon}
      ></HeadlessListbox>
    </div>
  );


};