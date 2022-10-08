/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useImageCreate, SAMPLER_OPTIONS } from "../../../../../../stores/imageCreateStore";

import {
  IconFont,
} from "../../../../../../styles/shared.css";

import HeadlessListbox, { listBoxOption } from "../../../../../atoms/headlessListbox";

const samplerList: listBoxOption[] = SAMPLER_OPTIONS.map((sample, index) => {
  return {
    id: index,
    value: sample,
    display: sample,
    unavailable: false,
  };
})


export default function SamplerOptions() {

  const { t } = useTranslation();
  const setRequestOption = useImageCreate((state) => state.setRequestOptions);

  const sampler = useImageCreate((state) =>
    state.getValueForRequestKey("sampler")
  );
  const [selectedSampleOption, setSelectedSampleOption] = useState(samplerList[0])


  useEffect(() => {

    console.log('SAMPLER USEEFFECT', sampler);
    if (sampler) {
      const sampleoption = samplerList.find((option) => option.value === sampler)
      if (sampleoption) {
        setSelectedSampleOption(sampleoption);
      }
    }
    else {
      setSelectedSampleOption(samplerList[0]);
    }
  }, [sampler]);

  // this type is not correct // SamplerOptionsProps
  const handleChange = (option: listBoxOption) => {
    setRequestOption("sampler", option.value);
  };

  const FAIcon = [IconFont, 'fa-solid', 'fa-chevron-down'].join(" ");


  return (
    // <div className={ListboxHeadless}>
    //   <Listbox value={selectedSampleOption} onChange={handleChange}>
    //     <Listbox.Label className={ListboxHeadlessLabel}>{t("settings.sampler")}</Listbox.Label>
    //     <div style={{ display: 'inline-block', }}>
    //       <Listbox.Button
    //         className={ListboxHeadlessButton}>
    //         {selectedSampleOption.value}
    //         <i className={[ListBoxIcon, IconFont, 'fa-solid', 'fa-chevron-down'].join(" ")}></i>
    //       </Listbox.Button>
    //       <Listbox.Options className={ListboxHeadlessOptions}>
    //         {samplerList.map((sample) => (
    //           <Listbox.Option
    //             // className={ListboxHeadlessOption}
    //             key={sample.id}
    //             value={sample.value}
    //             disabled={sample.unavailable}
    //             as={Fragment}
    //           >
    //             {({ active, selected }) => {
    //               // console.log('active', active);
    //               // console.log('selected', selected);
    //               return (
    //                 <li
    //                   className={ListboxHeadlessOptionItem}
    //                 // data-selected={selected}
    //                 >
    //                   {sample.value}
    //                 </li>
    //               )
    //             }}
    //           </Listbox.Option>
    //         ))}
    //       </Listbox.Options>
    //     </div>
    //   </Listbox>
    // </div>
    <HeadlessListbox
      options={samplerList}
      label={t("settings.sampler")}
      currentOption={selectedSampleOption}
      handleChange={handleChange}
      FAIcon={FAIcon}
    />
  );
};
