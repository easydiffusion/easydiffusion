/* eslint-disable @typescript-eslint/strict-boolean-expressions */
import React, { Fragment, useEffect, useState } from "react";
import { Listbox } from '@headlessui/react'
import { useTranslation } from "react-i18next";
import { useImageCreate, SAMPLER_OPTIONS } from "../../../../../../stores/imageCreateStore";

import { useCreateUI } from "../../../creationPanelUIStore";

import {
  IconFont,
} from "../../../../../../styles/shared.css";

import {
  ListboxHeadless,
  ListboxHeadlessButton,
  ListBoxIcon,
  ListboxHeadlessLabel,
  ListboxHeadlessOptions,
  ListboxHeadlessOptionItem,
} from "../../../../../_recipes/listbox_headless.css";

interface SamplerOptionsProps {
  id: number,
  value: string,
  unavailable: boolean,
}

const samplerList = SAMPLER_OPTIONS.map((sample) => {
  return {
    id: sample,
    value: sample,
    unavailable: false,
  }
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

  const handleChange = (option: SamplerOptionsProps) => {
    setRequestOption("sampler", option);
  };


  return (
    <div className={ListboxHeadless}>
      <Listbox value={selectedSampleOption} onChange={handleChange}>
        <Listbox.Label className={ListboxHeadlessLabel}>{t("settings.sampler")}</Listbox.Label>
        <div style={{ display: 'inline-block', }}>
          <Listbox.Button
            className={ListboxHeadlessButton}>
            {selectedSampleOption.value}
            <i className={[ListBoxIcon, IconFont, 'fa-solid', 'fa-chevron-down'].join(" ")}></i>
          </Listbox.Button>
          <Listbox.Options className={ListboxHeadlessOptions}>
            {samplerList.map((sample) => (
              <Listbox.Option
                // className={ListboxHeadlessOption}
                key={sample.id}
                value={sample.value}
                disabled={sample.unavailable}
                as={Fragment}
              >
                {({ active, selected }) => {
                  // console.log('active', active);
                  // console.log('selected', selected);
                  return (
                    <li
                      className={ListboxHeadlessOptionItem}
                    // data-selected={selected}
                    >
                      {sample.value}
                    </li>
                  )
                }}
              </Listbox.Option>
            ))}
          </Listbox.Options>
        </div>
      </Listbox>
    </div>);
};

            // {/* <label>
            //   {t("settings.sampler")}
            //   <select
            //     value={sampler}
            //     onChange={(e) => setRequestOption("sampler", e.target.value)}
            //   >
            //     {SAMPLER_OPTIONS.map((sampler) => (
            //       <option key={`sampler-option_${sampler}`} value={sampler}>
            //         {sampler}
            //       </option>
            //     ))}
            //   </select>
            // </label> */}