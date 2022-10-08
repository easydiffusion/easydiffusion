import React, { Fragment } from "react";
import { Listbox } from '@headlessui/react'

import {
  ListboxHeadless,
  ListboxHeadlessButton,
  ListBoxIcon,
  ListboxHeadlessLabel,
  ListboxHeadlessOptions,
  ListboxHeadlessOptionItem,
} from "./listbox_headless.css";


export interface listBoxOption {
  id: number,
  value: string | number | null,
  display: string,
  unavailable: boolean,
}

interface ListBoxProps {
  options: listBoxOption[],
  currentOption: listBoxOption,
  handleChange: (option: listBoxOption) => void,
  label: string,
  // depends on font awesome icon set
  FAIcon: string,
}

export default function HeadlessListbox(props: ListBoxProps) {
  const {
    options,
    currentOption,
    handleChange,
    label,
    FAIcon,
  } = props;

  return (
    <div className={ListboxHeadless}>
      <Listbox value={currentOption} onChange={handleChange}>
        <Listbox.Label className={ListboxHeadlessLabel}>{label}</Listbox.Label>
        <div style={{ display: 'inline-block' }}>
          <Listbox.Button
            className={ListboxHeadlessButton}>
            {currentOption.display}
            <i className={[ListBoxIcon, FAIcon].join(" ")}></i>
          </Listbox.Button>

          <Listbox.Options className={ListboxHeadlessOptions}>

            {options.map((option) => (
              <Listbox.Option
                key={option.id}
                value={option}
                disabled={option.unavailable}
                as={Fragment}
              >
                {({ active, selected }) => {

                  return (
                    <li
                      className={ListboxHeadlessOptionItem}
                    >
                      {option.display}
                    </li>
                  )
                }}
              </Listbox.Option>
            ))}
          </Listbox.Options>
        </div>
      </Listbox >
    </div >
  );
};