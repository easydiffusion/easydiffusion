import React, { Fragment } from "react";
import { Switch } from '@headlessui/react'

import {
  IconFont,
} from "../../../styles/shared.css";

import {
  CheckMain,
  CheckContent,
  CheckInner
} from "./checkbox.css";

interface CheckboxProps {
  isChecked: boolean;
  label: string;
  isLabelFirst?: boolean;
  disabled?: boolean;
  toggleCheck: (isChecked: boolean) => void;
}

export default function Checkbox({ isChecked, label, isLabelFirst, toggleCheck, disabled }: CheckboxProps) {

  const handChange = (checked: boolean) => {
    if (disabled !== true) {
      toggleCheck(checked);
    }
  };

  return (
    <Switch.Group as={Fragment}>
      <div className={CheckMain}
        data-disabled={disabled}
      >
        {/* TODO Make the lable first logic more eligant? */}
        {(isLabelFirst === true) && <Switch.Label> {label} </Switch.Label>}
        <Switch className={CheckContent} checked={isChecked} onChange={handChange}>
          <div
            className={CheckInner}
          >
            {isChecked
              ? <i className={[IconFont, 'fa-solid', 'fa-check'].join(" ")}></i>
              : <i className={[IconFont, 'fa-solid', 'fa-x'].join(" ")}></i>}
          </div>
        </Switch>
        {(isLabelFirst !== true) && <Switch.Label> {label} </Switch.Label>}
      </div>
    </Switch.Group>
  );
}