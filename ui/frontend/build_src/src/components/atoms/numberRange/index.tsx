import React, { ChangeEvent } from "react";

import {
  NumberRangeRoot,
  NumberRangeInput,
} from "./numberRange.css";

interface NumberRangeProps {
  min?: number;
  max?: number;
  step?: number;
  value?: number;
  onChange?: (event: ChangeEvent<HTMLInputElement>) => void;
  label?: string;
  shouldShowValue?: boolean;
}


export default function NumberRange({ min, max, step, value, onChange, label, shouldShowValue }: NumberRangeProps) {

  return (
    <div className={NumberRangeRoot}>
      <label> {label} </label>
      <input
        className={NumberRangeInput}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
      />
      {(shouldShowValue === true) && <span>{value}</span>}
    </div>
  );
}