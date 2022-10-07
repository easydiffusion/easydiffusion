import React from "react";

import {
  NumberInputMain,
  NumberInputLabel,
  NumberInputInput,
} from "./numberInput.css";

interface NumberInputProps {
  value: number,
  onChange: (value: number) => void,
  min?: number,
  max?: number,
  step?: number,
  label: string,
  disabled?: boolean,
}

export default function NumberInput(props: NumberInputProps) {
  const {
    value,
    onChange,
    min,
    max,
    step,
    label,
    disabled,
  } = props;

  return (
    <div className={NumberInputMain}>
      <label className={NumberInputLabel}>{label}</label>
      <input
        className={NumberInputInput}
        type="number"
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
      />
    </div>
  );
}

