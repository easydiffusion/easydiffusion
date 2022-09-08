import React from "react";
import { IconProps } from "./index";

export interface StarIconProps extends IconProps {
  filled: boolean;
}

export function StarIcon(props: StarIconProps) {
  const { label, filled } = props;

  const aria = label ? { "aria-label": label } : { "aria-hidden": true };

  return (
    <span {...aria} className="icon">
      {filled ? <>&#9733;</> : <>&#9734;</>}
    </span>
  );
}
