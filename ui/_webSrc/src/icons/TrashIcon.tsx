import React from "react";
import { IconProps } from "./index";

export function TrashIcon(props: IconProps) {
  const { label } = props;

  const aria = label ? { "aria-label": label } : { "aria-hidden": true };

  return (
    <span {...aria} className="icon">
      &#128465;
    </span>
  );
}
