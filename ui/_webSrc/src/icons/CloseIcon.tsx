import React from "react";
import { IconProps } from "./index";

export function CloseIcon(props: IconProps) {
	const {label} = props;

	const aria = label ? {'aria-label': label} : {'aria-hidden': true};

	return (
		<span {...aria} className="icon">
			&times;
		</span>
	)
}
