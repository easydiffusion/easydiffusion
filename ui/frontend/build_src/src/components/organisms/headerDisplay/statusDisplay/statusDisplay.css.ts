import { style } from "@vanilla-extract/css";

import { vars } from "../../../../styles/theme/index.css";

export const StartingStatus = style({
  color: vars.colors.warning,
});

export const ErrorStatus = style({
  color: vars.colors.error,
});

export const SuccessStatus = style({
  color: vars.colors.success,
});
