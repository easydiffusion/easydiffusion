import { style } from "@vanilla-extract/css";

// @ts-expect-error
import { vars } from "../../../../styles/theme/index.css.ts";

export const StartingStatus = style({
  color: vars.colors.warning,
});

export const ErrorStatus = style({
  color: vars.colors.error,
});

export const SuccessStatus = style({
  color: vars.colors.success,
});
