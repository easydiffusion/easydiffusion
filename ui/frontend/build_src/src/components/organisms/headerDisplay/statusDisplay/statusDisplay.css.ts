import { style } from "@vanilla-extract/css";

import { vars } from "../../../../styles/theme/index.css";

export const StartingStatus = style({
  color: `hsl(${vars.warningHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

export const ErrorStatus = style({
  color: `hsl(${vars.errorHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});

export const SuccessStatus = style({
  color: `hsl(${vars.successHue}, ${vars.colorMod.saturation.normal}, ${vars.colorMod.lightness.normal})`,
});
