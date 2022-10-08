import { style } from "@vanilla-extract/css";

import { vars } from "../../../../styles/theme/index.css";

export const AdvancedSettingsList = style({
  paddingLeft: 0,
  listStyleType: "none",
});

export const AdvancedSettingGrouping = style({
  marginTop: vars.spacing.small,
});
