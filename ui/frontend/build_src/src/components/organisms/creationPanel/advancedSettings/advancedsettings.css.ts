import { style, globalStyle } from "@vanilla-extract/css";

import { vars } from "../../../../styles/theme/index.css";

// import { PanelBox } from "../../../../styles/shared.css.ts";

export const AdvancedSettingsList = style({
  // marginBottom: vars.spacing.small,
  paddingLeft: 0,
  listStyleType: "none",
});

export const AdvancedSettingGrouping = style({
  marginTop: vars.spacing.medium,
});

export const MenuButton = style({
  display: "block",
  width: "100%",
  textAlign: "left",
  backgroundColor: "transparent",
  color: vars.colors.text.normal,
  border: "0 none",
  cursor: "pointer",
  padding: "0",
  marginBottom: vars.spacing.medium,
});

globalStyle(`${MenuButton}> h4`, {
  color: "#e7ba71",
});
