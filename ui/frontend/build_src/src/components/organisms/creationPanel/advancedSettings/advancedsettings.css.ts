import { style, globalStyle } from "@vanilla-extract/css";

// @ts-ignore
import { vars } from "../../../../styles/theme/index.css.ts";

export const AdvancedSettingsList = style({
  fontSize: vars.fonts.sizes.Body,
  marginBottom: vars.spacing.small,
  paddingLeft: vars.spacing.medium,
  listStyleType: "none",
});

export const AdvancedSettingItem = style({
  paddingBottom: vars.spacing.small,
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
  marginBottom: vars.spacing.small,
});

globalStyle(`${MenuButton}> h4`, {
  color: "#e7ba71",
  marginTop: "5px !important",
});
