import { style, globalStyle } from "@vanilla-extract/css";
// @ts-ignore
import { vars } from "./theme/index.css.ts";

export const PanelBox = style({
  background: vars.colors.backgroundAlt,
  color: vars.colors.text.normal,
  padding: vars.spacing.medium,
  borderRadius: vars.trim.smallBorderRadius,
  marginBottom: vars.spacing.medium,
  // TODO move this to the theme
  boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.15), 0 6px 20px 0 rgba(0, 0, 0, 0.15)",
});



globalStyle(`${PanelBox} .panel-box-toggle-btn`, {
  display: "block",
  width: "100%",
  textAlign: "left",
  backgroundColor: "transparent",
  color: vars.colors.text.normal,
  border: "0 none",
  cursor: "pointer",
  padding: "0",
});

export const SettingItem = style({
  marginBottom: vars.spacing.medium,
});
