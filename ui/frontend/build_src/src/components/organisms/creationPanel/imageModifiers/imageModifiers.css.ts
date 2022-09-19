import { style, globalStyle } from "@vanilla-extract/css";
// @ts-ignore
import { vars } from "../../../../styles/theme/index.css.ts";

export const ImagerModifierGroups = style({
  // marginBottom: vars.spacing.small,
  paddingLeft: 0,
  listStyleType: "none",
});

globalStyle(`${ImagerModifierGroups} li`, {
  marginTop: vars.spacing.medium,
});

export const ImageModifierGrouping = style({
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

export const ModifierListStyle = style({
  // marginBottom: vars.spacing.small,
  paddingLeft: 0,
  listStyleType: "none",
  display: "flex",
  flexWrap: "wrap",
});

globalStyle(`${ModifierListStyle} li`, {
  margin: 0,
});
