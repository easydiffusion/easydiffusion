import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../../styles/theme/index.css";

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


export const ModifierListStyle = style({
  paddingLeft: 0,
  listStyleType: "none",
  display: "flex",
  flexWrap: "wrap",
});

globalStyle(`${ModifierListStyle} li`, {
  margin: 0,
});
