import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";

export const HeaderDisplayMain = style({
  color: vars.colors.text.normal,
  display: "flex",
  justifyContent: "space-between"
});

globalStyle(`${HeaderDisplayMain} > h1`, {
  fontSize: vars.fonts.sizes.Title,
  fontWeight: "bold",
  marginRight: vars.spacing.medium,
});


export const HeaderTitle = style({
  marginLeft: vars.spacing.large,
});

export const HeaderLinks = style({
  display: "flex",
  alignItems: "center",
  flexGrow: 1,
  justifyContent: "space-between",
  maxWidth: "300px",
  marginRight: vars.spacing.large,
});