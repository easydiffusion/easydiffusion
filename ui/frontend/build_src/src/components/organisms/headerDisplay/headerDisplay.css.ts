import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";

export const HeaderDisplayMain = style({
  color: vars.colors.text.normal,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
});

globalStyle(`${HeaderDisplayMain} > h1`, {
  fontSize: vars.fonts.sizes.Title,
  fontWeight: "bold",
  marginRight: vars.spacing.medium,
});
