import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../../../styles/theme/index.css";

export const CreationActionMain = style({
  display: "flex",
  flexDirection: "column",
  width: "100%",
  marginTop: vars.spacing.medium,
});

globalStyle(`${CreationActionMain} button`, {
  marginBottom: vars.spacing.medium,
});