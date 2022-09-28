import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../../../styles/theme/index.css";

export const StopContainer = style({
  display: "flex",
  width: "100%",
  marginTop: vars.spacing.medium,
});

globalStyle(`${StopContainer} button`, {
  flexGrow: 1,
});

globalStyle(`${StopContainer} button:first-child`, {
  marginRight: vars.spacing.small,
});
