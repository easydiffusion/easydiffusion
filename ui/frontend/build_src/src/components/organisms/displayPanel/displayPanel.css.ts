import { style } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";

export const displayPanel = style({
  height: "100%",
  display: "flex",
  flexDirection: "column",
  paddingRight: vars.spacing.medium,
});

export const displayContainer = style({
  flexGrow: 1,
  overflow: 'auto',
});

export const previousImages = style({
  minHeight: '250px',
});
