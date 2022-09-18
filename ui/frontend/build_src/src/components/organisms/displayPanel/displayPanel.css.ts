import { style } from "@vanilla-extract/css";

// @ts-ignore
import { vars } from "../../../styles/theme/index.css.ts";

export const displayPanel = style({
  height: "100%",
  display: "flex",
  flexDirection: "column",
});

export const displayContainer = style({
  flexGrow: 1,
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
});

export const previousImages = style({
  // height: "150px",
});
