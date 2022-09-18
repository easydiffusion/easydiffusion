import { style } from "@vanilla-extract/css";

// @ts-ignore
import { vars } from "../../../styles/theme/index.css.ts";

export const displayPanel = style({
  padding: vars.spacing.medium,
});

export const displayContainer = style({
  display: "flex",
  flexDirection: "row",
  height: "100%",
  width: "100%",
  overflow: "hidden",
});

export const CurrentDisplay = style({
  width: "512px",
  height: "100%",
});

export const previousImages = style({
  marginLeft: vars.spacing.large,
  display: "flex",
  flex: "auto",
  flexWrap: "wrap",
});

export const previousImage = style({
  margin: `0 ${vars.spacing.small}`,
});
