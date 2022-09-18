import { style, globalStyle } from "@vanilla-extract/css";

// @ts-ignore
import { vars } from "../../../../styles/theme/index.css.ts";

export const completedImagesMain = style({
  display: "flex",
  flexDirection: "row",
  flexWrap: "nowrap",
  height: "100%",
  width: "100%",
  overflow: "auto",
  paddingBottom: vars.spacing.medium,
});

export const imageContain = style({
  width: "206px",
  backgroundColor: "black",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  flexShrink: 0,
  border: "0 none",
  padding: "0",
  marginLeft: vars.spacing.medium,
  cursor: "pointer",
});

globalStyle(`${imageContain} img`, {
  width: "100%",
  objectFit: "contain",
});

globalStyle(`${completedImagesMain} > ${imageContain}:first-of-type`, {
  marginLeft: vars.spacing.medium,
});

globalStyle(`${imageContain} > ${imageContain}:last-of-type`, {
  marginRight: 0,
});
