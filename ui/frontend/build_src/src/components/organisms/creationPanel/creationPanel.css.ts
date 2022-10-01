import { style, globalStyle } from "@vanilla-extract/css";
import { vars } from "../../../styles/theme/index.css";
export const CreationPaneMain = style({
  position: "relative",
  width: "100%",
  height: "100%",
  overflowY: "auto",
  overflowX: "hidden",
});

globalStyle(`${CreationPaneMain} > div`, {
  marginBottom: vars.spacing.medium,
});

export const InpaintingSlider = style({
  position: "absolute",
  top: "10px",
  left: "400px",
  zIndex: 1,
  backgroundColor: "rgba(0, 0, 0, 0.5)",
});
