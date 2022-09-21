import { style } from "@vanilla-extract/css";

export const CreationPaneMain = style({
  position: "relative",
  width: "100%",
  height: "100%",
  padding: "0 10px",
  overflowY: "auto",
});

export const InpaintingSlider = style({
  position: "absolute",
  top: "10px",
  left: "400px",
  zIndex: 1,
  backgroundColor: "rgba(0, 0, 0, 0.5)",
});
