import { style } from "@vanilla-extract/css";

export const generatedImage = style({
  position: "relative",
  width: "512px",
  height: "512px",
});

export const imageContain = style({
  width: "512px",
  height: "512px",
  backgroundColor: "black",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
});

export const image = style({
  width: "512px",
  height: "512px",
  objectFit: "contain",
});

export const saveButton = style({
  position: "absolute",
  bottom: "10px",
  left: "10px",
});

export const useButton = style({
  position: "absolute",
  bottom: "10px",
  right: "10px",
});
