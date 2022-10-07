import { style, globalStyle } from "@vanilla-extract/css";

export const generatedImageMain = style({
  position: "relative",
});

globalStyle(`${generatedImageMain} img`, {
  width: "100%",
  height: "100%",
  objectFit: "contain",
});
