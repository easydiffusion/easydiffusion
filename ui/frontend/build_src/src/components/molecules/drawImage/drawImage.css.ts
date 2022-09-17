import { style, globalStyle } from "@vanilla-extract/css";

export const DrawImageMain = style({
  position: "relative",
});

globalStyle(`${DrawImageMain} > canvas`, {
  position: "absolute",
  top: "0",
  left: "0",
  opacity: ".5",
});

globalStyle(`${DrawImageMain} > img`, {
  top: "0",
  left: "0",
});
