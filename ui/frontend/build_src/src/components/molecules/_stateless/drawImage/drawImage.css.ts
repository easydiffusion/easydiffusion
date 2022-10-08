import { style, globalStyle } from "@vanilla-extract/css";

export const DrawImageMain = style({
  position: "relative",
});

globalStyle(`${DrawImageMain} > canvas`, {
  position: "absolute",
  top: "0",
  left: "0",
  width: "100%",
  height: "100%",
});

globalStyle(`${DrawImageMain} > canvas:first-of-type`, {
  opacity: ".7",
});

globalStyle(`${DrawImageMain} > img`, {
  top: "0",
  left: "0",
});
