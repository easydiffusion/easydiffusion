import { style, globalStyle } from "@vanilla-extract/css";


export const DrawImageMain = style({
  position: "relative",
  width: "512px",
  height: "512px",
});

globalStyle(`${DrawImageMain} > canvas`, {
  position: "absolute",
  top: "0",
  left: "0",
  opacity: "0.5",
});

globalStyle(`${DrawImageMain} > img`, {
  position: "absolute",
  top: "0",
  left: "0",
});