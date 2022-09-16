import { style } from "@vanilla-extract/css";

export const displayPanel = style({
  padding: "10px",
  // width: '512px',
  // height: '512px',
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
  marginLeft: "30px",
  display: "flex",
  flex: "auto",
  flexWrap: "wrap",
});

export const previousImage = style({
  margin: "0 10px",
});
