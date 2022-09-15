import { style, globalStyle } from "@vanilla-extract/css";

export const CreationBasicMain = style({
  position: "relative",
  width: "100%",
});

globalStyle(`${CreationBasicMain} > *`, {
  marginBottom: "10px",
});

export const PromptDisplay = style({});

globalStyle(`${PromptDisplay} > p`, {
  fontSize: "1.5em",
  fontWeight: "bold",
  marginBottom: "10px",
});

globalStyle(`${PromptDisplay} > textarea`, {
  fontSize: "1.2em",
  fontWeight: "bold",
  fontFamily: "Arial",
  width: "100%",
  resize: "vertical",
  height: "100px",
});
