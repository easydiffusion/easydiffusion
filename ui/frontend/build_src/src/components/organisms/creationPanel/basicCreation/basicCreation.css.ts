import { style, globalStyle } from "@vanilla-extract/css";

import { card } from "../../../_recipes/card.css";


export const CreationBasicMain = style([
  card({
    backing: 'normal',
    level: 1
  }), {
    position: "relative",
    width: "100%",
  }]
);

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
  width: "100%",
  resize: "vertical",
  height: "100px",
});
