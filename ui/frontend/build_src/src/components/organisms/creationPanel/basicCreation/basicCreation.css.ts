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

export const PromptDisplay = style({});

globalStyle(`${CreationBasicMain} > *`, {
  marginBottom: '10px'
});
