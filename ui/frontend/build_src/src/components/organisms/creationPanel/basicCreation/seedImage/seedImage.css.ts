import { style } from "@vanilla-extract/css";

import { vars } from "../../../../../styles/theme/index.css";

export const ImageInputDisplay = style({
  display: "flex",
});

export const InputLabel = style({
  marginBottom: vars.spacing.small,
  display: "block",
});

export const ImageInput = style({
  display: "none",
});


// this is needed to fix an issue with the image input text
// when that is a drag an drop we can remove this
export const ImageFixer = style({
  marginLeft: "20px",
});
