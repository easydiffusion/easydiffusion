import { style } from "@vanilla-extract/css";

import { vars } from "../../../../../styles/theme/index.css";

import { BrandedButton } from "../../../../../styles/shared.css";

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

export const ImageInputButton = style([BrandedButton]);

// this is needed to fix an issue with the image input text
// when that is a drag an drop we can remove this
export const ImageFixer = style({
  marginLeft: "20px",
});

// just a 1 off component for now
// dont bother bringing in line with the rest of the app
export const XButton = style({
  position: "absolute",
  transform: "translateX(-50%) translateY(-35%)",
  background: "black",
  color: "white",
  border: "2pt solid #ccc",
  padding: "0",
  cursor: "pointer",
  outline: "inherit",
  borderRadius: "8pt",
  width: "16pt",
  height: "16pt",
  fontFamily: "Verdana",
  fontSize: "8pt",
});
