import { style, globalStyle } from "@vanilla-extract/css";

export const HeaderDisplayMain = style({
  color: "#ffffff",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
});

globalStyle(`${HeaderDisplayMain} > h1`, {
  fontSize: "1.5em",
  fontWeight: "bold",
  marginRight: "10px",
});

