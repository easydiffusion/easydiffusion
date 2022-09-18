import { globalStyle } from "@vanilla-extract/css";
// @ts-ignore
import { vars } from "./theme/index.css.ts";

// baisc body style
globalStyle("body", {
  margin: 0,
  minWidth: "320px",
  minHeight: "100vh",
  fontFamily: vars.fonts.body,
});

// single page style
globalStyle("#root", {
  position: "absolute",
  top: 0,
  left: 0,
  width: "100vw",
  height: "100vh",
  overflow: "hidden",
});

// border box all
globalStyle(`*`, {
  boxSizing: "border-box",
});

/** RESETS */
globalStyle(`p, h3, h4`, {
  margin: 0,
});

globalStyle(`textarea`, {
  margin: 0,
  padding: 0,
  border: "none",
});
